"""
Training loop driver — gradient accumulation, grad norm, auto-checkpoint.
Works with the portable pipeline (StageWorker list).
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass

import torch

from meshgpu.backends.portable.pipeline import pipeline_train_step, sync_tied_parameters
from meshgpu.backends.portable.stage_worker import StageWorker  # noqa: F401 (used in type hints)
from meshgpu.checkpoints.coordinator import CheckpointCoordinator

log = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    max_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    checkpoint_every_steps: int = 100
    log_every_steps: int = 10
    ignore_index: int = -100

    def __post_init__(self) -> None:
        for name, value in (
            ("max_steps", self.max_steps),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
            ("checkpoint_every_steps", self.checkpoint_every_steps),
            ("log_every_steps", self.log_every_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise TypeError("ignore_index must be an integer")


@dataclass
class StepResult:
    step: int
    loss: float
    grad_norm: float
    n_valid_tokens: int
    elapsed_s: float
    tokens_per_second: float
    checkpoint_id: str | None = None
    overflow: bool = False


class PortableTrainer:
    """
    Drives training over a portable pipeline.
    Handles gradient accumulation, global grad norm, overflow detection
    and periodic checkpointing.
    """

    def __init__(
        self,
        workers: list[StageWorker],
        optimizers: list[torch.optim.Optimizer],
        schedulers: list | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
        checkpoint: CheckpointCoordinator | None = None,
        cfg: TrainerConfig | None = None,
        job_id: str = "local",
    ) -> None:
        if not workers:
            raise ValueError("workers must not be empty")
        if len(workers) != len(optimizers):
            raise ValueError(
                f"need one optimizer per worker, got {len(optimizers)} for {len(workers)} workers"
            )
        if schedulers is not None and len(schedulers) not in (0, len(workers)):
            raise ValueError(
                "schedulers must be empty or contain one scheduler per worker"
            )
        self._workers = workers
        self._opts = optimizers
        self._scheds = schedulers or []
        self._scaler = scaler
        self._ckpt = checkpoint
        self._cfg = cfg or TrainerConfig()
        self._job_id = job_id
        self._global_step = 0
        self._accum_loss: float = 0.0
        self._accum_tokens: int = 0
        self._op_seq = 0
        # Number of input batches consumed at or before the current logical
        # optimizer boundary.  The CLI uses this to recreate a packed JSONL
        # iterator after a process restart.
        self._data_cursor = 0
        self._data_cursor_exact = True

    def _next_op(self) -> int:
        self._op_seq += 1
        return self._op_seq

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def train(
        self,
        data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
    ) -> Iterable[StepResult]:
        """
        Iterate over (input_ids, labels) batches. Yields StepResult per optimizer step.
        Caller controls the data iterator; this driver handles accumulation.
        Accumulation groups containing no valid labels are skipped.
        """
        cfg = self._cfg
        t_step_start = time.time()
        data_iter = iter(data_iter)

        while self._global_step < cfg.max_steps:
            group_cursor = self._data_cursor
            micro_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
            try:
                for _ in range(cfg.gradient_accumulation_steps):
                    try:
                        micro_batches.append(next(data_iter))
                    except StopIteration:
                        break
                    self._data_cursor += 1
            except BaseException:
                # A data-loader failure must not advertise batches that were
                # consumed after the last durable optimizer boundary.  The
                # caller can construct a fresh iterator from group_cursor.
                self._data_cursor = group_cursor
                raise
            if not micro_batches:
                break

            try:
                total_valid_tokens = sum(
                    int((labels != cfg.ignore_index).sum().item())
                    for _, labels in micro_batches
                )
            except BaseException:
                self._data_cursor = group_cursor
                self._discard_pending_training()
                raise
            if total_valid_tokens == 0:
                log.debug(
                    "skipping accumulation group at step %d: no valid labels",
                    self._global_step,
                )
                self._discard_pending_training()
                continue

            try:
                for micro_index, (input_ids, labels) in enumerate(micro_batches):
                    attempt_id = f"s{self._global_step}_a{micro_index}"
                    info = pipeline_train_step(
                        self._workers,
                        input_ids,
                        labels,
                        [_noop_opt(optimizer) for optimizer in self._opts],
                        operation_id=self._next_op(),
                        attempt_id=attempt_id,
                        ignore_index=cfg.ignore_index,
                        loss_normalizer=total_valid_tokens,
                        gradient_scaler=self._scaler,
                        step_optimizers=False,  # trainer owns clip+step
                    )
                    self._accum_loss += float(info["loss"])
                    self._accum_tokens += int(info["n_valid_tokens"])
            except BaseException:
                # A later microbatch can fail after earlier microbatches have
                # already accumulated gradients.  Those gradients belong to
                # an abandoned logical step and must never leak into a retry.
                self._data_cursor = group_cursor
                self._discard_pending_training()
                raise

            # --- Optimizer step boundary ---
            overflow = False
            if self._scaler is not None:
                for opt in self._opts:
                    self._scaler.unscale_(opt)
            grad_norm = self._clip_grad_norm()
            # GradScaler tracks ``found_inf`` per optimizer.  Calling
            # ``scaler.step`` independently can therefore update an early
            # stage while a later stage is overflowing.  A pipeline step is
            # one logical update: make the overflow decision globally before
            # touching any optimizer.
            overflow = not math.isfinite(grad_norm) or not self._all_gradients_finite()

            if self._scaler is not None:
                if not overflow:
                    for opt in self._opts:
                        self._scaler.step(opt)
                self._scaler.update()
                if overflow:
                    log.warning(
                        "step %d: overflow, all optimizer steps were skipped",
                        self._global_step,
                    )
            else:
                if not overflow:
                    for opt in self._opts:
                        opt.step()

            # Tied embeddings are replicated when the embedding and lm-head
            # live on different stages.  The pipeline helper can synchronize
            # around its own optimizer boundary, but this trainer owns the
            # real optimizer step (the pipeline receives no-op wrappers), so
            # refresh the head from the embedding after every trainer step.
            if not overflow:
                sync_tied_parameters(self._workers)

            for opt in self._opts:
                opt.zero_grad(set_to_none=True)
            if not overflow:
                for sched in self._scheds:
                    sched.step()

            elapsed = time.time() - t_step_start
            tps = self._accum_tokens / elapsed if elapsed > 0 else 0.0

            self._global_step += 1
            result = StepResult(
                step=self._global_step,
                loss=self._accum_loss,
                grad_norm=grad_norm,
                n_valid_tokens=self._accum_tokens,
                elapsed_s=elapsed,
                tokens_per_second=tps,
                overflow=overflow,
            )

            # Auto-checkpoint
            if self._ckpt and self._global_step % cfg.checkpoint_every_steps == 0:
                ckpt_id = self._save_checkpoint()
                result.checkpoint_id = ckpt_id

            if self._global_step % cfg.log_every_steps == 0:
                log.info(
                    "step=%d loss=%.4f grad_norm=%.3f tok/s=%.0f%s",
                    self._global_step, result.loss, result.grad_norm, result.tokens_per_second,
                    f" [ckpt={result.checkpoint_id}]" if result.checkpoint_id else "",
                )

            # Reset before yielding.  A caller is allowed to stop after one
            # yielded result and request an explicit final checkpoint; leaving
            # these counters live until the next ``next()`` would make that
            # valid optimizer boundary look like an unfinished accumulation.
            self._accum_loss = 0.0
            self._accum_tokens = 0
            t_step_start = time.time()

            yield result

    # ------------------------------------------------------------------
    # Grad norm (portable: iterate all stage params, sum norms)
    # ------------------------------------------------------------------

    def _clip_grad_norm(self) -> float:
        """
        Compute global gradient norm across all stage workers and clip in place.
        Each stage owns its own parameters; no distributed all-reduce needed here
        (portable pipeline = single process owning all stages).
        """
        all_params = [
            p for w in self._workers
            for p in w._model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if not all_params:
            return 0.0

        total_norm = torch.nn.utils.clip_grad_norm_(
            all_params,
            self._cfg.max_grad_norm,
            norm_type=2.0,
            error_if_nonfinite=False,
        )
        return float(total_norm)

    def _all_gradients_finite(self) -> bool:
        """Return whether every currently accumulated gradient is finite."""
        for worker in self._workers:
            for parameter in worker._model.parameters():
                if (
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ):
                    return False
        return True

    def _discard_pending_training(self) -> None:
        """Drop saved graphs, KV state, gradients and loss counters for a retry."""
        for worker in self._workers:
            worker.clear_saved()
            worker.clear_kv()
        for optimizer in self._opts:
            optimizer.zero_grad(set_to_none=True)
        self._accum_loss = 0.0
        self._accum_tokens = 0

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> str:
        checkpoint = self._ckpt
        if checkpoint is None:
            raise RuntimeError("cannot save a checkpoint without a coordinator")
        model_states = [w._model.state_dict() for w in self._workers]
        opt_states = []
        for worker, optimizer in zip(self._workers, self._opts):
            state = optimizer.state_dict()
            for group, names in zip(
                state['param_groups'], _optimizer_parameter_names(worker, optimizer)
            ):
                group['param_names'] = names
            opt_states.append(state)
        scheduler_states = [sched.state_dict() for sched in self._scheds]
        scaler_state = self._scaler.state_dict() if self._scaler else None
        rng_state = _capture_rng_state()

        return checkpoint.save(
            job_id=self._job_id,
            global_step=self._global_step,
            model_states=model_states,
            optimizer_states=opt_states,
            scheduler_states=scheduler_states or None,
            scaler_state=scaler_state,
            rng_states=[rng_state for _ in self._workers],
            data_cursor=self._data_cursor_state(),
        )

    @property
    def global_step(self) -> int:
        """Number of completed optimizer steps."""
        return self._global_step

    @property
    def data_cursor(self) -> int:
        """Number of packed data batches consumed by this trainer."""
        return self._data_cursor

    @property
    def data_cursor_exact(self) -> bool:
        """Whether the current cursor came from a durable checkpoint schema."""
        return self._data_cursor_exact

    def _data_cursor_state(self) -> dict[str, int | str]:
        """Serialize the stable cursor contract used by ``make_causal_batches``."""
        return {
            "schema": "packed_batch_v1",
            "batches_consumed": self._data_cursor,
            "gradient_accumulation_steps": self._cfg.gradient_accumulation_steps,
        }

    def save_checkpoint(self) -> str:
        """Persist the current optimizer boundary and return its ID.

        The caller must invoke this between optimizer steps.  Exposing this
        operation keeps CLI/service code from reaching into private state when
        the configured periodic interval does not land on the final step.
        """
        if self._ckpt is None:
            raise RuntimeError("cannot save a checkpoint without a coordinator")
        if self._accum_tokens or any(
            parameter.grad is not None
            for worker in self._workers
            for parameter in worker._model.parameters()
        ):
            raise RuntimeError(
                "save_checkpoint must be called at an optimizer boundary; "
                "finish or discard the current accumulation window first"
            )
        return self._save_checkpoint()

    def resume(
        self,
        checkpoint_id: str,
        *,
        allow_inexact_data_resume: bool = False,
    ) -> None:
        """Restore model/optimizer/step from a committed checkpoint.

        A training checkpoint without the packed-data cursor cannot safely
        resume a dataset iterator.  Such checkpoints are rejected by default;
        callers doing an explicitly inexact recovery may opt into starting
        the data cursor at zero.
        """
        manifest, payloads = self._checkpoint_payloads(checkpoint_id)
        self._restore_checkpoint_payloads(
            manifest,
            payloads,
            require_exact_data_cursor=not allow_inexact_data_resume,
        )
        log.info("resumed from checkpoint %s at step %d", checkpoint_id, self._global_step)

    def _checkpoint_payloads(self, checkpoint_id: str) -> tuple[dict, list[dict]]:
        """Load every committed shard and verify it matches this topology."""
        checkpoint = self._ckpt
        if checkpoint is None:
            raise RuntimeError("cannot restore without a checkpoint coordinator")
        try:
            manifest = checkpoint.load_manifest(checkpoint_id)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"checkpoint {checkpoint_id!r} not found or not readable"
            ) from exc

        checkpoint_job_id = manifest.get("job_id")
        if checkpoint_job_id != self._job_id:
            raise ValueError(
                f"checkpoint {checkpoint_id!r} belongs to job {checkpoint_job_id!r}, "
                f"but this trainer is {self._job_id!r}"
            )

        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise ValueError(f"checkpoint {checkpoint_id!r} has invalid shard metadata")
        stage_numbers: list[int] = []
        for entry in shards:
            if isinstance(entry, dict) and isinstance(entry.get("stage"), int):
                stage_numbers.append(entry["stage"])
        stage_numbers.sort()
        expected_stages = list(range(len(self._workers)))
        if stage_numbers != expected_stages:
            raise ValueError(
                f"checkpoint {checkpoint_id!r} contains stages {stage_numbers}, "
                f"but this trainer has {expected_stages}; use reshard before restore"
            )

        payloads: list[dict] = []
        for stage_idx in expected_stages:
            payload = checkpoint.load_stage(checkpoint_id, stage=stage_idx)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"checkpoint {checkpoint_id!r} stage {stage_idx} payload is invalid"
                )
            if not isinstance(payload.get("model"), dict):
                raise ValueError(
                    f"checkpoint {checkpoint_id!r} stage {stage_idx} has no model state"
                )
            if not isinstance(payload.get("optimizer"), dict):
                raise ValueError(
                    f"checkpoint {checkpoint_id!r} stage {stage_idx} has no optimizer state"
                )
            payloads.append(payload)
        return manifest, payloads

    def _restore_checkpoint_payloads(
        self,
        manifest: dict,
        payloads: list[dict],
        *,
        require_exact_data_cursor: bool = True,
    ) -> None:
        """Restore a complete checkpoint without leaving partial live state."""
        if len(payloads) != len(self._workers):
            raise ValueError(
                f"checkpoint has {len(payloads)} payloads for {len(self._workers)} workers"
            )
        global_step = manifest.get("global_step")
        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or global_step < 0
        ):
            raise ValueError("checkpoint global_step must be a non-negative integer")

        for stage_idx, payload in enumerate(payloads):
            if payload.get("global_step") != global_step:
                raise ValueError(
                    f"checkpoint stage {stage_idx} global_step does not match manifest"
                )
        has_scheduler = ["scheduler" in payload for payload in payloads]
        if self._scheds and has_scheduler != [True] * len(payloads):
            raise ValueError(
                "checkpoint scheduler state does not match the live trainer"
            )
        if not self._scheds and any(has_scheduler):
            raise ValueError(
                "checkpoint contains scheduler state but the live trainer has no schedulers"
            )
        has_scaler = ["scaler" in payload for payload in payloads]
        if self._scaler is not None and has_scaler != [True] * len(payloads):
            raise ValueError("checkpoint scaler state does not match the live trainer")
        if self._scaler is None and any(has_scaler):
            raise ValueError(
                "checkpoint contains scaler state but the live trainer has no scaler"
            )

        self._validate_restore_payloads(payloads)
        data_cursor, data_cursor_exact = _extract_data_cursor(
            manifest,
            payloads,
            expected_gradient_accumulation_steps=self._cfg.gradient_accumulation_steps,
            require_exact=require_exact_data_cursor,
        )
        previous = self._snapshot_live_training_state()
        try:
            self._discard_pending_training()
            for worker in self._workers:
                # Model weights and all request KV caches must describe the
                # same model version after a restore.  Keeping a named cache
                # would decode tokens with keys/values from the old weights.
                worker.clear_all_kv()
            for i, (worker, payload) in enumerate(zip(self._workers, payloads)):
                worker._model.load_state_dict(payload["model"])
                self._opts[i].load_state_dict(payload["optimizer"])
                if "scheduler" in payload and i < len(self._scheds):
                    self._scheds[i].load_state_dict(payload["scheduler"])

            scaler_state = payloads[0].get("scaler")
            if scaler_state is not None and self._scaler is not None:
                self._scaler.load_state_dict(scaler_state)
            rng_state = payloads[0].get("rng")
            if rng_state is not None:
                _restore_rng_state(rng_state)

            self._global_step = global_step
            self._data_cursor = data_cursor
            self._data_cursor_exact = data_cursor_exact
            sync_tied_parameters(self._workers)
        except BaseException as exc:
            try:
                self._restore_live_training_state(previous)
            except BaseException as rollback_exc:
                raise RuntimeError(
                    "checkpoint restore failed and live-state rollback failed"
                ) from rollback_exc
            raise exc

    def _validate_restore_payloads(self, payloads: list[dict]) -> None:
        """Reject incompatible model/optimizer state before mutating live objects."""
        for stage_idx, (worker, optimizer, payload) in enumerate(
            zip(self._workers, self._opts, payloads)
        ):
            saved_model = payload["model"]
            live_model = worker._model.state_dict()
            if set(saved_model) != set(live_model):
                missing = sorted(set(live_model) - set(saved_model))
                unexpected = sorted(set(saved_model) - set(live_model))
                raise ValueError(
                    f"stage {stage_idx} model layout differs: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            for name, current in live_model.items():
                saved = saved_model[name]
                if (
                    not isinstance(saved, torch.Tensor)
                    or tuple(saved.shape) != tuple(current.shape)
                ):
                    raise ValueError(
                        f"stage {stage_idx} parameter shape mismatch for {name!r}"
                    )

            saved_optimizer = payload["optimizer"]
            live_names = _optimizer_parameter_names(worker, optimizer)
            saved_groups = saved_optimizer.get("param_groups")
            if not isinstance(saved_groups, list):
                raise ValueError(f"stage {stage_idx} optimizer state has no param_groups")
            if len(saved_groups) != len(optimizer.param_groups):
                raise ValueError(
                    f"stage {stage_idx} optimizer has {len(saved_groups)} saved groups, "
                    f"but live optimizer has {len(optimizer.param_groups)}"
                )
            for group_idx, (saved_group, live_group) in enumerate(
                zip(saved_groups, optimizer.param_groups)
            ):
                if saved_group.get('param_names') != live_names[group_idx]:
                    raise ValueError(
                        f"stage {stage_idx} optimizer parameter names/order missing or mismatched "
                        f"in group {group_idx}; checkpoint cannot safely restore optimizer state"
                    )
                saved_params = saved_group.get("params")
                live_params = live_group.get("params")
                if not isinstance(saved_params, list) or not isinstance(live_params, list):
                    raise ValueError(
                        f"stage {stage_idx} optimizer group {group_idx} has invalid params"
                    )
                if len(saved_params) != len(live_params):
                    raise ValueError(
                        f"stage {stage_idx} optimizer group {group_idx} has "
                        f"{len(saved_params)} saved params, but live group has "
                        f"{len(live_params)}"
                    )

    def _snapshot_live_training_state(self) -> dict:
        """Capture enough state to undo a failed restore operation."""
        return {
            "models": [
                {
                    name: value.detach().clone()
                    for name, value in worker._model.state_dict().items()
                }
                for worker in self._workers
            ],
            "optimizers": [deepcopy(optimizer.state_dict()) for optimizer in self._opts],
            "schedulers": [deepcopy(scheduler.state_dict()) for scheduler in self._scheds],
            "scaler": deepcopy(self._scaler.state_dict()) if self._scaler is not None else None,
            "rng": _capture_rng_state(),
            "global_step": self._global_step,
            "data_cursor": self._data_cursor,
            "data_cursor_exact": self._data_cursor_exact,
            "accum_loss": self._accum_loss,
            "accum_tokens": self._accum_tokens,
            "grads": [
                [
                    parameter.grad.detach().clone() if parameter.grad is not None else None
                    for parameter in worker._model.parameters()
                ]
                for worker in self._workers
            ],
            "saved": [dict(worker._saved) for worker in self._workers],
            "default_kv": [worker._kv_caches for worker in self._workers],
            "named_kv": [dict(worker._kv_caches_by_key) for worker in self._workers],
        }

    def _restore_live_training_state(self, state: dict) -> None:
        """Restore the state captured before a failed checkpoint operation."""
        for worker, model_state in zip(self._workers, state["models"]):
            worker._model.load_state_dict(model_state)
        for optimizer, optimizer_state in zip(self._opts, state["optimizers"]):
            optimizer.load_state_dict(optimizer_state)
        for scheduler, scheduler_state in zip(self._scheds, state["schedulers"]):
            scheduler.load_state_dict(scheduler_state)
        if self._scaler is not None and state["scaler"] is not None:
            self._scaler.load_state_dict(state["scaler"])
        _restore_rng_state(state["rng"])
        self._global_step = state["global_step"]
        self._data_cursor = state["data_cursor"]
        self._data_cursor_exact = state["data_cursor_exact"]
        self._accum_loss = state["accum_loss"]
        self._accum_tokens = state["accum_tokens"]
        for worker, saved, default_kv, named_kv, grads in zip(
            self._workers,
            state["saved"],
            state["default_kv"],
            state["named_kv"],
            state["grads"],
        ):
            worker._saved = saved
            worker._kv_caches = default_kv
            worker._kv_caches_by_key = named_kv
            for parameter, grad in zip(worker._model.parameters(), grads):
                parameter.grad = grad.to(parameter.device) if grad is not None else None

    def replace_worker(
        self,
        stage_idx: int,
        new_worker: StageWorker,
        checkpoint_id: str | None = None,
    ) -> None:
        """
        Replace a failed stage worker with a fresh one.

        If checkpoint_id is provided (and self._ckpt is configured), the entire
        pipeline is restored to that committed logical step.  Restoring only
        the replacement stage would mix weights and optimizer state from two
        different steps.

        Without a checkpoint, the replacement receives a snapshot of the old
        stage's current model weights and starts with a fresh optimizer state.
        This path is useful for a controlled warm replacement, but it is not a
        crash-safe recovery point; callers handling a failed worker should
        always provide the latest committed checkpoint.

        This enables training to survive a stage crash:
          1. Detect dead stage (exception from pipeline_train_step).
          2. Spin up a fresh StageWorker on the replacement machine/device.
          3. Call replace_worker(stage_idx, new_worker, last_committed_ckpt).
          4. Resume the data iterator from where it left off (caller is responsible
             for iterator position or data-cursor in the checkpoint).
        """
        if stage_idx < 0 or stage_idx >= len(self._workers):
            raise IndexError(f"stage_idx {stage_idx} out of range [0, {len(self._workers)})")

        if checkpoint_id is not None and self._ckpt is None:
            raise RuntimeError(
                "cannot restore a replacement worker without a checkpoint coordinator"
            )
        old_worker = self._workers[stage_idx]
        old_opt = self._opts[stage_idx]

        manifest = None
        payloads = None
        if checkpoint_id is not None:
            manifest, payloads = self._checkpoint_payloads(checkpoint_id)
            # Validate the replacement model before changing the trainer's
            # live worker list.  The complete restore below repeats the load
            # after the swap, but this keeps a bad replacement fail-closed.
            new_worker._model.load_state_dict(payloads[stage_idx]["model"])

        # Rebuild the optimizer for every replacement, even when no checkpoint
        # is available.  Optimizers retain the exact Parameter objects passed
        # to them; leaving the old optimizer in place would make subsequent
        # training update the dead worker's tensors while the replacement
        # silently stays frozen.
        new_opt = _rebuild_optimizer(old_opt, old_worker, new_worker)

        if payloads is not None:
            new_opt.load_state_dict(payloads[stage_idx]["optimizer"])
        else:
            # A fresh replacement must not silently introduce randomly
            # initialized weights.  Copy the last live model version before
            # swapping the worker.  Optimizer state cannot be inferred safely
            # without a checkpoint and is intentionally reset above.
            live_state = {
                name: value.detach().cpu().clone()
                for name, value in old_worker._model.state_dict().items()
            }
            new_worker._model.load_state_dict(live_state, strict=True)
            log.warning(
                "stage %d replacement has no checkpoint; copied model weights and "
                "reset optimizer state",
                stage_idx,
            )

        scheduler = self._scheds[stage_idx] if stage_idx < len(self._scheds) else None
        try:
            self._opts[stage_idx] = new_opt
            if scheduler is not None:
                # PyTorch schedulers hold a reference to their optimizer.  Rebind
                # it for both checkpoint and fresh replacement paths.
                scheduler.optimizer = new_opt
            self._workers[stage_idx] = new_worker
            if checkpoint_id is not None:
                assert manifest is not None and payloads is not None
                self._restore_checkpoint_payloads(manifest, payloads)
                log.info(
                    "stage %d: replaced worker and restored from checkpoint %s",
                    stage_idx, checkpoint_id,
                )
            else:
                log.info("stage %d: replaced worker (no checkpoint restore)", stage_idx)

            sync_tied_parameters(self._workers)
        except BaseException:
            # A failed restore must leave the trainer pointing at the original
            # worker and optimizer objects.  The checkpoint restore method has
            # already rolled back their tensors when possible.
            self._workers[stage_idx] = old_worker
            self._opts[stage_idx] = old_opt
            if scheduler is not None:
                scheduler.optimizer = old_opt
            raise


def _optimizer_parameter_names(
    worker: StageWorker, optimizer: torch.optim.Optimizer,
) -> list[list[str]]:
    """Describe optimizer ordering by model names, never transient parameter IDs."""
    names = {id(parameter): name for name, parameter in worker._model.named_parameters()}
    try:
        return [[names[id(parameter)] for parameter in group['params']]
                for group in optimizer.param_groups]
    except KeyError as exc:
        raise ValueError("optimizer contains a parameter outside the stage model") from exc


def _extract_data_cursor(
    manifest: dict,
    payloads: list[dict],
    *,
    expected_gradient_accumulation_steps: int,
    require_exact: bool,
) -> tuple[int, bool]:
    """Validate and extract one shared data cursor from all checkpoint shards.

    Older checkpoints predate the cursor contract.  They remain readable for
    weight/export compatibility, but a trainer marks their cursor as
    inexact and starts at zero; the CLI reports that limitation instead of
    claiming an exact data resume.
    """
    present = ["data_cursor" in payload for payload in payloads]
    if not any(present):
        if require_exact:
            raise ValueError(
                "checkpoint has no packed-batch data cursor; exact training resume "
                "is unavailable (use allow_inexact_data_resume=True only if replay "
                "from the beginning is intentional)"
            )
        return 0, False
    if present != [True] * len(payloads):
        raise ValueError("checkpoint data cursor must be present in every stage shard")

    cursors = [payload["data_cursor"] for payload in payloads]
    for stage_idx, cursor in enumerate(cursors):
        if not isinstance(cursor, dict):
            raise ValueError(f"checkpoint stage {stage_idx} data cursor is invalid")
        if cursor.get("schema") != "packed_batch_v1":
            raise ValueError(
                f"checkpoint stage {stage_idx} has unsupported data cursor schema"
            )
        batches = cursor.get("batches_consumed")
        if isinstance(batches, bool) or not isinstance(batches, int) or batches < 0:
            raise ValueError(
                f"checkpoint stage {stage_idx} data cursor has invalid batches_consumed"
            )
        accumulation = cursor.get("gradient_accumulation_steps")
        if (
            isinstance(accumulation, bool)
            or not isinstance(accumulation, int)
            or accumulation < 1
        ):
            raise ValueError(
                f"checkpoint stage {stage_idx} data cursor has invalid accumulation policy"
            )
        if accumulation != expected_gradient_accumulation_steps:
            raise ValueError(
                "checkpoint gradient accumulation policy does not match the live trainer: "
                f"saved={accumulation}, live={expected_gradient_accumulation_steps}"
            )

    first = cursors[0]
    if any(cursor != first for cursor in cursors[1:]):
        raise ValueError("checkpoint data cursor differs between stage shards")

    manifest_cursor = manifest.get("data_cursor")
    if manifest_cursor is not None and manifest_cursor != first:
        raise ValueError("checkpoint manifest data cursor differs from stage shards")
    return int(first["batches_consumed"]), True


def _rebuild_optimizer(
    old_optimizer: torch.optim.Optimizer,
    old_worker: StageWorker,
    new_worker: StageWorker,
) -> torch.optim.Optimizer:
    """Recreate an optimizer with the old param-group layout on new tensors."""
    old_named = dict(old_worker._model.named_parameters())
    new_by_name = dict(new_worker._model.named_parameters())
    if set(old_named) != set(new_by_name):
        missing_model = sorted(set(old_named) - set(new_by_name))
        unexpected_model = sorted(set(new_by_name) - set(old_named))
        raise ValueError(
            "replacement worker model layout differs: "
            f"missing={missing_model}, unexpected={unexpected_model}"
        )
    shape_mismatches = [
        f"{name}: old={tuple(old_param.shape)}, new={tuple(new_by_name[name].shape)}"
        for name, old_param in old_named.items()
        if tuple(old_param.shape) != tuple(new_by_name[name].shape)
    ]
    if shape_mismatches:
        raise ValueError(
            "replacement worker parameter shapes differ: "
            + ", ".join(shape_mismatches)
        )
    old_names = {id(param): name for name, param in old_named.items()}
    old_params = [param for group in old_optimizer.param_groups for param in group["params"]]
    missing = [old_names.get(id(param), "<unknown>") for param in old_params
               if id(param) not in old_names or old_names[id(param)] not in new_by_name]
    if missing:
        raise ValueError(
            "replacement worker is missing optimizer parameters: "
            + ", ".join(missing)
        )
    replacement_by_old_id = {
        id(old): new_by_name[old_names[id(old)]] for old in old_params
    }
    groups = []
    for old_group in old_optimizer.param_groups:
        group = {
            key: deepcopy(value)
            for key, value in old_group.items()
            if key != "params"
        }
        group["params"] = [replacement_by_old_id[id(param)] for param in old_group["params"]]
        groups.append(group)
    return type(old_optimizer)(groups)  # type: ignore[call-arg]


class _noop_opt:
    """Wraps an optimizer to accumulate gradients without stepping."""
    def __init__(self, opt: torch.optim.Optimizer) -> None:
        self._opt = opt

    def step(self) -> None:
        pass

    def zero_grad(self, *, set_to_none: bool = False) -> None:
        pass

    def state_dict(self) -> dict:
        return self._opt.state_dict()


def _capture_rng_state() -> dict:
    """Capture process RNGs needed for a deterministic same-topology resume."""
    state: dict = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    """Restore RNG state when the saved device topology is compatible."""
    cpu_state = state.get("torch_cpu")
    if cpu_state is not None:
        torch.set_rng_state(cpu_state)
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        if len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_state)
        else:
            log.warning(
                "checkpoint has RNG state for %d CUDA devices; current process has %d; "
                "leaving CUDA RNG unchanged",
                len(cuda_state),
                torch.cuda.device_count(),
            )
