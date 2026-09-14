"""Credit-based backpressure — sender waits when receiver is slow."""
from __future__ import annotations

import asyncio


class CreditManager:
    """
    Sender acquires `n` bytes of credit before sending; receiver grants credit
    back after consuming. Prevents unbounded queuing when the far end is slow.
    """

    def __init__(self, initial_bytes: int, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if initial_bytes < 0 or initial_bytes > max_bytes:
            raise ValueError("initial_bytes must be in [0, max_bytes]")
        self._available = initial_bytes
        self._max = max_bytes
        self._lock = asyncio.Lock()
        self._granted = asyncio.Condition(self._lock)

    @property
    def available(self) -> int:
        return self._available

    async def acquire(self, n: int, *, timeout: float | None = None) -> None:
        """Block until n bytes of credit are available."""
        if n < 0:
            raise ValueError("credit request must be non-negative")
        if n > self._max:
            raise ValueError(f"requested {n} > max credit {self._max}")
        async with self._granted:
            await asyncio.wait_for(
                self._granted.wait_for(lambda: self._available >= n),
                timeout=timeout,
            )
            self._available -= n

    def release(self, n: int) -> None:
        """Grant n bytes of credit back (called by receiver side)."""
        if n < 0:
            raise ValueError("credit release must be non-negative")

        async def _release() -> None:
            async with self._granted:
                self._available = min(self._available + n, self._max)
                self._granted.notify_all()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # The synchronous compatibility method is only useful outside an
            # active loop; there cannot be async waiters in that situation.
            self._available = min(self._available + n, self._max)
        else:
            loop.create_task(_release())

    async def release_async(self, n: int) -> None:
        if n < 0:
            raise ValueError("credit release must be non-negative")
        async with self._granted:
            self._available = min(self._available + n, self._max)
            self._granted.notify_all()
