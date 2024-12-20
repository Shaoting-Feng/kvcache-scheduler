class SlidingWindow:
    def __init__(self, size: int) -> None:
        self._size = size
        self.window = [(-1, 0)] * size
        self._idx = 0

    def add(self, val: int, ver: float) -> None:
        self.window[self._idx] = (val, ver)
        self._idx = (self._idx + 1) % self._size

class SlidingWindowTimeBased:
    def __init__(self) -> None:
        self.window = []

    def add(self, val: int, ver: float) -> None:
        self.window.append((val, ver))
        
    def clear(self) -> None:
        self.window = []
