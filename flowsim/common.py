import ctypes
import platform
from dataclasses import dataclass
from time import monotonic_ns, sleep
from typing import Callable

from pandas import DataFrame

ST: float = monotonic_ns() / 1e6


@dataclass
class Document:
    id: int
    size: int
    quality: int
    gen_lat: int


usleep: Callable[[int], None] = (
    ctypes.CDLL("libc.so.6").usleep
    if platform.system() == "Linux"
    else lambda sleep_time_us: sleep(sleep_time_us / 1e6)
)


def get_cur_bw(trace: DataFrame) -> float:
    cur_ts: float = monotonic_ns() / 1e6 - ST
    bw: float = -1
    for _, row in trace.iterrows():
        if row["ts"] <= cur_ts:
            bw = row["bw"]  # type: ignore
    return bw
