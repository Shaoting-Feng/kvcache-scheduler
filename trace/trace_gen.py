import pathlib
from typing import List

import numpy as np
import pandas as pd


def main():
    n_docs: int = 200
    interval_ms: List[int] = [200, 400, 600, 800, 1000]
    tot_time_ms: int = 120000

    for avg_interval in interval_ms:
        trace: pd.DataFrame = pd.DataFrame(columns=["ts", "doc_id"])  # type: ignore
        ts_cur: int = 0
        ts_end: int = tot_time_ms
        rng = np.random.default_rng()

        while ts_cur <= ts_end:
            interval = rng.exponential(avg_interval)
            ts_cur += int(interval)
            trace = pd.concat(
                [
                    trace,
                    pd.DataFrame(
                        {
                            "ts": [ts_cur],
                            "doc_id": [rng.integers(0, n_docs)],
                        }
                    ),
                ],
                ignore_index=True,
            )
        out_path: pathlib.Path = pathlib.Path(
            f"poisson-{avg_interval}ms.trace"
        )
        trace.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
