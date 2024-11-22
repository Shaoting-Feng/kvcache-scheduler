import pathlib
from typing import List

import numpy as np
import pandas as pd


def main():
    n_docs: int = 200
    interval_ms: List[int] = [200, 400, 600, 800, 1000, 5000]
    tot_time_ms: int = 200000
    sigma: float = 3  # Standard deviation for Gaussian transition

    for avg_interval in interval_ms:
        trace: pd.DataFrame = pd.DataFrame(columns=["ts", "doc_id"])  # type: ignore
        ts_cur: int = 0
        ts_end: int = tot_time_ms
        rng = np.random.default_rng()
        current_doc_id: int = rng.integers(0, n_docs)  # Start with a random doc_id

        while ts_cur <= ts_end:
            interval = rng.exponential(avg_interval)
            ts_cur += int(interval)

            # Generate the next doc_id using Gaussian transition
            next_doc_id = int(rng.normal(loc=current_doc_id, scale=sigma))
            # Ensure the doc_id stays within valid bounds
            next_doc_id = max(0, min(n_docs - 1, next_doc_id))

            trace = pd.concat(
                [
                    trace,
                    pd.DataFrame(
                        {
                            "ts": [ts_cur],
                            "doc_id": [next_doc_id],
                        }
                    ),
                ],
                ignore_index=True,
            )

            # Update the current doc_id for the next iteration
            current_doc_id = next_doc_id
     
        out_path: pathlib.Path = pathlib.Path(
            f"trace/poisson-{avg_interval}ms.trace"
        )
        trace.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
