import pathlib
import json
import time
from threading import Thread

from flowsim import Buffer, Receiver, Sender, SlidingWindow, SlidingWindowTimeBased

TRACE_DIR = pathlib.Path(__file__).parent / "trace"

# Use timestamp to avoid overwriting the result file
timestamp = time.strftime("%Y%m%d_%H%M%S")
file_name = f"log/result_{timestamp}.json"

# The parameters that need to be changed
trace = "poisson-1000ms.trace"
sender_strategy = "sliding_sliding"
buffer_startegy = "fifo"
sliding_window_size = 40

# Record the configuration
with open(file_name, mode='w') as file:
    data = {'trace': trace, 'sender_strategy': sender_strategy, 'buffer_scheduler_value': buffer_startegy, 'miss_rate': None, 'quality_score': None, 'retrieval_time': None} 
    file.seek(0)
    json.dump(data, file, indent=4)

def main():
    if sender_strategy != "convex_optimization":
        sliding_window = SlidingWindow(sliding_window_size)
    else:
        sliding_window = SlidingWindowTimeBased()
    clock = [0]

    buf = Buffer(file_name, buffer_startegy, sliding_window)
    receiver = Receiver(buf, TRACE_DIR / trace, file_name)
    sender = Sender(buf, TRACE_DIR / "doc_stats.csv", sliding_window, sender_strategy, clock)

    def update_clock_periodically():
        while True:
            time.sleep(1)  # 每10秒钟执行一次
            clock[0] += 1 * 1e6  # 每次增加 10 * 1e6

    timer_thread = Thread(target=update_clock_periodically)
    timer_thread.daemon = True
    timer_thread.start()

    receiver_thread: Thread = Thread(target=receiver.trace_replay)
    sender_thread: Thread = Thread(target=sender.run)
    sender_thread.daemon = True

    sender_thread.start()
    receiver_thread.start()

    receiver_thread.join()


if __name__ == "__main__":
    main()
