import pathlib
from threading import Thread

from flowsim import Buffer, Receiver, Sender

TRACE_DIR = pathlib.Path(__file__).parent / "trace"


def main():
    buf = Buffer()
    receiver = Receiver(buf, TRACE_DIR / "poisson-600ms.trace")
    sender = Sender(buf, TRACE_DIR / "doc_stats.csv")

    receiver_thread: Thread = Thread(target=receiver.trace_replay)
    sender_thread: Thread = Thread(target=sender.run)
    sender_thread.daemon = True

    sender_thread.start()
    receiver_thread.start()

    receiver_thread.join()


if __name__ == "__main__":
    main()
