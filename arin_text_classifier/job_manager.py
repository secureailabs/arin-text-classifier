from queue import Queue
from threading import Thread
from typing import Any, Dict, List


class JobManager:

    def __init__(self, dict_classifier) -> None:
        self.dict_classifier = dict_classifier
        self.thread = Thread(target=self.run, daemon=True)
        self.is_running = False
        self.queue = Queue()
        self.thread.start()

    def start(self) -> None:
        self.is_running = True
        self.thread.start()

    def run(self) -> None:
        while self.is_running:
            classifier_id = self.queue.get()
            job.run()
            self.queue.task_done()

    def stop(self) -> None:
        self.is_running = False
        self.thread.join()

    def add_job(self, job: Any) -> None:
        self.queue.put(job)