from dataclasses import dataclass
import heapq
import pysubs2
from itertools import count

# Time is in ms since start file
@dataclass
class Subtitle:
    startTime: int
    endTime: int
    sub_string: str

class SubtitleManager:
    def __init__(self) -> None:
        self.heap = []
        self.counter = count()

    def push(self, sub: Subtitle) -> None:
        heapq.heappush(self.heap,(sub.startTime,next(self.counter), sub))

    def peek(self) -> Subtitle | None:
        if not self.heap:
            return None

        return self.heap[0][2]

    def pop(self) -> Subtitle | None:
        if not self.heap:
            return None

        return heapq.heappop(self.heap)[2]

    def get_ready(self, currentTime: int) -> list[Subtitle]:
        ready = []

        while self.heap and self.peek().endTime < currentTime:
            self.pop()

        i = 0
        while self.heap and i < len(self.heap) and self.heap[i][0] <= currentTime:
            ready.append(self.heap[i][2])
            i += 1

        return ready

    def import_from_srt(self, file_path: str):
        subs = pysubs2.load(file_path)

        for sub in subs:
            self.push(Subtitle(startTime=sub.start, endTime=sub.end, sub_string=sub.text))


