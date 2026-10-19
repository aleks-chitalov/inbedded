
import cv2
import segno
import shutil
import subprocess
import tempfile
import numpy as np

from pathlib import Path
from subtitle_manager import Subtitle, SubtitleManager
from protocol import SendSubtitle, CurrentTime, encode_message


class VideoProcessor:
    SUBTITLE_LEAD_MS = 1000

    # Brightness offset
    STRENGTH = 0.04

    # QR configuration
    QR_VERSION = 10
    QR_MODULE_SIZE = 8

    def __init__(self, video_path: str, sub_path: str) -> None:
        self.video_path = video_path

        self.sub_manager = SubtitleManager()
        self.sub_manager.import_from_srt(sub_path)

        self.current_subtitle = None

    def _select_message(self, current_time: int):
        # Fetch subtitles starting within the next second
        ready = self.sub_manager.get_ready(
            current_time + self.SUBTITLE_LEAD_MS
        )

        if ready:
            # Assume non-overlapping subtitles
            self.current_subtitle = ready[0]

        subtitle = self.current_subtitle

        # Repeat the subtitle until it expires
        if subtitle is not None and subtitle.endTime > current_time:
            return SendSubtitle(
                currentTime=current_time,
                subtitleStartTime=subtitle.startTime,
                subtitleEndTime=subtitle.endTime,
                sub_string=subtitle.sub_string
            )

        self.current_subtitle = None
        return None

    def _create_qr_signal(self, packet: bytes) -> np.ndarray:
        qr = segno.make_qr(
            packet,
            version=self.QR_VERSION,
            mode="byte",
            error="H",
            boost_error=False
        )

        # Segno matrix: 1 = black, 0 = white
        matrix = np.array(qr.matrix, dtype=np.uint8)

        # Add the standard 4-module quiet zone
        matrix = np.pad(matrix, 4, constant_values=0)

        matrix = cv2.resize(
            matrix,
            None,
            fx=self.QR_MODULE_SIZE,
            fy=self.QR_MODULE_SIZE,
            interpolation=cv2.INTER_NEAREST
        )

        # Black QR module = +1, white = -1
        return np.where(matrix == 1, 1.0, -1.0)

    def place_qr(self, frame: np.ndarray, qr_signal: np.ndarray | None, phase: int) -> np.ndarray:
        result = frame.copy()

        if qr_signal is not None:
            h, w = frame.shape[:2]
            qr_h, qr_w = qr_signal.shape

            if qr_h > h or qr_w > w:
                raise ValueError("QR code exceeds video dimensions")

            # Center the QR
            x = (w - qr_w) // 2
            y = (h - qr_h) // 2


            region = result[
                y:y + qr_h,
                x:x + qr_w
            ].astype(np.float32)

            offset = phase * self.STRENGTH * 255.0

            region += offset * qr_signal[:, :, None]

            result[y:y + qr_h, x:x + qr_w] = np.clip(
                region, 0, 255
            ).astype(np.uint8)

        return result

    def _mux_audio(self, silent_path: str, output_path: str):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("FFmpeg is not installed")

        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", silent_path,
            "-i", self.video_path,

            # Modified video from first input
            "-map", "0:v:0",

            # Original audio, if present
            "-map", "1:a?",

            # Copy both streams without re-encoding
            "-c:v", "copy",
            "-c:a", "copy",

            "-movflags", "+faststart",
            output_path
        ], check=True)

    def output(self, output_path: str):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        
        print("Opened:", cap.isOpened())
        print("FPS:", cap.get(cv2.CAP_PROP_FPS))
        print("Frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))


        source_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not np.isfinite(source_fps) or source_fps <= 0:
            cap.release()
            raise ValueError("Invalid input FPS")

        output_fps = source_fps * 2

        Path(output_path).parent.mkdir(
            parents=True, exist_ok=True
        )

        try:
            with tempfile.TemporaryDirectory() as tmp:
                silent_path = str(Path(tmp) / "silent.mp4")

                writer = cv2.VideoWriter(
                    silent_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError("Could not create output video")

                frame_index = 0

                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break

                        current_time = round(
                            frame_index / source_fps * 1000
                        )
                        
                        qr_signal = None
                        msg = self._select_message(current_time)

                        if msg is not None:
                            packet = encode_message(msg)
                            qr_signal = self._create_qr_signal(packet)


                        print(f"Frame index: {frame_index}")

                        # Generate two output frames
                        bright_frame = self.place_qr(
                            frame, qr_signal, +1
                        )

                        dark_frame = self.place_qr(
                            frame, qr_signal, -1
                        )

                        writer.write(bright_frame)
                        writer.write(dark_frame)

                        frame_index += 1

                finally:
                    writer.release()

                self._mux_audio(silent_path, output_path)

        finally:
            cap.release()

        print(f"Saved {output_fps:.2f} FPS video to {output_path}")


