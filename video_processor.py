
import cv2
import segno
import shutil
import subprocess
import tempfile
import numpy as np

from pathlib import Path


class VideoProcessor:
    """Embed Segno QR codes in order, repeating the list until the video ends."""

    QR_CHANGE_DELAY = 0.5
    
    # Brightness offset
    STRENGTH = 0.04

    # Render the supplied codes at this many pixels per module.
    QR_MODULE_SIZE = 8



    def __init__(self, video_path: str, qr_codes: list[segno.QRCode]) -> None:
        if not qr_codes:
            raise ValueError("At least one QR code is required")
        if any(not isinstance(code, segno.QRCode) for code in qr_codes):
            raise TypeError("qr_codes must contain Segno QRCode objects")
        self.video_path = video_path
        self.qr_codes = list(qr_codes)

    def _create_qr_signal(self, qr: segno.QRCode) -> np.ndarray:
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
            ]
            hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_region)

            v_float = v.astype(np.float32) + phase * self.STRENGTH * 255.0 * qr_signal
            # v_float = v.astype(np.float32) + np.where(qr_signal > 0, (phase * self.STRENGTH * 255.0), 0)
            v_modified = np.clip(v_float, 0.0, 255.0).astype(np.uint8)

            hsv_region = cv2.merge([h, s, v_modified])
            region = cv2.cvtColor(hsv_region, cv2.COLOR_HSV2BGR)

            result[y:y + qr_h, x:x + qr_w] = np.clip(
                region, 0, 255
            ).astype(np.uint8)
        return result

    def place_qr_corners(self, frame: np.ndarray, qr_signal: np.ndarray | None, phase: int) -> np.ndarray:
            result = frame.copy()
            if qr_signal is not None:
                h, w = frame.shape[:2]
                qr_h, qr_w = qr_signal.shape
    
                if qr_h > h or qr_w > w:
                    raise ValueError("QR code exceeds video dimensions")
                margin = 15
    
                corners = [
                    (margin, margin),
                    (w - qr_w - margin, margin),
                    (margin, h - qr_h - margin),
                    (w - qr_w - margin, h - qr_h - margin)
                ]
    
                for x, y in corners:
                    x = max(0, min(x, w - qr_w))
                    y = max(0, min(y, h - qr_h))
    
                    region = result[
                        y:y + qr_h,
                        x:x + qr_w
                    ]
                    
                    if region.shape[0] != qr_h or region.shape[1] != qr_w:
                        continue
    
                    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                    chan_h, chan_s, chan_v = cv2.split(hsv_region)
    
                    v_float = chan_v.astype(np.float32) + phase * self.STRENGTH * 255.0 * qr_signal
                    v_modified = np.clip(v_float, 0.0, 255.0).astype(np.uint8)
    
                    hsv_region = cv2.merge([chan_h, chan_s, v_modified])
                    region = cv2.cvtColor(hsv_region, cv2.COLOR_HSV2BGR)
    
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

    def add_border(self, img: np.ndarray, width=40):
        h, w = img.shape[:2]
        cv2.rectangle(
            img, 
            (0, 0), 
            (w - 1, h - 1), 
            (255, 255, 255), 
            thickness=width
        )
        cv2.rectangle(
            img, 
            (0, 0), 
            (w - 1, h - 1), 
            (0, 0, 0), 
            thickness=int(width/2)
        )
        return img

    def output(self, output_path: str, double_fps = False, use_skip = False):
        if not np.isfinite(self.QR_CHANGE_DELAY) or self.QR_CHANGE_DELAY <= 0:
            raise ValueError("QR_CHANGE_DELAY must be finite and greater than zero")
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

        output_fps = (source_fps * 2) if double_fps else source_fps

        Path(output_path).parent.mkdir(
            parents=True, exist_ok=True
        )

        previous_qr_index = None
        qr_signal = None

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
                    qr_phase = True
                    skip_qr = True
                    while True:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break

                        qr_index = int(
                            frame_index / source_fps / self.QR_CHANGE_DELAY
                        ) % len(self.qr_codes)

                        if qr_index != previous_qr_index:
                            qr_signal = self._create_qr_signal(self.qr_codes[qr_index])
                            previous_qr_index = qr_index

                        # # Generate two output frames
                        # bright_frame = self.place_qr(
                        #     frame, qr_signal, +1
                        # )

                        # dark_frame = self.place_qr(
                        #     frame, qr_signal, -1
                        # )

                        # writer.write(bright_frame)
                        # writer.write(dark_frame)

                        for _ in range(2 if double_fps else 1):
                            if skip_qr and use_skip:
                                writer.write(frame)
                                skip_qr = False
                            else:
                                qr_frame = self.place_qr(
                                    frame, qr_signal, +1 if qr_phase else -1
                                )
                                writer.write(qr_frame)
                                qr_phase = not qr_phase
                                # skip next qr once a pair is complete
                                if qr_phase:
                                    skip_qr = True

                        frame_index += 1

                finally:
                    writer.release()

                self._mux_audio(silent_path, output_path)

        finally:
            cap.release()

        print(f"Saved {output_fps:.2f} FPS video to {output_path}")

