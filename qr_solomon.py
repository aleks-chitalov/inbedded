"""Transfer arbitrary bytes with Reed–Solomon parity across QR codes.

Each byte position forms a separate RS codeword across the shards. Losing a
whole QR code therefore erases one symbol in every codeword. With r parity
codes, any r missing codes can be recovered. QR image correction (L/M/Q/H)
is independent of this outer erasure correction.
"""

from dataclasses import dataclass
from hashlib import sha256

import segno
from reedsolo import RSCodec, ReedSolomonError

from qr_frame import FRAME_OVERHEAD, FrameChecksumError, QRFrame


@dataclass(frozen=True)
class QRConfig:
    shard_bytes: int = 64
    parity_shards: int = 4
    qr_version: int = 10
    qr_error: str = "H"

    def __post_init__(self) -> None:
        for name in ("shard_bytes", "parity_shards", "qr_version"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.shard_bytes <= 65535:
            raise ValueError("shard_bytes must be between 1 and 65535")
        if not 0 <= self.parity_shards <= 254:
            raise ValueError("parity_shards must be between 0 and 254")
        if not 1 <= self.qr_version <= 40:
            raise ValueError("qr_version must be between 1 and 40")
        if self.qr_error not in ("L", "M", "Q", "H"):
            raise ValueError("qr_error must be L, M, Q, or H")
        # Byte mode makes capacity independent of the actual packet contents.
        make_qr(bytes(FRAME_OVERHEAD + self.shard_bytes), self)


def make_qr(packet: bytes, config: QRConfig) -> segno.QRCode:
    try:
        return segno.make_qr(
            packet, version=config.qr_version, error=config.qr_error,
            mode="byte", boost_error=False,
        )
    except segno.DataOverflowError as exc:
        raise ValueError(
            f"{len(packet)}-byte packet does not fit QR version "
            f"{config.qr_version} at correction {config.qr_error}; "
            "reduce shard_bytes, increase qr_version, or lower qr_error"
        ) from exc


def encode_message(data: bytes, config: QRConfig) -> list[segno.QRCode]:
    """Return data QR codes followed by parity QR codes for a single message.

    Each code can be rendered through its matrix or exported with code.save(...).
    Push scanned packet bytes into QRDecoder to recover the original message.
    """
    if not isinstance(data, bytes):
        raise TypeError("Message must be bytes")
    if not data:
        raise ValueError("Message must not be empty")
    width = config.shard_bytes
    count = (len(data) + width - 1) // width
    total = count + config.parity_shards
    if total > 255:
        raise ValueError(
            f"Message requires {total} shards; one group supports at most 255 "
            f"({(255 - config.parity_shards) * width} message bytes with this configuration)"
        )

    padded = data.ljust(count * width, b"\x00")
    shards = [bytearray(padded[i * width:(i + 1) * width]) for i in range(count)]
    shards.extend(bytearray(width) for _ in range(config.parity_shards))
    if config.parity_shards:
        codec = RSCodec(config.parity_shards)
        for column in range(width):
            codeword = codec.encode(padded[column::width])
            for index in range(count, total):
                shards[index][column] = codeword[index]

    message_id = sha256(data).digest()
    return [
        make_qr(
            QRFrame(
                message_id, len(data), count, config.parity_shards,
                index, width, bytes(shard),
            ).to_bytes(),
            config,
        )
        for index, shard in enumerate(shards)
    ]


class QRDecoder:
    """Collect one message from scanned QR packets, in any order.

    push() accepts one complete QR packet, returning None until enough unique
    shards arrive. The completed message is returned once; later duplicates
    return None. Call reset() before receiving another message.

    CRC failures are ignored. Mixed messages, inconsistent metadata, conflicting
    duplicates, or failed reconstruction raise ValueError.
    reset() has to be called after message is received.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Discard the current message and allow a new message to be received."""
        self._first: QRFrame | None = None
        self._received: dict[int, bytes] = {}
        self._complete = False

    def push(self, packet: bytes) -> bytes | None:
        """Accept the raw bytes from one scanned QR code."""
        try:
            frame = QRFrame.from_bytes(packet)
        except FrameChecksumError:
            return None
        if self._first is None:
            self._first = frame
        elif frame.metadata != self._first.metadata:
            raise ValueError("Packets belong to mixed messages or have inconsistent metadata")
        if frame.index in self._received:
            if self._received[frame.index] != frame.payload:
                raise ValueError(f"Conflicting packets for shard {frame.index}")
            return None
        if self._complete:
            return None
        self._received[frame.index] = frame.payload
        if len(self._received) < self._first.data_shards:
            return None

        data = self._reconstruct(self._first)
        self._complete = True
        return data

    def _reconstruct(self, first: QRFrame) -> bytes:
        received = self._received
        count, width = first.data_shards, first.shard_bytes
        total = count + first.parity_shards
        if all(index in received for index in range(count)):
            data = b"".join(received[index] for index in range(count))
        else:
            missing = [index for index in range(total) if index not in received]
            shards = [bytearray(received.get(index, bytes(width))) for index in range(total)]
            recovered = [bytearray(width) for _ in range(count)]
            codec = RSCodec(first.parity_shards)
            try:
                for column in range(width):
                    codeword = bytes(shard[column] for shard in shards)
                    decoded, _, _ = codec.decode(
                        codeword, erase_pos=missing, only_erasures=True,
                    )
                    for index in range(count):
                        recovered[index][column] = decoded[index]
            except ReedSolomonError as exc:
                raise ValueError("Reed–Solomon reconstruction failed") from exc
            data = b"".join(recovered)

        data = data[:first.message_size]
        if sha256(data).digest() != first.message_id:
            raise ValueError("Reconstructed message SHA-256 mismatch")
        return data
