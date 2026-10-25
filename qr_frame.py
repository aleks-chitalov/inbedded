"""Binary packets carried by individual QR codes (network byte order).

magic:4 | version:u8 | sha256:32 | message_size:u32 | data_shards:u8 |
parity_shards:u8 | index:u8 | shard_bytes:u16 | payload | crc32:u32
"""

from dataclasses import dataclass
import struct
import zlib


MAGIC = b"QRRS"
VERSION = 1
HEADER = struct.Struct("!4sB32sIBBBH")
CRC = struct.Struct("!I")
FRAME_OVERHEAD = HEADER.size + CRC.size


class FrameChecksumError(ValueError):
    """A damaged packet that can be treated as a missing shard."""


@dataclass(frozen=True)
class QRFrame:
    message_id: bytes
    message_size: int
    data_shards: int
    parity_shards: int
    index: int
    shard_bytes: int
    payload: bytes

    def __post_init__(self) -> None:
        if len(self.message_id) != 32:
            raise ValueError("Message identifier must be a SHA-256 digest")
        if not 1 <= self.data_shards <= 255:
            raise ValueError("Data shard count must be between 1 and 255")
        if not 0 <= self.parity_shards <= 255 - self.data_shards:
            raise ValueError("Total data and parity shards must not exceed 255")
        if not 0 <= self.index < self.data_shards + self.parity_shards:
            raise ValueError("Shard index is out of range")
        if not 1 <= self.shard_bytes <= 65535:
            raise ValueError("Shard size must be between 1 and 65535 bytes")
        minimum_size = (self.data_shards - 1) * self.shard_bytes + 1
        if not minimum_size <= self.message_size <= self.data_shards * self.shard_bytes:
            raise ValueError("Message size does not match the data shard count")
        if len(self.payload) != self.shard_bytes:
            raise ValueError("Payload length does not match shard size")

    @property
    def metadata(self) -> tuple[bytes, int, int, int, int]:
        """Fields that must agree across all packets of a message."""
        return (
            self.message_id, self.message_size, self.data_shards,
            self.parity_shards, self.shard_bytes,
        )

    def to_bytes(self) -> bytes:
        data = HEADER.pack(
            MAGIC, VERSION, self.message_id, self.message_size,
            self.data_shards, self.parity_shards, self.index, self.shard_bytes,
        ) + self.payload
        return data + CRC.pack(zlib.crc32(data))

    @classmethod
    def from_bytes(cls, packet: bytes) -> "QRFrame":
        if len(packet) < FRAME_OVERHEAD:
            raise FrameChecksumError("Truncated QR packet")
        data, checksum = packet[:-CRC.size], packet[-CRC.size:]
        if zlib.crc32(data) != CRC.unpack(checksum)[0]:
            raise FrameChecksumError("QR packet CRC32 mismatch")
        magic, version, message_id, size, count, parity, index, width = HEADER.unpack(
            data[:HEADER.size]
        )
        if magic != MAGIC or version != VERSION:
            raise ValueError("Unsupported QR packet format or version")
        return cls(message_id, size, count, parity, index, width, data[HEADER.size:])
