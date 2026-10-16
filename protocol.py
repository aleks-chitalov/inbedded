"""

Time is in ms since playback started.

SendSubtitle (type 0x01):
    type:u8 | current_time:u32 | subtitle_start_time:u32 |
    subtitle_end_time:u32 | text_length:u16 | text:utf8[text_length] | crc32:u32

CurrentTime (type 0x02):
    type:u8 | current_time:u32 | crc32:u32
"""

from dataclasses import dataclass
from enum import IntEnum
import struct
import zlib
from typing import TypeAlias

from cobs import cobs


class MessageType(IntEnum):
    SEND_SUBTITLE = 1
    CURRENT_TIME = 2


@dataclass(frozen=True)
class SendSubtitle:
    currentTime: int
    subtitleStartTime: int
    subtitleEndTime: int
    sub_string: str


@dataclass(frozen=True)
class CurrentTime:
    currentTime: int


Message: TypeAlias = SendSubtitle | CurrentTime

MAX_SUBTITLE_BYTES = 4096
# 1 type + 3*u32 + 1*u16 + UTF-8 text + 4-byte CRC
MAX_RAW_MESSAGE_SIZE = 1 + 12 + 2 + MAX_SUBTITLE_BYTES + 4
# COBS worst-case overhead: ceil(raw_bytes / 254) plus initial code byte.
MAX_ENCODED_FRAME_SIZE = MAX_RAW_MESSAGE_SIZE + (MAX_RAW_MESSAGE_SIZE // 254) + 1


def encode_message(message: Message) -> bytes:
    if isinstance(message, SendSubtitle):
        text = message.sub_string.encode("utf-8")
        if len(text) > MAX_SUBTITLE_BYTES:
            raise ValueError("Subtitle exceeds maximum UTF-8 byte length")
        header = struct.pack(
            "!BIIIH",
            MessageType.SEND_SUBTITLE,
            message.currentTime,
            message.subtitleStartTime,
            message.subtitleEndTime,
            len(text),
        )
        data = header + text
    elif isinstance(message, CurrentTime):
        data = struct.pack("!BI", MessageType.CURRENT_TIME, message.currentTime)
    else:
        raise TypeError(f"Unsupported message: {type(message).__name__}")

    data += struct.pack("!I", zlib.crc32(data))
    return cobs.encode(data) + b"\x00"


def decode_frame(encoded_frame: bytes) -> Message:
    data = cobs.decode(encoded_frame)
    if len(data) < 5:
        raise ValueError("Frame too short")

    payload, crc_bytes = data[:-4], data[-4:]
    expected_crc = struct.unpack("!I", crc_bytes)[0]
    if zlib.crc32(payload) != expected_crc:
        raise ValueError("CRC32 mismatch")

    message_type = payload[0]
    if message_type == MessageType.CURRENT_TIME:
        if len(payload) != struct.calcsize("!BI"):
            raise ValueError("Invalid CurrentTime length")
        _, current_time = struct.unpack("!BI", payload)
        return CurrentTime(currentTime=current_time)

    if message_type == MessageType.SEND_SUBTITLE:
        header_size = struct.calcsize("!BIIIH")
        if len(payload) < header_size:
            raise ValueError("Truncated SendSubtitle header")
        _, current_time, start_time, end_time, text_length = struct.unpack(
            "!BIIIH", payload[:header_size]
        )
        if text_length > MAX_SUBTITLE_BYTES or len(payload) != header_size + text_length:
            raise ValueError("Invalid subtitle length")
        text = payload[header_size:].decode("utf-8")
        return SendSubtitle(current_time, start_time, end_time, text)

    raise ValueError(f"Unknown message type: {message_type}")


class MessageDecoder:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.discarding = False

    def push(self, data: bytes) -> list[Message]:
        messages: list[Message] = []
        for byte in data:
            if byte == 0:
                if self.buffer and not self.discarding:
                    try:
                        messages.append(decode_frame(bytes(self.buffer)))
                    except (ValueError, UnicodeDecodeError, cobs.DecodeError) as e:
                        print(f"Decoding error: {e}") 
                        pass  # Discard malformed frames and continue.
                self.buffer.clear()
                self.discarding = False
            elif not self.discarding:
                if len(self.buffer) >= MAX_ENCODED_FRAME_SIZE:
                    self.buffer.clear()
                    self.discarding = True  # Wait for next delimiter to resynchronize.
                else:
                    self.buffer.append(byte)
        return messages
