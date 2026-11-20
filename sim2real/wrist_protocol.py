"""STS3215 protocol from left-hand-model e33f873; caller owns port."""
import struct
import serial
from bus import BusError,find_header,read_exact
STS_ACC=41
def sts_packet(device_id: int, instruction: int, params: bytes = b"") -> bytes:
    """ID, 명령, 데이터를 FEETECH STS 패킷으로 만든다."""
    length = len(params) + 2
    checksum = (~(device_id + length + instruction + sum(params))) & 0xFF
    return b"\xFF\xFF" + bytes([device_id, length, instruction]) + params + bytes([checksum])


class _ReadTrace:
    """Capture a bounded amount of received data for failed status reads only."""
    def __init__(self, port):
        self.port=port
        self.timeout=port.timeout
        self.data=bytearray()
    def read(self, count):
        data=self.port.read(count)
        self.data.extend(data)
        if len(self.data)>128: del self.data[:-128]
        return data


def sts_status(port: serial.Serial, expected_id: int) -> bytes:
    """Read one complete, validated STS status, including raw data on failure."""
    trace=_ReadTrace(port)
    try:
        return _sts_status_inner(trace,expected_id)
    except BusError as exc:
        raise BusError(f'{exc}; received bytes={trace.data.hex(" ") or "<none>"}') from exc


def _sts_status_inner(port: serial.Serial, expected_id: int) -> bytes:
    try:
        find_header(port, b"\xFF\xFF")
    except BusError as exc:
        raise BusError(f"STS3215 ID {expected_id}: {exc}") from exc
    device_id, length = read_exact(port, 2)
    tail = read_exact(port, length)
    if device_id != expected_id or length < 2:
        raise BusError("STS 응답 형식 오류")
    error, params, checksum = tail[0], tail[1:-1], tail[-1]
    expected = (~(device_id + length + error + sum(params))) & 0xFF
    if checksum != expected:
        packet=(b"\xff\xff"+bytes([device_id,length])+tail).hex(' ')
        raise BusError(f"STS ID {device_id} 응답 체크섬 오류: received=0x{checksum:02x}, expected=0x{expected:02x}, packet={packet}")
    if error:
        raise BusError(f"STS ID {device_id} 장치 오류: 0x{error:02X}")
    return params


def sts_request(
    port: serial.Serial,
    device_id: int,
    instruction: int,
    params: bytes = b"",
    expect_reply: bool = True,
) -> bytes:
    """STS 명령을 전송하고 필요한 경우 응답을 기다린다."""
    port.reset_input_buffer()
    packet=sts_packet(device_id, instruction, params)
    if port.write(packet)!=len(packet): raise BusError("Incomplete STS write")
    port.flush()
    return sts_status(port, device_id) if expect_reply else b""


def sts_ping(port: serial.Serial, device_id: int) -> None:
    """지정한 STS3215가 통신 버스에 있는지 확인한다."""
    sts_request(port, device_id, 0x01)


def sts_read(port: serial.Serial, device_id: int, address: int, size: int) -> bytes:
    """STS3215 제어 테이블의 값을 읽는다."""
    try:
        data = sts_request(port, device_id, 0x02, bytes([address, size]))
        # A late WRITE status can precede the requested READ status. Consume
        # at most one and still require the full, checksum-verified response.
        if not data:
            data = sts_status(port, device_id)
        if len(data) != size:
            raise BusError(f"expected {size} bytes, received {len(data)}")
        return data
    except BusError as exc:
        raise BusError(f"STS ID {device_id} READ addr={address}: {exc}") from exc


def sts_write(port: serial.Serial, device_id: int, address: int, data: bytes, *, expect_reply: bool = False) -> None:
    """Write a register. When enabled, consume and validate the write status first."""
    reply=sts_request(port, device_id, 0x03, bytes([address]) + data, expect_reply=expect_reply)
    if expect_reply and reply:
        raise BusError(f"STS ID {device_id} WRITE addr={address}: unexpected status data ({len(reply)} bytes)")


def sts_write_position(
    port: serial.Serial,
    device_id: int,
    position: int,
    speed: int,
    acceleration: int,
) -> None:
    """STS3215 위치, 속도, 가속도를 한 번에 전송한다."""
    # 주소 41부터 가속도, 위치, 시간, 속도를 연속 기록한다.
    data = bytes([acceleration]) + struct.pack("<HHH", position, 0, speed)
    sts_write(port, device_id, STS_ACC, data, expect_reply=True)
