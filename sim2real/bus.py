"""Standalone XL330 Protocol 2.0 transport. No wrist traffic or test-script dependency."""
import struct
import time
import serial
from core import Fault

# Protocol routines copied without behavioral changes from the original transport.
class BusError(RuntimeError):
    """모터 통신 오류."""


def dxl_crc(data: bytes) -> int:
    """패킷 손상 여부를 확인하는 DYNAMIXEL CRC-16을 계산한다."""
    crc = 0
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def dxl_stuff(body: bytes) -> bytes:
    """데이터 안의 헤더 모양이 실제 헤더로 오인되지 않게 바이트를 추가한다."""
    result = bytearray()
    for value in body:
        result.append(value)
        if result[-3:] == b"\xFF\xFF\xFD":
            result.append(0xFD)
    return bytes(result)


def dxl_unstuff(body: bytes) -> bytes:
    """수신 패킷에 추가됐던 stuffing 바이트를 제거한다."""
    return body.replace(b"\xFF\xFF\xFD\xFD", b"\xFF\xFF\xFD")


def dxl_packet(device_id: int, instruction: int, params: bytes = b"") -> bytes:
    """ID, 명령, 데이터를 DYNAMIXEL Protocol 2.0 패킷으로 만든다."""
    body = dxl_stuff(bytes([instruction]) + params)
    length = len(body) + 2
    packet = bytearray(b"\xFF\xFF\xFD\x00")
    packet.extend((device_id, length & 0xFF, length >> 8))
    packet.extend(body)
    packet.extend(struct.pack("<H", dxl_crc(packet)))
    return bytes(packet)


def read_exact(port: serial.Serial, count: int) -> bytes:
    """요청한 바이트 수를 받지 못하면 통신 오류로 처리한다."""
    data = port.read(count)
    if len(data) != count:
        raise BusError(f"응답 시간 초과: {count}바이트 중 {len(data)}바이트 수신")
    return data


def find_header(port: serial.Serial, header: bytes) -> None:
    """수신 데이터에서 프로토콜 응답 헤더가 나올 때까지 찾는다."""
    window = bytearray()
    deadline = time.monotonic() + port.timeout
    while time.monotonic() < deadline:
        value = port.read(1)
        if not value:
            continue
        window.extend(value)
        if len(window) > len(header):
            del window[0]
        if bytes(window) == header:
            return
    raise BusError("응답 헤더를 찾지 못했습니다")


def dxl_status(port: serial.Serial, expected_id: int) -> bytes:
    """XL330 응답의 ID, CRC, 장치 오류를 검사하고 데이터만 반환한다."""
    find_header(port, b"\xFF\xFF\xFD\x00")
    prefix = read_exact(port, 3)
    device_id = prefix[0]
    length = prefix[1] | prefix[2] << 8
    tail = read_exact(port, length)
    packet = b"\xFF\xFF\xFD\x00" + prefix + tail
    if dxl_crc(packet[:-2]) != int.from_bytes(tail[-2:], "little"):
        raise BusError("XL330 응답 CRC 오류")
    if device_id != expected_id:
        raise BusError(f"XL330 응답 ID 불일치: {device_id}")
    body = dxl_unstuff(tail[:-2])
    if len(body) < 2 or body[0] != 0x55:
        raise BusError("XL330 응답 형식 오류")
    if body[1]:
        raise BusError(f"XL330 ID {device_id} 장치 오류: 0x{body[1]:02X}")
    return body[2:]


def dxl_request(
    port: serial.Serial,
    device_id: int,
    instruction: int,
    params: bytes = b"",
    expect_reply: bool = True,
) -> bytes:
    """XL330 명령을 전송하고 필요한 경우 응답을 기다린다."""
    # 이전에 남은 수신 데이터를 지워 현재 명령의 응답만 읽는다.
    port.reset_input_buffer()
    port.write(dxl_packet(device_id, instruction, params))
    port.flush()
    return dxl_status(port, device_id) if expect_reply else b""


def dxl_read(port: serial.Serial, device_id: int, address: int, size: int) -> bytes:
    """XL330 제어 테이블의 값을 읽는다."""
    data = dxl_request(port, device_id, 0x02, struct.pack("<HH", address, size))
    if len(data) != size:
        raise BusError("XL330 READ 길이 오류")
    return data


def dxl_write(port: serial.Serial, device_id: int, address: int, data: bytes) -> None:
    """XL330 제어 테이블에 값을 기록한다."""
    dxl_request(port, device_id, 0x03, struct.pack("<H", address) + data)


class Bus:
    def __init__(self, port, baud):
        self.port = serial.Serial(port, baud, timeout=.05, write_timeout=.05)
        self.armed_ids = []

    def read_int(self, i, addr, size):
        return int.from_bytes(dxl_read(self.port, i, addr, size), 'little')

    def inventory(self):
        return [{'id': i, 'model_number': self.read_int(i, 0, 2),
                 'mode': self.read_int(i, 11, 1), 'torque': self.read_int(i, 64, 1),
                 'drive_mode': self.read_int(i, 10, 1),
                 'secondary_id': self.read_int(i, 12, 1),
                 'status_return_level': self.read_int(i, 68, 1),
                 'hardware_error': self.read_int(i, 70, 1),
                 'current_limit_ma': self.read_int(i, 38, 2),
                 'temperature_limit_c': self.read_int(i, 31, 1),
                 'shutdown_mask': self.read_int(i, 63, 1),
                 'min_position_limit_tick': self.read_int(i, 52, 4),
                 'max_position_limit_tick': self.read_int(i, 48, 4),
                 'velocity_limit_raw': self.read_int(i, 44, 4),
                 'position_p_gain': self.read_int(i, 84, 2),
                 'position_i_gain': self.read_int(i, 82, 2),
                 'position_d_gain': self.read_int(i, 80, 2),
                 'pwm_limit_raw': self.read_int(i, 36, 2),
                 'goal_pwm_raw': self.read_int(i, 100, 2),
                 'profile_velocity_raw': self.read_int(i, 112, 4),
                 'profile_acceleration_raw': self.read_int(i, 108, 4),
                 'goal_current_raw': int.from_bytes(dxl_read(self.port, i, 102, 2), 'little', signed=True),
                 'watchdog': self.read_int(i, 98, 1)} for i in range(1, 5)]

    def read(self):
        rows = []
        for i in range(1, 5):
            start = time.monotonic()
            raw = dxl_read(self.port, i, 126, 21)
            current, velocity, position = struct.unpack_from('<hii', raw)
            rows.append({'id': i, 'sample_start_s': start, 'read_end_s': time.monotonic(),
                         'position_tick': position, 'velocity_tick_s': velocity * .229 * 4096 / 60,
                         'current_ma': current, 'temperature_c': raw[20]})
        return rows

    def write_checked(self, i, addr, value, size):
        dxl_write(self.port, i, addr, value.to_bytes(size, 'little'))
        actual=self.read_int(i, addr, size)
        if actual != value:
            raise Fault(f'ID {i}: write readback mismatch at {addr}: expected={value}, actual={actual}')

    def arm(self, cfg, selected, rows):
        # All preconditions are read before the first write. Mode changes are manual.
        inv = self.inventory()
        for item in inv:
            m = next(m for m in cfg['motors'] if m['id'] == item['id'])
            if item['model_number'] != m['model_number'] or item['torque'] != 0:
                raise Fault('Confirmed model and torque OFF on all four motors required')
            if item['mode'] != 5 or item['drive_mode'] != 0:
                raise Fault('Requires preconfigured mode 5, Drive Mode 0; no automatic mode changes')
            if item['secondary_id'] != 255 or item['status_return_level'] != 2 or item['hardware_error']:
                raise Fault('Requires secondary ID disabled, status return level 2, no hardware error')
            if item['current_limit_ma'] > m['limits']['current_ma'] or item['current_limit_ma'] == 0:
                raise Fault('EEPROM current limit must be nonzero and within verified limit')
            if item['watchdog'] != 0:
                raise Fault('Existing watchdog state must be reviewed and cleared manually')
        # Register cleanup responsibility BEFORE any actuator mutation.
        self.armed_ids = list(selected)
        by_id = {r['id']: r for r in rows}
        for m in cfg['motors']:
            i, l = m['id'], m['limits']
            if i not in selected:
                continue
            saved_limit = next(x['current_limit_ma'] for x in inv if x['id'] == i)
            self.write_checked(i, 102, min(l['current_ma'], saved_limit), 2)
            self.write_checked(i, 112, l['profile_velocity_raw'], 4)
            self.write_checked(i, 116, by_id[i]['position_tick'], 4)
            self.write_checked(i, 98, cfg['watchdog_ticks'], 1)
        # Fresh positions must remain stable while preparing the command.
        fresh = self.read()
        if any(abs(r['position_tick'] - by_id[r['id']]['position_tick']) > 1 for r in fresh):
            raise Fault('Position changed during arming; abort')
        for i in selected:
            self.write_checked(i, 64, 1, 1)

    def command(self, positions, selected, rows, max_age):
        oldest = min(r['sample_start_s'] for r in rows)
        for i in selected:
            if time.monotonic() - oldest > max_age:
                raise Fault('Stale state before write')
            if self.read_int(i, 70, 1):
                raise Fault(f'ID {i}: hardware error')
            if time.monotonic() - oldest > max_age:
                raise Fault('State expired during hardware error read')
            self.write_checked(i, 116, positions[i], 4)

    def stop(self):
        errors = []
        # Torque-off is a selected, verified stop strategy, not a universal safe pose.
        for i in self.armed_ids:
            try:
                self.write_checked(i, 64, 0, 1)
            except Exception as exc:
                errors.append(f'ID {i}: torque-off UNCONFIRMED: {exc}')
        return errors

    def close(self):
        self.port.close()

    def read_sync(self):
        """Single Sync READ request; four CRC/ID checked replies. No register writes.
        Shared request time is conservative, not a simultaneous sampling claim.
        """
        start = time.monotonic()
        self.port.reset_input_buffer()
        packet = dxl_packet(254, 0x82, struct.pack('<HH', 126, 21) + bytes([1,2,3,4]))
        if self.port.write(packet) != len(packet):
            raise Fault('Incomplete sync-read request')
        self.port.flush()
        rows = []
        for i in range(1,5):
            raw = dxl_status(self.port, i)
            if len(raw) != 21:
                raise Fault('Sync response length mismatch')
            current, velocity, position = struct.unpack_from('<hii', raw)
            rows.append({'id':i, 'sample_start_s':start, 'read_end_s':time.monotonic(),
                         'position_tick':position, 'velocity_tick_s':velocity*.229*4096/60,
                         'current_ma':current, 'temperature_c':raw[20]})
        return rows

    def read_register_sync(self, address, size):
        self.port.reset_input_buffer()
        packet = dxl_packet(254, 0x82, struct.pack('<HH', address, size) + bytes([1,2,3,4]))
        if self.port.write(packet) != len(packet):
            raise Fault('Incomplete register sync-read request')
        self.port.flush()
        values = {}
        for i in range(1,5):
            raw = dxl_status(self.port, i)
            if len(raw) != size:
                raise Fault('Register sync response length mismatch')
            values[i] = int.from_bytes(raw, 'little')
        return values

    def command_sync(self, positions, rows, max_age):
        # Validate every target before any goal write. Broadcast addresses only IDs 1..4.
        if set(positions) != {1,2,3,4} or len(rows) != 4 or {r['id'] for r in rows} != {1,2,3,4}:
            raise Fault('Incomplete four-motor command/state')
        if any(not isinstance(v, int) or not 0 <= v <= 4095 for v in positions.values()):
            raise Fault('Invalid position target')
        oldest = min(r['sample_start_s'] for r in rows)
        def fresh():
            if time.monotonic() - oldest > max_age:
                raise Fault('Stale state before sync write')
        fresh()
        errors = self.read_register_sync(70, 1)
        if any(errors.values()):
            raise Fault(f'Hardware error before sync write: {errors}')
        fresh()
        params = struct.pack('<HH',116,4) + b''.join(bytes([i]) + struct.pack('<I',positions[i]) for i in range(1,5))
        packet = dxl_packet(254, 0x83, params)
        if self.port.write(packet) != len(packet):
            raise Fault('Incomplete sync goal write; partial delivery possible')
        self.port.flush()
        # Sync WRITE has no acknowledgement; verify all goal registers explicitly.
        if self.read_register_sync(116, 4) != positions:
            raise Fault('Sync goal readback mismatch')

    def read_diagnostic(self):
        """Read goal, trajectory, feedback and output in one Sync Read."""
        start=time.monotonic()
        self.port.reset_input_buffer()
        packet=dxl_packet(254,0x82,struct.pack('<HH',116,31)+bytes([1,2,3,4]))
        if self.port.write(packet)!=len(packet): raise Fault('Incomplete diagnostic read')
        self.port.flush()
        rows=[]
        for i in range(1,5):
            raw=dxl_status(self.port,i)
            if len(raw)!=31: raise Fault('Diagnostic response length mismatch')
            goal=struct.unpack_from('<i',raw,0)[0]
            pwm,current,velocity,position,veltraj,postraj=struct.unpack_from('<hhiiii',raw,8)
            rows.append({'id':i,'sample_start_s':start,'read_end_s':time.monotonic(),
                'position_tick':position,'velocity_tick_s':velocity*.229*4096/60,
                'current_ma':current,'temperature_c':raw[30],
                'goal_position_tick':goal,'position_trajectory_tick':postraj,
                'velocity_trajectory_tick_s':veltraj*.229*4096/60,
                'present_pwm_raw':pwm,'input_voltage_v':struct.unpack_from('<H',raw,28)[0]*.1,
                'moving_status_raw':raw[7],'profile_ongoing':bool(raw[7]&2),
                'goal_error_tick':goal-position,'trajectory_error_tick':postraj-position})
        return rows

    def prepare_hold(self, i, reference):
        """Refresh initial hold only with torque OFF; never retry a live goal."""
        last=None
        for attempt in range(3):
            if self.read_int(i,64,1)!=0: raise Fault('Hold preparation requires torque OFF')
            current=self.read_int(i,132,4)
            if not 0<=current<=4095 or abs(current-reference)>2:
                raise Fault(f'ID {i}: position changed during hold preparation: reference={reference}, actual={current}')
            try:
                self.write_checked(i,116,current,4)
                return current
            except Fault as exc:
                if 'write readback mismatch at 116:' not in str(exc): raise
                last=exc
        raise last