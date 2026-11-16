"""Offline tests for fault context in mixed DXL/STS traffic."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from bus import BusError,dxl_packet,dxl_status
from core import Fault
from primitive_hardware import Hardware


class Port:
    timeout=.003
    def __init__(self,data=b''):self.data=data
    def read(self,n):
        if not self.data:
            time.sleep(self.timeout)
            return b''
        result,self.data=self.data[:n],self.data[n:]
        return result


class DxlTraceTests(unittest.TestCase):
    def test_valid_status_still_returns_only_data(self):
        packet=dxl_packet(1,0x55,b'\x00\x2a')
        self.assertEqual(dxl_status(Port(packet),1),b'\x2a')
    def test_empty_response_reports_no_bytes(self):
        with self.assertRaisesRegex(BusError,'received bytes=<none>'):
            dxl_status(Port(),1)
    def test_non_dxl_bytes_are_recorded_on_header_timeout(self):
        with self.assertRaisesRegex(BusError,'received bytes=ff ff 0a 02 00 f3'):
            dxl_status(Port(bytes.fromhex('ff ff 0a 02 00 f3')),1)
    def test_corrupt_crc_keeps_raw_status(self):
        packet=bytearray(dxl_packet(1,0x55,b'\x00\x2a'));packet[-1]^=1
        with self.assertRaisesRegex(BusError,'CRC.*received bytes=ff ff fd 00'):
            dxl_status(Port(bytes(packet)),1)
    def test_wrong_id_is_not_accepted(self):
        packet=dxl_packet(2,0x55,b'\x00\x2a')
        with self.assertRaisesRegex(BusError,'received bytes=ff ff fd 00'):
            dxl_status(Port(packet),1)
    def test_finger_read_failure_identifies_wrist_initialization_stage(self):
        h=Hardware.__new__(Hardware)
        h.c={'min_tick':352,'max_tick':3424,'reference_tick':2400,'tolerance':10}
        h.bus=Mock();h.bus.read_diagnostic.side_effect=BusError('no header; received bytes=<none>')
        h.guard=Mock();h._pos=lambda:2400;h._read=lambda addr,n:1
        h._goal=Mock();h.wrist_armed=False
        with patch('primitive_hardware.sts_write'):
            with self.assertRaisesRegex(Fault,'wrist reference: finger sync read failed:.*received bytes=<none>'):
                h._move_wrist_reference(lambda:None)
        h.guard.check.assert_not_called()
        self.assertTrue(h.wrist_armed)

if __name__=='__main__':unittest.main()
