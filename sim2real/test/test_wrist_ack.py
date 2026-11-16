"""STS write/read sequencing with a memory port; no serial hardware."""
import time
import unittest
from types import SimpleNamespace
from bus import BusError
from primitive_hardware import Hardware
from wrist_protocol import sts_packet,sts_write


class FakePort:
    timeout=.003
    def __init__(self,drop_ack=False,write_error=0):
        self.data=b'';self.writes=[];self.discarded=[]
        self.drop_ack=drop_ack;self.write_error=write_error
    def reset_input_buffer(self):
        if self.data:self.discarded.append(self.data)
        self.data=b''
    def write(self,packet):
        self.writes.append(packet)
        instruction=packet[4]
        if instruction==3 and not self.drop_ack:
            self.data+=sts_packet(10,self.write_error)
        elif instruction==2:
            self.data+=sts_packet(10,0,(2400).to_bytes(2,'little'))
        return len(packet)
    def flush(self):pass
    def read(self,n):
        if not self.data:
            time.sleep(self.timeout)
            return b''
        data,self.data=self.data[:n],self.data[n:]
        return data


class WristAckTests(unittest.TestCase):
    def hardware(self,port):
        hardware=Hardware.__new__(Hardware)
        hardware.bus=SimpleNamespace(port=port)
        return hardware
    def test_goal_consumes_write_ack_before_requesting_readback(self):
        port=FakePort()
        self.hardware(port)._goal(2400)
        self.assertEqual([p[4] for p in port.writes],[3,2])
        self.assertEqual(port.discarded,[])
        self.assertEqual(port.data,b'')
    def test_missing_write_ack_stops_before_readback(self):
        port=FakePort(drop_ack=True)
        with self.assertRaises(BusError):self.hardware(port)._goal(2400)
        self.assertEqual([p[4] for p in port.writes],[3])
    def test_device_error_in_write_ack_is_not_ignored(self):
        port=FakePort(write_error=4)
        with self.assertRaisesRegex(BusError,'장치 오류'):
            self.hardware(port)._goal(2400)
        self.assertEqual([p[4] for p in port.writes],[3])
    def test_stop_write_can_be_sent_without_waiting_for_ack(self):
        port=FakePort(drop_ack=True)
        sts_write(port,10,40,b'\x00')
        self.assertEqual([p[4] for p in port.writes],[3])


if __name__=='__main__':unittest.main()
