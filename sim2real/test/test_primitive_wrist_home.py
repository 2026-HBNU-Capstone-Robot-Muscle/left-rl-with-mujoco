"""Motor-free checks for the explicit, unloaded wrist return."""
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from core import Fault
from primitive_hardware import Hardware


class WristHomeTests(unittest.TestCase):
    def hardware(self):
        h=Hardware.__new__(Hardware)
        h.bus=None
        h.cfg={'port':'unused','baud':1000000}
        h.c={'reference_tick':2400,'tolerance':10}
        return h

    def test_return_uses_current_hold_then_reference_and_turns_off(self):
        h=self.hardware()
        bus=Mock()
        bus.read_register_sync.return_value={1:0,2:0,3:0,4:0}
        h._pos=Mock(side_effect=[3000,2400,2400,2400,2400])
        h._goal=Mock()
        torque=[0]
        h._read=Mock(side_effect=lambda addr,size:torque[0])
        def write(port,motor,addr,data,**kwargs):
            if addr==40:torque[0]=data[0]
        clock=[0.]
        def monotonic():
            clock[0]+=.1
            return clock[0]
        with patch('primitive_hardware.Bus',return_value=bus), \
             patch('primitive_hardware.sts_write',side_effect=write) as writes, \
             patch('primitive_hardware.time.monotonic',side_effect=monotonic), \
             patch('primitive_hardware.time.sleep'), \
             contextlib.redirect_stdout(io.StringIO()):
            h.return_wrist_to_reference()
        self.assertEqual([call.args[0] for call in h._goal.call_args_list],[3000,2400])
        self.assertEqual([call.args[3] for call in writes.call_args_list],[b'\x01',b'\x00'])
        self.assertIsNone(h.bus)
        bus.close.assert_called_once()

    def test_finger_torque_on_blocks_wrist_motion(self):
        h=self.hardware()
        bus=Mock()
        bus.read_register_sync.return_value={1:0,2:1,3:0,4:0}
        h._read=Mock(return_value=0)
        h._goal=Mock()
        with patch('primitive_hardware.Bus',return_value=bus), \
             patch('primitive_hardware.sts_write') as write, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(Fault,'finger torque must be OFF'):
                h.return_wrist_to_reference()
        h._goal.assert_not_called()
        self.assertEqual([call.args[3] for call in write.call_args_list],[b'\x00'])
        self.assertIsNone(h.bus)
        bus.close.assert_called_once()


if __name__=='__main__':unittest.main()
