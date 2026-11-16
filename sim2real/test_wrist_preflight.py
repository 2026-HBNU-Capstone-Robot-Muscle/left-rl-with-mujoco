"""Offline tests for the existing wrist torque-OFF startup requirement."""
import unittest
from unittest.mock import Mock,patch
from core import Fault
from primitive_hardware import Hardware


class WristPreflightTests(unittest.TestCase):
    def make_hardware(self):
        h=Hardware.__new__(Hardware)
        h.cfg={'port':'COM8','baud':1000000}
        h.c={'reference_tick':2400,'min_tick':352,'max_tick':3424}
        h.wrist_armed=False
        return h
    def test_wrist_on_is_not_automatically_switched_or_moved(self):
        h=self.make_hardware();h._read=Mock(return_value=1)
        bus=Mock();bus.inventory.return_value=[]
        with patch('primitive_hardware.Bus',return_value=bus),patch('primitive_hardware.preflight'),patch('primitive_hardware.sts_write') as write:
            with self.assertRaisesRegex(Fault,'wrist must start torque OFF'):
                h.start(lambda:None)
        write.assert_not_called()
        bus.read_diagnostic.assert_not_called()
        bus.write_checked.assert_not_called()
    def test_wrist_off_proceeds_to_position_validation_without_writes(self):
        h=self.make_hardware();h._read=Mock(return_value=0)
        h._pos=Mock(side_effect=RuntimeError('position validation reached'))
        bus=Mock();bus.inventory.return_value=[]
        with patch('primitive_hardware.Bus',return_value=bus),patch('primitive_hardware.preflight'),patch('primitive_hardware.sts_write') as write:
            with self.assertRaisesRegex(RuntimeError,'position validation reached'):
                h.start(lambda:None)
        write.assert_not_called()
        bus.write_checked.assert_not_called()

if __name__=='__main__':unittest.main()
