"""Offline regression tests for mixed-bus emergency-stop recovery."""
import struct
import time
import unittest
from bus import Bus, dxl_packet

class SimPort:
    def __init__(self,missing=None,enabled=None):
        self.timeout=.01;self.data=b'';self.writes=[];self.missing=missing;self.enabled=enabled
    def write(self,packet):
        self.writes.append(packet)
        motor=packet[4];instruction=packet[7]
        if motor==254 and instruction==0x83:
            # Delayed replies from the failed operation, not stop confirmation.
            self.data+=b''.join(dxl_packet(i,0x55,b'\x00\x00') for i in (1,2,3,4))
        elif instruction==2 and motor!=self.missing:
            value=1 if motor==self.enabled else 0
            self.data+=dxl_packet(motor,0x55,bytes([0,value]))
        return len(packet)
    def flush(self):pass
    def reset_input_buffer(self):self.data=b''
    def read(self,n):
        if not self.data:
            time.sleep(self.timeout)
            return b''
        out,self.data=self.data[:n],self.data[n:]
        return out

class StopTransportTests(unittest.TestCase):
    def make(self,**kwargs):
        bus=Bus.__new__(Bus);bus.armed_ids=[1,2,3,4];bus.port=SimPort(**kwargs);return bus
    def test_stop_broadcasts_before_fresh_individual_verification(self):
        b=self.make();self.assertEqual(b.stop(),[])
        expected=dxl_packet(254,0x83,struct.pack('<HH',64,1)+b''.join(bytes([i,0]) for i in range(1,5)))
        self.assertEqual(b.port.writes[0],expected)
        self.assertEqual([p[4] for p in b.port.writes[1:]],[1,2,3,4])
    def test_missing_response_is_not_treated_as_confirmed(self):
        b=self.make(missing=2);errors=b.stop()
        self.assertEqual(len(errors),1)
        self.assertIn('ID 2: torque-off UNCONFIRMED',errors[0])
        self.assertEqual(b.port.writes[-1][4],4)
    def test_report_when_motor_still_enabled(self):
        b=self.make(enabled=3);errors=b.stop()
        self.assertEqual(len(errors),1)
        self.assertIn('ID 3: torque-off UNCONFIRMED',errors[0])

class DiagnosticTests(unittest.TestCase):
    def test_torque_on_skips_mixed_reads(self):
        from unittest.mock import Mock, patch
        from diagnose_bus import diagnose
        bus=Mock();bus.read_int.side_effect=[0,0,1,0]
        with patch('diagnose_bus.sts_read',return_value=b'\x00'):
            result=diagnose(bus,3)
        self.assertIn('skipped',result)
        bus.read_diagnostic.assert_not_called()

    def test_read_only_mixed_diagnostics(self):
        from unittest.mock import Mock, patch
        from diagnose_bus import diagnose
        bus=Mock();bus.read_int.return_value=0
        bus.read_diagnostic.return_value=[{'id':i,'position_tick':2000+i} for i in range(1,5)]
        def status(port,i,addr,size):return bytes(size)
        with patch('diagnose_bus.sts_read',side_effect=status), patch('diagnose_bus.time.sleep'):
            result=diagnose(bus,3)
        self.assertEqual(result['counts'],{'finger_ok':3,'wrist_ok':3})
        self.assertEqual(result['errors'],[])
        bus.send_torque_off.assert_not_called()
        bus.write_checked.assert_not_called()
        bus.command_sync.assert_not_called()

class DangerousDiagnosticDisabledTests(unittest.TestCase):
    def test_old_wrist_goal_diagnostic_rejected_before_port_open(self):
        import sys
        from unittest.mock import patch
        import diagnose_bus
        with patch.object(sys,'argv',['diagnose_bus.py','--wrist-goal-check']), patch('diagnose_bus.Bus') as bus:
            with self.assertRaises(SystemExit):diagnose_bus.main()
        bus.assert_not_called()

if __name__=='__main__':unittest.main()
