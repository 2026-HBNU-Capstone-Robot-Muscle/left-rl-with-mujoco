"""Motor-free tests for the explicit CLI home command."""
import contextlib
import io
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import primitive_cli


class HomeTests(unittest.TestCase):
    def test_home_after_fault_closes_owner_then_fingers_then_wrist(self):
        worker=Mock();worker.close.return_value=True
        hardware=Mock()
        order=[]
        worker.close.side_effect=lambda:order.append('close') or True
        hardware.return_wrist_to_reference.side_effect=lambda:order.append('wrist')
        with patch('primitive_cli.subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()):
            run.side_effect=lambda *args,**kwargs:order.append('fingers') or Mock(returncode=0)
            self.assertEqual(primitive_cli.explicit_home_after_stop(worker,hardware),(True,True))
        self.assertEqual(order,['close','fingers','wrist'])
        command=run.call_args.args[0]
        self.assertEqual(command[0],sys.executable)
        self.assertEqual(Path(command[1]).parent, Path(primitive_cli.__file__).resolve().parent)
        self.assertEqual(command[2],'--execute')

    def test_home_does_not_move_if_stop_unconfirmed(self):
        worker=Mock();worker.close.return_value=False
        hardware=Mock()
        with patch('primitive_cli.subprocess.run') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(primitive_cli.explicit_home_after_stop(worker,hardware),(False,False))
        run.assert_not_called()
        hardware.return_wrist_to_reference.assert_not_called()

    def test_home_does_not_move_wrist_if_finger_return_fails(self):
        worker=Mock();worker.close.return_value=True
        hardware=Mock()
        with patch('primitive_cli.subprocess.run',return_value=Mock(returncode=1)), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(primitive_cli.explicit_home_after_stop(worker,hardware),(True,False))
        hardware.return_wrist_to_reference.assert_not_called()

    def test_home_while_holding_releases_fingers_then_wrist_without_port(self):
        output=io.StringIO()
        with patch.object(sys,'argv',['primitive_cli.py']), \
             patch('builtins.input',side_effect=['grasp','home']), \
             patch('primitive_cli.subprocess.run') as run, \
             contextlib.redirect_stdout(output):
            self.assertEqual(primitive_cli.main(),0)
        self.assertIn("FINGER RETURN RESULT: (True, '', 'open verified')",output.getvalue())
        self.assertIn('SIM: wrist to reference 2400; no motor output',output.getvalue())
        run.assert_not_called()


if __name__=='__main__':unittest.main()
