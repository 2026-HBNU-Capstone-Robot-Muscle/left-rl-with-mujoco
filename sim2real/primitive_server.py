"""Custom hand action endpoint. Hardware requires explicit startup configuration."""

import rclpy
import re
import math
import sys
from pathlib import Path
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from system_interfaces.action import ExecuteHandPrimitive


class CustomHandPrimitiveServer(Node):
    """Validate requests for the Custom Hand primitive endpoint."""

    def __init__(self):
        super().__init__('left_hand_primitive_server')
        self.worker = None
        self.declare_parameter('hardware_enabled', False)
        self.declare_parameter('wrist_calibration', '')
        self.declare_parameter('primitive_timeout_s', 30.0)
        self.timeout = self.get_parameter('primitive_timeout_s').value
        if not 0 < self.timeout <= 120:
            raise ValueError('primitive_timeout_s must be in (0,120]')
        if self.get_parameter('hardware_enabled').value:
            directory = Path(__file__).resolve().parent
            if not (directory/'primitive_hardware.py').is_file():
                raise ValueError('Missing primitive_hardware.py beside this server')
            sys.path.insert(0, str(directory))
            from primitive_worker import Worker
            from primitive_hardware import Hardware
            device = Hardware(self.get_parameter('wrist_calibration').value)
            self.worker = Worker(device, period=device.cfg['period_s'],
                                 report=self.get_logger().warning)
        self._server = ActionServer(
            self,
            ExecuteHandPrimitive,
            '/custom_hand/execute_primitive',
            callback_group=ReentrantCallbackGroup(),
            execute_callback=self._execute,
            goal_callback=self._goal_request,
            cancel_callback=self._cancel_request,
        )
        self.get_logger().info(
            f'Ready: /custom_hand/execute_primitive hardware_enabled={self.worker is not None}')

    @staticmethod
    def _goal_request(_goal_request):
        return GoalResponse.ACCEPT

    @staticmethod
    def _cancel_request(_goal_handle):
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        goal = goal_handle.request
        result = ExecuteHandPrimitive.Result()
        feedback = ExecuteHandPrimitive.Feedback()
        feedback.state = feedback.STATE_ACCEPTED
        feedback.message = 'Custom Hand primitive 요청을 확인했습니다.'
        goal_handle.publish_feedback(feedback)

        if (not goal.scan_id or not goal.execution_id or not goal.primitive_id
                or goal.move_index == 0 or goal.step_index == 0
                or not re.fullmatch(r"[URFDLB](?:2|')?", goal.cube_move)
                or goal.target_face not in ('U', 'R', 'F', 'D', 'L', 'B')
                or goal.primitive_id not in {
                    'grasp_upper_layer', 'rotate_upper_layer', 'release_upper_layer'}
                or (goal.primitive_id == 'rotate_upper_layer'
                    and goal.clockwise_quarter_turns not in (-1, 1, 2))
                or (goal.primitive_id != 'rotate_upper_layer'
                    and goal.clockwise_quarter_turns != 0)):
            result.success = False
            result.error_code = 'INVALID_PRIMITIVE'
            result.error_message = (
                '식별자, 유효한 face/move/primitive와 회전량이 필요합니다.')
            feedback.state = feedback.STATE_FAILED
            feedback.message = result.error_message
            goal_handle.publish_feedback(feedback)
            goal_handle.abort()
            return result

        if (goal.face_direction_order
                and (len(goal.face_direction_order) != 6
                     or set(goal.face_direction_order) != set('URFDLB')
                     or goal.target_face_direction_index >= 6
                     or goal.face_direction_order[
                         goal.target_face_direction_index] != goal.target_face)):
            result.success = False
            result.error_code = 'INVALID_FACE_DIRECTION_STATE'
            result.error_message = (
                'face_direction_order는 URFDLB의 순열이어야 하고 index의 면은 '
                'target_face와 일치해야 합니다.')
            feedback.state = feedback.STATE_FAILED
            feedback.message = result.error_message
            goal_handle.publish_feedback(feedback)
            goal_handle.abort()
            return result

        if (goal.policy_id
                and (goal.policy_id != 'custom_finger_ppo_v1'
                     or len(goal.policy_action_values) != 4
                     or not all(math.isfinite(value)
                                for value in goal.policy_action_values)
                     or any(abs(value) > 1.000001
                            for value in goal.policy_action_values))):
            result.success = False
            result.error_code = 'INVALID_POLICY_OUTPUT'
            result.error_message = (
                'Custom primitive에는 custom_finger_ppo_v1의 유한한 action 4개가 필요합니다.')
            feedback.state = feedback.STATE_FAILED
            feedback.message = result.error_message
            goal_handle.publish_feedback(feedback)
            goal_handle.abort()
            return result
        if not goal.policy_id and goal.policy_action_values:
            result.success = False
            result.error_code = 'INVALID_POLICY_OUTPUT'
            result.error_message = 'policy_id 없이 policy_action_values를 받을 수 없습니다.'
            feedback.state = feedback.STATE_FAILED
            feedback.message = result.error_message
            goal_handle.publish_feedback(feedback)
            goal_handle.abort()
            return result

        if goal_handle.is_cancel_requested:
            result.error_code = 'CANCELLED'
            result.error_message = '요청이 취소되었습니다.'
            feedback.state = feedback.STATE_CANCELLED
            feedback.message = result.error_message
            goal_handle.publish_feedback(feedback)
            goal_handle.canceled()
            return result

        if not goal.dry_run and self.worker is not None:
            from primitive_worker import Rejected
            try:
                job = self.worker.submit(goal.primitive_id, goal.clockwise_quarter_turns,
                                         goal.execution_id, self.timeout)
            except Rejected as exc:
                result.error_code = str(exc)
                result.error_message = str(exc)
                goal_handle.abort()
                return result
            feedback.state = feedback.STATE_EXECUTING
            feedback.message = goal.primitive_id
            goal_handle.publish_feedback(feedback)
            while not job.done.wait(.01):
                if goal_handle.is_cancel_requested:
                    job.cancel.set()
            result.success, result.error_code, result.error_message = job.result
            canceled = goal_handle.is_cancel_requested and result.error_code == 'CANCELLED'
            feedback.state = (feedback.STATE_CANCELLED if canceled else
                              feedback.STATE_COMPLETED if result.success else feedback.STATE_FAILED)
            feedback.message = result.error_message
            goal_handle.publish_feedback(feedback)
            if canceled:
                goal_handle.canceled()
            elif result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            return result

        if not goal.dry_run:
            result.success = False
            result.error_code = 'HARDWARE_NOT_CONFIGURED'
            result.error_message = (
                'Custom Hand 손가락 정책과 Feetech 손목 제어를 연결하지 않았습니다.')
            feedback.state = feedback.STATE_FAILED
            feedback.message = result.error_message
            goal_handle.publish_feedback(feedback)
            goal_handle.abort()
            return result

        result.success = True
        result.error_code = ''
        result.error_message = (
            'dry-run 요청과 custom_finger_ppo_v1 shadow action을 확인했습니다. '
            '모터 명령은 보내지 않았습니다.' if goal.policy_id else
            'dry-run 요청을 검증했습니다. 모터 명령은 보내지 않았습니다.')
        feedback.state = feedback.STATE_COMPLETED
        feedback.message = result.error_message
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = CustomHandPrimitiveServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.worker is not None:
            node.worker.close()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
