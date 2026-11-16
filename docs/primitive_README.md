# 좌수 PPO 파지·손목 회전 제어

이 문서는 left-rl-with-mujoco의 sim2real 독립 실기 시험과 ROS 액션 서버 초안을 설명합니다. 아래 PowerShell 명령은 이 저장소의 루트에서 실행합니다.

## 사용하는 파일

- sim2real/config.provisional.json: COM 포트, 모터 ID, 손가락 초기 위치·접힘 범위, 정책 변환 계수와 주기.
- sim2real/primitive_calibration.local.json: 손목 기준·방향·회전 범위와 손가락 파지 완료 판정 범위.
- model/finger_robot_ppo_best.zip: 실기 추론에 사용하는 PPO 모델. 설정 파일의 SHA-256과 일치해야 합니다.
- sim2real/primitive_cli.py: 키보드로 단독 시험하는 실행기.
- sim2real/primitive_server.py: 같은 제어기를 사용하는 ROS 액션 서버 초안.
- sim2real/return_fingers.py: 명시적 home 명령에서 오류 후 손가락을 펼치는 복구 도구.

현재 설정의 손가락 초기 위치는 ID1=1299, ID2=2749, ID3=2378, ID4=2292틱입니다. 접힘 경계는 ID1=705, ID2=2000, ID3=1700, ID4=1700틱입니다. 정책의 목표는 매 주기 달라지며, 접힘 경계가 항상 명령되는 고정 목표는 아닙니다. 파지 완료 판정에는 별도 grasp_ticks 범위를 사용합니다.

## 단독 실행

모터 출력 없는 CLI 확인:

    python .\sim2real\primitive_cli.py

실기 실행:

    python .\sim2real\primitive_cli.py --execute --allow-provisional --calibration .\sim2real\primitive_calibration.local.json

명령은 grasp, rotate 1, rotate -1, rotate 2, release, home, quit입니다. grasp는 손가락을 저장된 초기 위치로 옮기고 손목을 2400틱에 맞춘 뒤 PPO를 실행합니다. 파지 성공 응답 후에도 PPO 제어를 유지합니다. release는 네 손가락을 초기 위치로 펼치고 토크를 끕니다. home은 손가락을 먼저 펼치고 손목을 2400틱으로 돌린 뒤 토크를 끄고 CLI를 종료합니다. 오류 후 복귀는 home을 명시적으로 입력했을 때만 시도하며, 정지 확인에 실패하면 복귀를 시작하지 않습니다.

손목 기준은 2400틱, 시계 방향은 틱 감소입니다. 논리적 90도 요청은 손목을 50도씩 두 구간 움직이고 중간에 네 손가락을 100틱씩 더 감아 2초간 재파지합니다. 논리적 180도 요청은 45도씩 네 구간과 중간 재파지 세 번을 사용합니다. 현재 손목 속도 레지스터 값은 600입니다. 엔코더 위치만으로 큐브 접촉이나 회전 완료를 확인할 수는 없습니다.

기존 apply_policy.py를 독립 실행할 때도 모델은 이 저장소의 model/에서 읽습니다. 해당 파일은 손목 primitive를 실행하지 않습니다.

## ROS 서버 상태

sim2real/primitive_server.py는 /custom_hand/execute_primitive 액션을 제공하며 기본값은 하드웨어 비활성입니다. 현재 primitive_calibration.local.json의 verified=false 때문에 이 서버의 실기 모드는 시작되지 않습니다. ROS 액션으로 실제 모터를 움직인 시험도 아직 수행되지 않았습니다. middleware-bridge의 기존 Custom Hand 서버도 dry-run 전용입니다. 두 서버가 같은 액션 이름을 사용하므로 동시에 실행하지 않습니다.

## 모터 출력 없는 확인

    python -B -m unittest discover -s .\sim2real -p "test_*.py" -q

현재 설정은 COM8과 1 Mbps를 사용합니다. 다른 PC 또는 조립 상태에서는 포트·ID·초기 위치를 먼저 확인합니다. 설정의 verified=false는 유지하고, 실기 CLI 시험에는 --allow-provisional을 명시합니다.
