# 좌수 PPO 파지·손목 회전: 실기 확인 결과와 미들웨어 연결 계획

이 문서는 left-rl-with-mujoco의 sim2real 코드로 확인한 동작, 단독 실행 방법, 파일 역할, middleware-bridge에 연결할 작업을 정리합니다. 실행 명령은 Windows PowerShell에서 left-rl-with-mujoco 저장소 루트를 기준으로 합니다.

## 1. 미들웨어 연결 전까지 완료한 작업

- 손가락 XL330 모터 ID1~4의 실제 위치·속도를 읽어 8개 관측값으로 변환하고, finger_robot_ppo_best.zip의 4개 출력을 모터별 위치 명령으로 변환합니다. 정책과 실기 제어의 명목 주기는 30ms입니다.
- 손가락 초기 위치는 ID1=1299, ID2=2749, ID3=2378, ID4=2292틱입니다. 접힘 경계는 ID1=705, ID2=2000, ID3=1700, ID4=1700틱입니다. 정책 목표 위치는 매 주기 바뀝니다. 접힘 경계가 항상 전송되는 고정 목표는 아닙니다.
- 손목 STS3215 ID10의 기준은 2400틱이고 시계 방향은 틱 감소입니다. 현재 손목 명령 범위는 352~3538틱입니다. 논리적 90도 회전은 손목을 50도씩 두 번 움직이고 중간에 네 손가락을 100틱씩 더 감아 2초간 재파지합니다. 논리적 180도 회전은 45도씩 네 번 움직이며 중간에 세 번 재파지합니다.
- 독립 실행기 primitive_cli.py에서 파지, 파지 유지, 회전, 해제 및 초기 위치 복귀를 구현했습니다. 사용자가 해당 실기 동작을 확인했습니다. 통신 실패·오래된 피드백·장치 오류·전류와 온도 제한·명령 확인 실패·시간 초과 때는 정지를 시도하고 토크 OFF 결과를 확인합니다.
- 모터 출력 없는 단위 테스트 70개가 통과했습니다. 이 결과와 실기 CLI 확인은 ROS 액션 경로의 실기 시험을 대신하지 않습니다.

파지 성공 판정은 네 모터의 엔코더 위치가 설정한 grasp_ticks 범위에서 안정된 경우입니다. 별도 접촉 센서가 없으므로 이 판정만으로 큐브 접촉력이나 미끄러짐을 측정할 수는 없습니다. 손목의 기준·방향·회전 방식은 현재 조립 상태에서 확인한 값을 사용합니다.

## 2. 파일 역할

| 위치 | 역할 |
| --- | --- |
| model/finger_robot_ppo_best.zip | 현재 실기에서 사용하는 PPO 모델. 설정 파일의 SHA-256과 일치해야 합니다. |
| sim2real/config.provisional.json | COM 포트, 모터 ID·순서, 손가락 초기 위치·접힘 경계, PPO 좌표 변환과 정책 주기. |
| sim2real/primitive_calibration.local.json | 손목 기준·방향·범위, 손가락 파지 완료 판정 범위와 재파지 설정. 실제 단독 시험에 사용한 파일입니다. |
| sim2real/primitive_calibration.example.json | 새 장비의 값을 기록할 때 사용하는 빈 양식. |
| sim2real/primitive_cli.py | 키보드 명령으로 파지·회전·해제·복귀를 시험하는 독립 실행기. ROS를 사용하지 않습니다. |
| sim2real/primitive_worker.py | 파지 유지, 명령 순서·실행 ID, 완료·취소·시간 초과·오류 상태 관리. |
| sim2real/primitive_hardware.py | 실측 피드백, PPO 추론, 손가락·손목 명령, 재파지와 정지 처리. |
| sim2real/wrist_protocol.py | 손목 STS3215의 패킷 작성·응답 검사·위치 명령. |
| sim2real/apply_policy.py, bus.py, core.py, run.py, position_compensation.py | 손가락 초기 이동, DYNAMIXEL 통신, 관측·행동 변환, PPO 로딩과 위치 오차 보정. primitive_hardware.py가 이 기능들을 재사용합니다. |
| sim2real/return_fingers.py | 명시적 home 명령에서 오류 후 손가락 네 개를 저장된 초기 위치로 복귀시키는 도구. |
| sim2real/primitive_server.py | 같은 Worker와 Hardware를 ROS 액션에 연결하기 위해 작성한 서버 초안. ROS 실기 호출은 아직 검증하지 않았습니다. |
| sim2real/diagnose_bus.py, test/ | 통신 상태 진단과 모터 출력 없는 테스트. |

initial_pose.open3.json은 초기 자세를 기록한 파일입니다. 현재 실행 코드가 실제로 읽는 초기 틱은 config.provisional.json의 initial_position_tick입니다. 손가락 파지 완료 판정에 쓰는 grasp_ticks는 모터에 전송하는 목표 틱이 아닙니다.

## 3. 실기 단독 실행 방법

저장소 루트에서 가상환경을 활성화하고 의존성을 설치합니다. 기존 가상환경이 있으면 생성 단계는 생략합니다.

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt

먼저 모터 명령을 보내지 않는 CLI 모드로 명령 순서를 확인합니다.

    python .\sim2real\primitive_cli.py

현재 실기 설정으로 실행할 때는 다음 명령을 사용합니다. 이 명령은 실제 모터를 움직입니다.

    python .\sim2real\primitive_cli.py --execute --allow-provisional --calibration .\sim2real\primitive_calibration.local.json

손가락 PPO만 10초간 별도로 확인하려면 다음 명령을 사용합니다. 이 명령도 실제 모터를 움직이며 손목 회전은 실행하지 않습니다.

    python .\sim2real\apply_policy.py --execute --seconds 10 --policy-period-ms 30 --position-p 1000 --position-i 0 --profile-velocity 80 --profile-acceleration 40 --position-compensation 100

CLI에서 입력 가능한 명령은 grasp, rotate 1, rotate -1, rotate 2, release, home, quit입니다. 예를 들어 grasp 성공 결과를 확인한 다음 rotate 1을 실행하고, 회전 결과를 확인한 뒤 release를 실행합니다. rotate 1은 시계 방향, rotate -1은 반시계 방향, rotate 2는 시계 180도 요청입니다.

- grasp: 손가락을 저장된 초기 위치로 옮기고 손목을 2400틱으로 맞춘 다음 PPO 파지를 시작합니다. 완료 응답 뒤에도 손가락 제어를 계속합니다.
- rotate: 파지를 유지하며 손목을 구간별로 회전하고 중간에 재파지합니다. 완료 판정은 손목 위치와 손가락 엔코더 상태를 사용합니다.
- release: PPO 파지를 중단하고 손가락 네 개를 초기 위치로 펼친 뒤 토크를 끕니다. 손목을 2400틱으로 되돌리지는 않습니다.
- home: 손가락을 먼저 초기 위치로 펼치고 토크 OFF한 뒤 손목을 2400틱으로 복귀시키고 종료합니다. 오류 후에는 사용자가 home을 명시적으로 입력했을 때만 복귀를 시도합니다. 정지 확인이 실패하면 복귀 명령을 보내지 않습니다.
- quit: 파지 중이면 손가락 release를 시도한 뒤 종료합니다.

현재 config.provisional.json은 COM8, 1Mbps를 지정합니다. 다른 PC 또는 조립 상태에서는 포트·모터 ID·초기 틱이 맞는지 확인합니다. primitive_calibration.local.json의 verified=false 때문에 CLI 실기 시험에는 --allow-provisional이 필요합니다. 손목 확인 결과를 버리라는 의미가 아니라, 파지 판정 등 전체 설정의 최종 검증 표시가 아직 분리되어 있지 않기 때문입니다.

모터 출력 없는 테스트는 저장소 루트에서 실행합니다.

    python -B -m unittest discover -s .\sim2real -p "test_*.py" -q

## 4. middleware-bridge에 연결할 작업

현재 middleware-bridge/src/control/custom_hand_control/custom_hand_control/primitive_server.py는 dry-run 전용이며 dry_run=false를 거부합니다. 이 저장소의 sim2real/primitive_server.py는 같은 ROS 액션 이름을 사용하는 별도 서버 초안입니다. 두 서버를 동시에 실행하지 않습니다. 현재 보정 파일은 verified=false이므로 이 초안도 현 설정 그대로는 실기 모드로 시작되지 않습니다.

미들웨어에서는 다음 순서로 통합합니다.

1. 현재 실기에서 사용한 PPO ZIP과 SHA-256을 맞춥니다. 미들웨어에 들어 있는 기존 ZIP은 현재 실기 ZIP과 다르므로 모델을 교체하고 관측 8개·행동 4개 규격을 다시 확인합니다.
2. Worker, Hardware, 손가락 통신·정책 변환 함수, 손목 프로토콜을 custom_hand_control 패키지에 맞게 이식합니다. Python import와 모델·설정 파일 경로를 ROS 패키지 설치 구조에 맞추고 setup.py와 package.xml의 데이터·의존성을 갱신합니다. 손가락과 손목이 공유하는 COM 포트는 한 제어기에서 관리합니다.
3. 기존 /custom_hand/execute_primitive 서버의 grasp_upper_layer, rotate_upper_layer, release_upper_layer 요청을 Worker에 연결합니다. 파지 완료 응답 후에도 제어를 유지하고, 회전·해제는 같은 execution_id의 순서로 처리합니다. 완료·실패·취소·시간 초과와 토크 OFF 확인 결과를 ROS 응답에 반영합니다.
4. 현재 보정 파일의 verified 표시와 ROS 서버 시작 조건을 실제 확인 결과에 맞게 정리합니다. 손목 값은 실기에서 확인한 그대로 사용합니다. 손가락 엔코더 파지 판정과 큐브 접촉력은 다른 검증 항목으로 구분합니다.
5. 모터 출력 없는 ROS 요청 시험을 거쳐, ROS 액션으로 파지→회전→해제를 실제 한 번씩 시험합니다. 각 결과를 확인한 뒤 미들웨어의 전체 실행 순서에 연결합니다.

카메라 기반 큐브 위치 추정과 다른 손의 제어는 이 단독 좌수 코드에 포함되어 있지 않습니다. 이 문서의 실기 확인 범위는 좌수 CLI이며, 미들웨어 ROS 경로의 실기 완료를 뜻하지 않습니다.
