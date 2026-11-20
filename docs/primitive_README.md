# 좌수 PPO 파지·손목 회전: 실기 확인 결과와 미들웨어 연결 계획

이 문서는 left-rl-with-mujoco의 sim2real 코드로 확인한 동작, 단독 실행 방법, 파일 역할, middleware-bridge에 연결할 작업을 정리합니다. 실행 명령은 Windows PowerShell에서 left-rl-with-mujoco 저장소 루트를 기준으로 합니다.

## 1. 좌수 실기 검증과 현재 설정

- 손가락 XL330 모터 ID1~4의 실제 위치·속도를 읽어 8개 관측값으로 변환하고, finger_robot_ppo_best.zip의 4개 출력을 모터별 위치 명령으로 변환합니다. 정책과 실기 제어의 명목 주기는 30ms입니다.
- 손가락 초기 위치는 ID1=1299, ID2=2749, ID3=2378, ID4=2292틱입니다. 접힘 경계는 ID1=705, ID2=1800, ID3=1700, ID4=1700틱입니다. 정책 목표 위치는 매 주기 바뀝니다. 접힘 경계가 항상 전송되는 고정 목표는 아닙니다.
- 손목 STS3215 ID10의 기준은 2400틱이고 시계 방향은 틱 감소입니다. 임시 시험 모드에서 손목 명령 범위는 295~3538틱이고 피드백 허용 범위는 265~3568틱입니다. 논리적 90도 회전은 시계 방향 50°+45°, 반시계 방향 50°+50°로 손목을 움직이고 첫 50° 구간에서 손목 실측 진행률에 따라 네 손가락을 최대 100틱까지 나눠 감고, 중간에는 명령 피드백을 확인한 뒤 다음 구간으로 넘어갑니다. 두 번째 구간에서도 진행률에 따라 최대 50틱을 더 감습니다. 논리적 180도 요청의 손목 구간은 47.5° → 47.5° → 45° → 45°이며 첫 세 구간에서 손목 실측 진행률에 따라 구간마다 최대 75틱을 나눠 감고, 두 번째 구간 끝(큐브 기준 90°, 손목 누적 명령 95°)에서만 1초 유지합니다. 마지막 구간에서 50틱을 한 번 더 당기고 명령 피드백이 확인되면 완료 조건을 검사합니다. 2400틱에서 180도 요청의 최종 목표는 295틱입니다. 자동 여유를 적용한 범위이며, 기존 실기 확인 범위는 352~3538틱입니다.
- 독립 실행기 primitive_cli.py에서 파지, 파지 유지, 회전, 해제 및 초기 위치 복귀를 구현했습니다. 사용자가 해당 실기 동작을 확인했습니다. 통신 실패·오래된 피드백·장치 오류·전류와 온도 제한·명령 확인 실패·시간 초과 때는 정지를 시도하고 토크 OFF 결과를 확인합니다.
- 좌수 단독 ROS 액션으로 실제 장비에서 파지·회전·해제 및 손목 복귀를 확인했습니다. 미들웨어 GUI에서는 모터 출력 없이 합성 풀이의 완료와 취소를 확인했습니다.

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
| sim2real/primitive_server.py | 같은 Worker와 Hardware를 ROS 액션에 연결합니다. 기본 모드에서는 모터 출력을 비활성화합니다. |
| sim2real/session_logging.py, ros_test_commands.ps1 | 날짜별 ROS 로그 수집과 PowerShell 실기 요청 함수. |
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
- release: PPO 파지를 중단하고 손가락 네 개를 초기 위치로 펼치고 토크 OFF를 확인합니다. 손목을 2400틱으로 되돌린 뒤 손목 토크 OFF를 확인합니다.
- home: 손가락을 먼저 초기 위치로 펼치고 토크 OFF한 뒤 손목을 2400틱으로 복귀시키고 종료합니다. 정지 확인이 실패하면 home 복귀 명령을 보내지 않습니다.
- quit: 파지 중이면 손가락 release를 시도한 뒤 종료합니다.

현재 config.provisional.json은 COM8, 1Mbps를 지정합니다. 다른 PC 또는 조립 상태에서는 포트·모터 ID·초기 틱이 맞는지 확인합니다. primitive_calibration.local.json의 verified=false 때문에 CLI 실기 시험에는 --allow-provisional이 필요합니다. 손목 확인 결과를 버리라는 의미가 아니라, 파지 판정 등 전체 설정의 최종 검증 표시가 아직 분리되어 있지 않기 때문입니다.

모터 출력 없는 테스트는 저장소 루트에서 실행합니다.

    $env:PYTHONPATH = (Resolve-Path .\sim2real).Path
    python -B -m unittest discover -s .\sim2real\test -p "test_*.py" -q

## 4. middleware-bridge에 연결할 작업

### 2026-11-18 별도 ROS 서버 dry-run 결과

Windows Pixi 환경과 feature/hardware의 빌드된 system_interfaces로 좌수 서버를 실행했다. ROS_DOMAIN_ID=89, ROS_LOCALHOST_ONLY=1에서 액션 서버 1개와 타입 system_interfaces/action/ExecuteHandPrimitive를 확인했다.

- 서버 hardware_enabled=false, 요청 dry_run=true로 시험했다.
- 회전량 1, -1, 2마다 새 execution_id를 만들고 같은 ID의 grasp → rotate → release를 한 요청씩 완료까지 기다렸다. 총 9개 모두 SUCCEEDED, success=true, 빈 error_code와 ACCEPTED → COMPLETED 피드백을 확인했다.
- 회전량 3은 ABORTED, success=false, INVALID_PRIMITIVE와 FAILED 피드백을 확인했다.
- 별도 시작 검사에서는 Worker와 Bus를 mock으로 대체했다. ROS 기본 옵션에서 임시 보정 파일 거부, allow_provisional=true에서 기존 모델 로딩·추론과 30ms 주기 전달을 확인했다. 실제 COM 포트 생성은 0회였다.
- 요청·응답 원문은 이 PC의 C:\Users\sanghun\Desktop\robothand\tmp\pixi-left-ros\action-results.log에 있다. 무출력 시험 종료 후 서버 프로세스를 종료했다.

이 시험은 ROS 전송·요청 검증·완료 응답을 확인한다. 실기 Worker의 파지 유지·취소·정지와 실제 큐브 동작은 사용자 ROS 실기 시험에서 확인할 항목이다. 미들웨어 GUI의 전체 실행 경로는 추가 통합 대상이다.

미들웨어의 `custom_hand_control` 서버는 dry-run 전용입니다. 좌수의 `sim2real/primitive_server.py`가 같은 액션을 제공하므로 시험 중에는 좌수 서버 하나만 실행합니다. CLI, 모터 진단기와 GUI dry-run 런처도 종료합니다.

### Windows Pixi 준비

PowerShell에서 미들웨어 저장소 루트로 이동합니다. Pixi를 설치해 PATH에 등록하고 두 저장소를 같은 상위 폴더에 둡니다. 아래 경로는 실제 체크아웃 위치로 바꿉니다. 최초 빌드에는 Visual Studio 2022 C++ x64 빌드 도구와 Windows SDK가 필요합니다. 첫 시험에는 액션 타입을 제공하는 패키지만 빌드하면 됩니다.

```powershell
cd C:\path\to\middleware-bridge
pixi install --locked
$prefix = (Resolve-Path .pixi/envs/default).Path.Replace('\','/')
$env:PYTHONUTF8 = '1'
pixi run --locked colcon --log-base .colcon-win/log build --merge-install --base-paths src --packages-select system_interfaces --build-base .colcon-win/build --install-base .colcon-win/install --cmake-clean-cache --cmake-args "-DPYTHON_EXECUTABLE=$prefix/python.exe" "-DPYTHON_INCLUDE_DIR=$prefix/include" "-DPYTHON_LIBRARY=$prefix/libs/python314.lib"
```

이 PC에서는 CMake 자동 검색이 MSYS2의 Python과 시스템 3.12 라이브러리를 선택했습니다. 위 명령은 잠금 환경의 Windows Python 3.14를 지정합니다. 빌드 후 아래 `check_for_type_support`가 성공해야 실제 액션 서버를 띄울 수 있습니다.

**서버와 요청용 PowerShell 두 창에서 각각** 아래 명령을 실행합니다. `pixi shell`로 진입한 뒤 나머지 줄을 실행합니다. 같은 ROS 도메인을 사용합니다.

```powershell
cd C:\path\to\middleware-bridge
pixi shell
$env:ROS_DOMAIN_ID = '89'
$env:ROS_LOCALHOST_ONLY = '1'
$env:PYTHONUTF8 = '1'
. .\.colcon-win\install\local_setup.ps1
$left = (Resolve-Path ..\left-rl-with-mujoco).Path
python -c "import sys, rclpy; from system_interfaces.action import ExecuteHandPrimitive as A; from rclpy.type_support import check_for_type_support; check_for_type_support(A); print(sys.executable, A.Goal())"
```

### 모터 출력 없는 ROS 요청

서버 창에서 다음을 실행합니다. 기본값 `hardware_enabled=false`에서는 Hardware와 Worker를 생성하지 않습니다.

```powershell
python "$left\sim2real\primitive_server.py" --ros-args -p hardware_enabled:=false
```

요청 창에서 서버 개수 1, 노드 이름 `left_hand_primitive_server`, 액션 타입을 확인합니다. 다음 함수는 목표 한 개를 보내고 최종 결과까지 기다립니다.

```powershell
ros2 action info /custom_hand/execute_primitive
ros2 action list -t
. "$left\sim2real\ros_test_commands.ps1"
$quarter = 1
Send-LeftGoal grasp 0 1
Send-LeftGoal rotate $quarter 2
Send-LeftGoal release 0 3
```

**각 호출은 한 줄씩 실행합니다.** 다음 요청으로 넘어가는 기준은 `SUCCEEDED`, `success: true`, 빈 `error_code`입니다. Dry-run은 요청 필드와 액션 피드백(ACCEPTED → COMPLETED)을 검사합니다. 순서·소유권, 파지 유지, 정지는 실기 Worker가 검사합니다. `$quarter = -1`, `$quarter = 2`도 각각 새 execution ID로 파지부터 해제까지 반복할 수 있습니다.

### 사용자 실기 시험

Dry-run 서버를 Ctrl+C로 종료합니다. 새 환경에서 `serial` 또는 `stable_baselines3` import가 실패하면 **Pixi shell 밖**의 미들웨어 루트에서 `python -m pip --python .\.pixi\envs\default\python.exe install --only-binary=:all: pyserial==3.5 stable-baselines3==2.7.0`을 실행합니다. 외부 Python의 pip로 설치 대상을 Pixi에 지정하는 방식입니다. 이후 서버 창의 Pixi shell에서 다음을 실행합니다.

```powershell
python -c "import rclpy, serial, torch, stable_baselines3; from system_interfaces.action import ExecuteHandPrimitive; print(serial.VERSION, torch.__version__, stable_baselines3.__version__)"
Get-FileHash "$left\model\finger_robot_ppo_best.zip","$left\sim2real\config.provisional.json","$left\sim2real\primitive_calibration.local.json" | Format-List Hash,Path
python "$left\sim2real\primitive_server.py" --ros-args -p hardware_enabled:=true -p allow_provisional:=true -p "wrist_calibration:=$left\sim2real\primitive_calibration.local.json"
```

이 PC에서 확인한 파일 SHA-256은 다음과 같습니다. JSON 값은 Windows 작업 트리의 CRLF 바이트 기준입니다.

| 파일 | SHA-256 |
| --- | --- |
| 좌수 실기 ZIP | `BE90522E32676F1E815267E8CE962EEB9E404EBEF55D97D3FCA37505A84478F9` |
| config.provisional.json | `0425E8FFE7786E79ED646D5E164C9E5E5DEDA28EA6C2CB542AC0CD9623FA78AB` |
| primitive_calibration.local.json | `879BD9FE2F6B5B819720F56555B4E279A45337C56745F58F9DA9DE8F772C99B7` |
| 미들웨어 기존 ZIP | `5B9A3F36B348FA7FCC16370432E9F5CE476091CBEE38431686166FFABABD426B` |

서버는 기존 Hardware의 모델 해시·입출력 검사와 보정 검사를 사용합니다. 현재 장비의 `verified=false` 파일에는 `provisional_confirmed_for_test=true`가 들어 있습니다. 이 조합은 서버 시작 시 `allow_provisional:=true`를 지정한 경우에만 허용됩니다. 기본값은 false입니다. 시작 시 모델을 로드하고 첫 실기 grasp에서 Worker가 COM8을 엽니다. 실기 모드에서는 grasp 전이라도 종료 시 기존 정지 루틴이 포트를 열어 토크 OFF를 요청하고 확인할 수 있습니다.

요청 창에서 서버를 다시 확인하고 `$dry = 'false'`, `$quarter = 1`, 새 `$execution = 'left-test-' + [guid]::NewGuid().ToString('N')`를 설정합니다. 위의 **grasp, rotate, release 명령을 한 줄씩** 같은 execution ID로 보냅니다. 다른 회전량도 해제 후 새 파지에서 시작하면 매번 손목 기준 2400틱에서 시험할 수 있습니다.

| 단계 | 다음 단계로 넘어가기 전 확인할 내용 |
| --- | --- |
| grasp | EXECUTING 뒤 성공 응답. 손가락 엔코더가 설정 범위에서 안정되고 응답 후에도 파지 유지. 큐브 접촉과 미끄러짐은 직접 관찰합니다. |
| rotate | 1은 손목 50°+45°, -1은 50°+50°, 2는 47.5° → 47.5° → 45° → 45°이며 임시 시험 범위에서 295틱까지 명령합니다. 구간별 100/50틱(1/-1) 또는 첫 세 구간 75틱씩(2)·마지막 50틱 재파지와 손목 목표 도달 결과를 확인하고 실제 큐브 면의 방향·회전량을 별도로 기록합니다. |
| release | 손가락 초기 위치 도달과 손가락 토크 OFF를 확인합니다. 같은 연결에서 손목을 2400틱으로 복귀시킨 뒤 손목 토크 OFF를 확인하고 성공 응답합니다. |

실패·취소 시 다음 동작을 보내지 않고 `error_code`, `error_message`와 서버 로그를 확인합니다. 통신·시간 초과 또는 취소로 Worker가 FAULT이면 장치 상태를 확인한 뒤 서버를 재시작합니다. `STOP_UNCONFIRMED`는 정지 확인 실패입니다. ROS에 home 요청은 없습니다. CLI home은 ROS 서버 종료와 포트 해제 후에만 사용합니다.

후속 작업에서 `custom_hand_control`에 Worker·Hardware와 공통 제어 코드를 이식합니다. 그때 패키지 import, 설치 위치의 모델·JSON 경로, setup.py 데이터 파일과 package.xml 의존성, 기존 shadow 모델 호환성을 함께 확인합니다.

카메라 기반 큐브 위치 추정과 다른 손의 제어는 이 단독 좌수 코드에 포함되어 있지 않습니다. 실기 확인 범위는 좌수 단독 CLI와 ROS 액션입니다. 미들웨어 GUI와 실행 관리자의 확인 범위는 무출력 시험입니다.

### 2026-11-18 GUI 연결 dry-run 결과

기존 monitoring_ui를 Chrome에서 열고 rosbridge → 기존 CubeSequenceExecutor → 오른손 dry-run 서버 및 좌수 primitive_server.py 경로를 확인했습니다. 기본 custom_hand 서버 대신 좌수 서버 하나만 등록했고 hardware_enabled=false로 Hardware/Worker와 모터 출력을 비활성화했습니다. 카메라 대신 합성 solution `U U' R R' F2 F2`를 발행했으며, 취소 버튼을 누를 시간을 확보하려고 임시 시험 래퍼에서 좌수 요청마다 1초 대기를 추가했습니다. 제품 소스에는 이 대기를 추가하지 않았습니다.

- 정상 완료: execution_id `b13134f3-0718-48eb-81eb-62609b701ff0`, 전체 24단계 완료(state=5). 좌수 18개 요청이 모두 dry_run=true로 성공했습니다.
- 최초 취소 시험에서는 rosbridge의 동기 액션 처리가 WebSocket 수신을 막아 취소가 지연됐습니다. Windows run_rosbridge.ps1에 `--ros-args -p send_action_goals_in_new_thread:=true`를 적용합니다.
- 해당 옵션으로 재시작한 뒤 사용자 GUI 취소: execution_id `035b2d48-61eb-45e8-b672-9543fb30e146`, 7단계 rotate_upper_layer가 success=false / CANCELLED로 끝났고 전체 상태도 CANCELLED(state=7)였습니다. 후속 단계 요청은 없었습니다.

로컬 증거는 작업 루트의 `tmp/pixi-left-ros/gui/`에 있습니다. `statuses-before-thread-fix.jsonl` 및 `left-before-thread-fix.log`는 정상 완료와 최초 취소 지연을, `statuses.jsonl` 및 `left.log`는 수정 후 취소를 기록합니다. 이 시험은 브라우저 GUI와 ROS 메시지 연결 확인이며, Tauri 앱 실행·카메라 인식·실기 PPO 유지·모터 정지를 검증한 결과는 아닙니다. 실제 파지 완료 후 대기 중 취소와 통신 단절의 정지 처리는 실기 통합 전에 별도로 확인해야 합니다.

### 2026-11-19 최신 미들웨어 실행기 연결

미들웨어 작업 폴더를 최신 main d03c5b6 기반의 로컬 codex/left-gui-dryrun 브랜치로 갱신했다. 이제 미들웨어 루트의 `start.bat -LeftDryRun`이 기존 GUI/실행 관리자와 이 저장소의 좌수 ROS 서버를 함께 실행한다. `stop.bat`으로 종료한다. 카메라 없는 합성 풀이, 브라우저 GUI, hardware_enabled=false 구성이다. 최신 인터페이스 빌드, 24단계 완료, 취소, 실기 요청 거부, 중복 실행 차단과 종료를 확인했다. 자세한 명령과 증거는 미들웨어 docs/left_gui_dryrun.md를 참고한다. 실제 Worker/Hardware 패키지 이식은 사용자 ROS 실기 검증 후 진행한다.

### 실기 오류와 자동 로그

서버를 실행하면 `logs/primitive/YYYY-MM-DD/시각_PID`에 `console.log`, `events.jsonl`,
`session.json` 및 해당 실행의 설정 스냅샷이 저장됩니다. `latest.json`은 최근 실행 폴더를 가리킵니다.
`events.jsonl`에서 실행 ID별 요청·결과와 Worker 오류를 확인할 수 있습니다.

명목 제어 주기는 30ms이고, 제어 주기 간격이나 한 주기의 I/O가 100ms를 넘으면 오류로 처리합니다.
오류 기록의 마지막 정상 위치·전류·전압에는 측정 시점이 함께 남습니다.
일부 오류에서 손목 정지와 손가락의 통신·상태를 다시 확인한 경우에만 손가락 초기 복귀를 시도합니다.
`STOP_UNCONFIRMED`가 기록되면 정지 확인에 실패한 것이므로 추가 동작을 보내지 말고 장비 상태를 확인합니다.
