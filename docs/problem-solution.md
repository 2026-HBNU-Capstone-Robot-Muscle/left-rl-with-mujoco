# 발생한 문제 및 해결법 정리

환경 설정 및 코딩 과정에서 발생한 문제와 그에 대한 해결 방법을 작성한다.

# 목차

1. [MuJoCo 설치 문제](#mujoco-설치-문제)
2. [MuJoCo와 Gymnasium 버전 호환성 문제](#mujoco와-gymnasium-버전-호환성-문제)

---

## MuJoCo 설치 문제

`pip install mujoco`를 통해 설치하는 과정에서 아래의 에러 메시지가 뜬다면 가상환경의 파이썬 버전의 문제로 발생하는 것이다.

```powershell
(.env) (base) PS C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco> pip install mujoco==3.1.6
Collecting mujoco==3.1.6
  Downloading mujoco-3.1.6.tar.gz (670 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 670.6/670.6 kB 7.5 MB/s  0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Requirement already satisfied: absl-py in .\.env\Lib\site-packages (from mujoco==3.1.6) (2.5.0)
Requirement already satisfied: etils[epath] in .\.env\Lib\site-packages (from mujoco==3.1.6) (1.14.0)
Requirement already satisfied: glfw in .\.env\Lib\site-packages (from mujoco==3.1.6) (2.10.2)
Requirement already satisfied: numpy in .\.env\Lib\site-packages (from mujoco==3.1.6) (2.5.2)
Requirement already satisfied: pyopengl in .\.env\Lib\site-packages (from mujoco==3.1.6) (3.1.10)
Requirement already satisfied: fsspec in .\.env\Lib\site-packages (from etils[epath]->mujoco==3.1.6) (2026.7.0)
Requirement already satisfied: typing_extensions in .\.env\Lib\site-packages (from etils[epath]->mujoco==3.1.6) (4.16.0)
Requirement already satisfied: zipp in .\.env\Lib\site-packages (from etils[epath]->mujoco==3.1.6) (4.1.0)
Building wheels for collected packages: mujoco
  Building wheel for mujoco (pyproject.toml) ... error
  error: subprocess-exited-with-error
  
  × Building wheel for mujoco (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [154 lines of output]
      C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\config\_apply_pyprojecttoml.py:82: SetuptoolsDeprecationWarning: `project.license` as a TOML table is deprecated
      !!
      
              ********************************************************************************
              Please use a simple string containing a SPDX expression for `project.license`. You can also use `project.license-files`. (Both options available on setuptools>=77.0.0).
      
              By 2027-Feb-18, you need to update your project and remove deprecated calls
              or your builds will no longer be supported.
      
              See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.
              ********************************************************************************
      
      !!
        corresp(dist, value, root_dir)
      C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\config\_apply_pyprojecttoml.py:61: SetuptoolsDeprecationWarning: License classifiers are deprecated.
      !!
      
              ********************************************************************************
              Please consider removing the following classifiers in favor of a SPDX license expression:
      
              License :: OSI Approved :: Apache Software License
      
              See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.
              ********************************************************************************
      
      !!
        dist._finalize_license_expression()
      C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\dist.py:765: SetuptoolsDeprecationWarning: License classifiers are deprecated.
      !!
      
              ********************************************************************************
              Please consider removing the following classifiers in favor of a SPDX license expression:
      
              License :: OSI Approved :: Apache Software License
      
              See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.
              ********************************************************************************
      
      !!
        self._finalize_license_expression()
      running bdist_wheel
      running build
      running build_py
      creating build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\bindings_test.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\gl_context.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\minimize.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\minimize_test.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\msh2obj.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\msh2obj_test.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\renderer.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\renderer_test.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\render_test.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\rollout.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\rollout_test.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\viewer.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\viewer_test.py -> build\lib.win-amd64-cpython-314\mujoco
      copying mujoco\__init__.py -> build\lib.win-amd64-cpython-314\mujoco
      creating build\lib.win-amd64-cpython-314\mujoco\cgl
      copying mujoco\cgl\cgl.py -> build\lib.win-amd64-cpython-314\mujoco\cgl
      copying mujoco\cgl\__init__.py -> build\lib.win-amd64-cpython-314\mujoco\cgl
      creating build\lib.win-amd64-cpython-314\mujoco\egl
      copying mujoco\egl\egl_ext.py -> build\lib.win-amd64-cpython-314\mujoco\egl
      copying mujoco\egl\__init__.py -> build\lib.win-amd64-cpython-314\mujoco\egl
      creating build\lib.win-amd64-cpython-314\mujoco\glfw
      copying mujoco\glfw\__init__.py -> build\lib.win-amd64-cpython-314\mujoco\glfw
      creating build\lib.win-amd64-cpython-314\mujoco\osmesa
      copying mujoco\osmesa\__init__.py -> build\lib.win-amd64-cpython-314\mujoco\osmesa
      creating build\lib.win-amd64-cpython-314\mujoco\usd
      copying mujoco\usd\component.py -> build\lib.win-amd64-cpython-314\mujoco\usd
      copying mujoco\usd\exporter.py -> build\lib.win-amd64-cpython-314\mujoco\usd
      copying mujoco\usd\exporter_test.py -> build\lib.win-amd64-cpython-314\mujoco\usd
      copying mujoco\usd\utils.py -> build\lib.win-amd64-cpython-314\mujoco\usd
      creating build\lib.win-amd64-cpython-314\mujoco\testdata
      copying mujoco\testdata\msh.xml -> build\lib.win-amd64-cpython-314\mujoco\testdata
      copying mujoco\testdata\abdomen_1_body.msh -> build\lib.win-amd64-cpython-314\mujoco\testdata
      running build_ext
      Traceback (most recent call last):
        File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 389, in <module>
          main()
          ~~~~^^
        File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 373, in main
          json_out["return_val"] = hook(**hook_input["kwargs"])
                                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^
        File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 280, in build_wheel
          return _build_backend().build_wheel(
                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
              wheel_directory, config_settings, metadata_directory
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          )
          ^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\build_meta.py", line 441, in build_wheel
          return _build(['bdist_wheel', '--dist-info-dir', str(metadata_directory)])
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\build_meta.py", line 429, in _build
          return self._build_with_temp_dir(
                 ~~~~~~~~~~~~~~~~~~~~~~~~~^
              cmd,
              ^^^^
          ...<3 lines>...
              self._arbitrary_args(config_settings),
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          )
          ^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\build_meta.py", line 410, in _build_with_temp_dir
          self.run_setup()
          ~~~~~~~~~~~~~~^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\build_meta.py", line 317, in run_setup
          exec(code, locals())  # noqa: S102 # exec is intentional here
          ~~~~^^^^^^^^^^^^^^^^
        File "<string>", line 334, in <module>
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\__init__.py", line 117, in setup
          return distutils.core.setup(**attrs)  # type: ignore[return-value]
                 ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\_distutils\core.py", line 168, in setup
          return run_commands(dist)
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\_distutils\core.py", line 184, in run_commands
          dist.run_commands()
          ~~~~~~~~~~~~~~~~~^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\_distutils\dist.py", line 1028, in run_commands
          self.run_command(cmd)
          ~~~~~~~~~~~~~~~~^^^^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\dist.py", line 1106, in run_command
          super().run_command(command)
          ~~~~~~~~~~~~~~~~~~~^^^^^^^^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\_distutils\dist.py", line 1047, in run_command
          cmd_obj.run()
          ~~~~~~~~~~~^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\command\bdist_wheel.py", line 385, in run
          self.run_command("build")
          ~~~~~~~~~~~~~~~~^^^^^^^^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\_distutils\cmd.py", line 342, in run_command
          self.distribution.run_command(command)
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\dist.py", line 1106, in run_command
          super().run_command(command)
          ~~~~~~~~~~~~~~~~~~~^^^^^^^^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\_distutils\dist.py", line 1047, in run_command
          cmd_obj.run()
          ~~~~~~~~~~~^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\_distutils\command\build.py", line 137, in run
          self.run_command(cmd_name)
          ~~~~~~~~~~~~~~~~^^^^^^^^^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\_distutils\cmd.py", line 342, in run_command
          self.distribution.run_command(command)
          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\dist.py", line 1106, in run_command
          super().run_command(command)
          ~~~~~~~~~~~~~~~~~~~^^^^^^^^^
        File "C:\Users\jeonj\AppData\Local\Temp\pip-build-env-jy9nu5yo\overlay\Lib\site-packages\setuptools\_distutils\dist.py", line 1047, in run_command
          cmd_obj.run()
          ~~~~~~~~~~~^^
        File "<string>", line 152, in run
        File "<string>", line 166, in _find_mujoco
      RuntimeError: MUJOCO_PATH environment variable is not set
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for mujoco
Failed to build mujoco
error: failed-wheel-build-for-install

× Failed to build installable wheels for some pyproject.toml based projects
╰─> mujoco
```

#### 발생 상황 (2026.09.06)

Gymnasium과 호환성 문제가 발생하며 MuJoCo를 다운그레이드 시키는 과정에서 MuJoCo를 3.1.6 버전까지 내리면서 발생

#### 문제 발생 원인

MuJoCo 3.1.6 버전을 설치하는 과정에서 Python 3.14용 사전 빌드 wheel이 없어서 pip가 소스코드를 직접 빌드하려고 시도하다가 `MUJOCO_PATH` 환경변수가 없어서 에러가 발생한다.

#### 해결 방법

Python 3.14는 너무 최신 버전이어서 MuJoCo, Gymnasium을 비롯한 여러 라이브러리들이 사전 빌드 툴인 wheel을 제공하지 않는 경우가 많다. 이에 가상환경을 Python 3.11 혹은 3.12 로 버전을 낮춘 가상 환경을 다시 만들어 설치를 진행한다.
```powershell
# 설치된 파이썬 버전 확인
py -0

# pyenv 가상환경 제거
Remove-Item -Recurse -Force <가상환경명>

# 가상환경 재구축
py -3.12 -m venv <가상환경명>

# 설치된 파이썬 버전 최종 확인
python --version
```

## MuJoCo와 Gymnasium 버전 호환성 문제

MuJoCo 테스트 실행 시 화면은 정상적으로 뜨나 휠스크롤과 같은 행동을 취했을 때, 아래의 에러 메시지가 뜨며 종료된다면, 이는 MuJoCo와 Gymnasium의 버전 호환성이 맞지 않기 때문이다.

```bash
Traceback (most recent call last):
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\test-env.py", line 11, in <module>
    obs, reward, terminated, truncated, info = env.step(action)
                                               ~~~~~~~~^^^^^^^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\gymnasium\wrappers\common.py", line 124, in step
    observation, reward, terminated, truncated, info = self.env.step(action)
                                                       ~~~~~~~~~~~~~^^^^^^^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\gymnasium\wrappers\common.py", line 392, in step
    return super().step(action)
           ~~~~~~~~~~~~^^^^^^^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\gymnasium\core.py", line 327, in step
    return self.env.step(action)
           ~~~~~~~~~~~~~^^^^^^^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\gymnasium\wrappers\common.py", line 284, in step
    return self.env.step(action)
           ~~~~~~~~~~~~~^^^^^^^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\gymnasium\envs\mujoco\half_cheetah_v5.py", line 235, in step
    self.render()
    ~~~~~~~~~~~^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\gymnasium\envs\mujoco\mujoco_env.py", line 157, in render
    return self.mujoco_renderer.render(self.render_mode)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\gymnasium\envs\mujoco\mujoco_rendering.py", line 770, in render
    return viewer.render()
           ~~~~~~~~~~~~~^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\gymnasium\envs\mujoco\mujoco_rendering.py", line 492, in render
    update()
    ~~~~~~^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\gymnasium\envs\mujoco\mujoco_rendering.py", line 474, in update
    glfw.poll_events()
    ~~~~~~~~~~~~~~~~^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\glfw\__init__.py", line 1861, in poll_events
    _glfw.glfwPollEvents()
    ~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\glfw\__init__.py", line 692, in errcheck
    _reraise(exc[1], exc[2])
    ~~~~~~~~^^^^^^^^^^^^^^^^
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\glfw\__init__.py", line 70, in _reraise
    raise exception.with_traceback(traceback)
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\glfw\__init__.py", line 671, in callback_wrapper
    return func(*args, **kwargs)
  File "C:\Users\jeonj\Desktop\rl\left-rl-with-mujoco\.env\Lib\site-packages\gymnasium\envs\mujoco\mujoco_rendering.py", line 620, in _scroll_callback
    mujoco.mjv_moveCamera(
    ~~~~~~~~~~~~~~~~~~~~~^
        self.model,
        ^^^^^^^^^^^
    ...<4 lines>...
        self.cam,
        ^^^^^^^^^
    )
    ^
TypeError: mjv_moveCamera(): incompatible function arguments. The following argument types are supported:
    1. (m: mujoco._structs.MjModel, action: typing.SupportsInt | typing.SupportsIndex, reldx: typing.SupportsFloat | typing.SupportsIndex, reldy: typing.SupportsFloat | typing.SupportsIndex, cam: mujoco._structs.MjvCamera) -> None

Invoked with: <mujoco._structs.MjModel object at 0x00000245574E7070>, <mjtMouse.mjMOUSE_ZOOM: 5>, 0, -0.05, <mujoco._structs.MjvScene object at 0x00000245574A4430>, <MjvCamera
  azimuth: 90.0
  distance: 4.0
  elevation: -45.0
  fixedcamid: -1
  lookat: array([0., 0., 0.])
  orthographic: 0
  trackbodyid: -1
  type: 0
>
```

2026.09.06 기준 설치되는 버전

1. MuJoCo
- 설치 코드: `pip install mujoco`
- 설치 버전: 3.12.0

2. Gymnasium
- 설치 코드: `pip install gymnasium[mujoco]`
- 설치 버전: 1.3.0

#### 버전 비호환성 발생 원인

MuJoCo 3.11.0(2026.07.27)버전에서 `mjv_moveCamera()` 함수의 시그니처가 변경(`mjvScene`인자가 제거됨)되면서 Gymnasium의 렌더링 코드와 호환이 깨진다.

#### 해결방법

MuJoCo가 3.11.0 버전 이상이라 발생하는 문제이므로, 사용할 때 3.10.0 버전으로 고정하여 설치하면 해결된다.
```powershell
# MuJoCo 제거
pip uninstall mujoco -y

# MuJoCo 3.10.0 버전으로 고정하여 설치
pip install "mujoco==3.10.0"
```