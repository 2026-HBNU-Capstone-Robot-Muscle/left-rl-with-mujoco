"""Default: offline archive inspection. No motor access without explicit mode."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile

from core import Fault, Safety, observation, action_targets, validate, states_by_id

ROOT = Path(__file__).resolve().parent


def inspect_policy(path):
    with zipfile.ZipFile(path) as z:
        d = json.loads(z.read('data'))
        spaces = {k: {f: d[k].get(f) for f in ('_shape', 'dtype', 'low', 'high')}
                  for k in ('observation_space', 'action_space')}
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'spaces': spaces,
            'num_timesteps': d.get('num_timesteps')}


class Policy:
    def __init__(self, path, cfg):
        digest = inspect_policy(path)['sha256']
        if digest != cfg['policy']['sha256']:
            raise Fault('Policy SHA256 mismatch or unset; inspect first')
        # Only load user-provided trusted SB3 archives: SB3 uses pickle internally.
        from stable_baselines3 import PPO
        import numpy as np
        self.np = np
        self.model = PPO.load(str(path), device='cpu')
        if self.model.observation_space.shape != (8,) or self.model.action_space.shape != (4,):
            raise Fault('Expected observation(8), action(4)')
        if not (np.all(self.model.action_space.low == -1) and np.all(self.model.action_space.high == 1)):
            raise Fault('Action bounds differ from source contract')

    def predict(self, obs):
        array = self.np.asarray(obs, dtype=self.np.float32)
        if not self.np.isfinite(array).all():
            raise Fault('Nonfinite float32 observation')
        a, _ = self.model.predict(array, deterministic=True)
        return a.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['inspect', 'read', 'replay', 'shadow', 'live'], default='inspect')
    p.add_argument('--config', type=Path, default=ROOT / 'config.json')
    p.add_argument('--policy', type=Path)
    p.add_argument('--input', type=Path, help='JSONL log for replay; no hardware traffic')
    p.add_argument('--port')
    p.add_argument('--cycles', type=int, default=1)
    p.add_argument('--ids', type=int, nargs='+', help='Live output IDs: one ID first, or 1 2 3 4')
    p.add_argument('--allow-provisional', action='store_true', help='Allow explicit provisional mapping in shadow/replay only')
    args = p.parse_args()
    bus = None
    log = None
    exit_code = 0
    try:
        cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
        if args.cycles <= 0:
            raise Fault('cycles must be positive')
        if args.mode == 'inspect':
            validate(cfg)
            print(json.dumps(inspect_policy(args.policy), indent=2) if args.policy else
                  'Configuration IDs/order valid. No port opened. Calibration and live gates remain unverified.')
            return 0
        if args.allow_provisional and args.mode not in ('shadow', 'replay'):
            raise Fault('--allow-provisional is restricted to shadow/replay')
        is_policy = args.mode in ('replay', 'shadow', 'live')
        validate(cfg, policy=is_policy, live=args.mode == 'live', allow_provisional=args.allow_provisional)
        selected = args.ids or []
        if args.mode == 'live':
            if len(set(selected)) != len(selected) or not (len(selected) == 1 and selected[0] in (1, 2, 3, 4) or set(selected) == {1, 2, 3, 4}):
                raise Fault('Live requires explicit --ids: one motor or all four')
            if len(selected) == 4 and not cfg['single_motor_tests_verified']:
                raise Fault('Complete single motor tests before all-four execution')
        if is_policy and args.policy is None:
            raise Fault('--policy required')
        policy = Policy(args.policy, cfg) if is_policy else None
        ROOT.joinpath('logs').mkdir(exist_ok=True)
        log_path = ROOT / 'logs' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.jsonl')
        log = log_path.open('x', encoding='utf-8')

        def emit(item):
            log.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
            log.flush()

        emit({'type': 'metadata', 'utc': datetime.now(timezone.utc).isoformat(),
              'mode': args.mode, 'config': cfg, 'selected_ids': selected,
              'policy': inspect_policy(args.policy) if is_policy else None})
        if args.mode == 'replay':
            if args.input is None:
                raise Fault('--input required')
            count = 0
            with args.input.open(encoding='utf-8') as stream:
                for line in stream:
                    record = json.loads(line)
                    if record.get('type') != 'cycle':
                        continue
                    obs = observation(cfg, record['rows'], record['read_end_s'], allow_provisional=args.allow_provisional)
                    a = policy.predict(obs)
                    emit({'type': 'replay', 'obs': obs, 'action': a,
                          'raw_targets_tick': action_targets(cfg, a), 'commands_sent': False})
                    count += 1
                    if count >= args.cycles:
                        break
            if not count:
                raise Fault('No complete cycles in replay file')
        else:
            from bus import Bus
            bus = Bus(args.port or cfg['port'], cfg['baud'])
            inventory = bus.inventory()
            emit({'type': 'inventory', 'motors': inventory})
            print(json.dumps(inventory, indent=2))
            if any(i['model_number'] not in (1190, 1200) for i in inventory):
                raise Fault('Non-XL330 model: telemetry units not validated')
            if args.mode == 'live':
                rows = bus.read()
                safety = Safety(cfg, rows, time.monotonic())
                bus.arm(cfg, selected, rows)
            for cycle in range(args.cycles):
                start = time.monotonic()
                rows = bus.read_sync() if cfg.get('read_method', 'sequential') == 'sync' else bus.read()
                read_end = time.monotonic()
                if args.mode == 'read':
                    print(json.dumps({'motor_feedback': rows}, indent=2))
                states_by_id(rows, read_end, cfg['max_age_s'])
                record = {'type': 'cycle', 'cycle': cycle, 'rows': rows, 'read_end_s': read_end,
                          'read_ms': (read_end - start) * 1000, 'commands_sent': False}
                if is_policy:
                    obs = observation(cfg, rows, read_end, allow_provisional=args.allow_provisional)
                    a = policy.predict(obs)
                    targets = action_targets(cfg, a)
                    if args.mode == 'shadow':
                        print('SHADOW ONLY targets_tick:', {i: round(v, 2) for i, v in targets.items()})
                    record.update(obs=obs, action=a, raw_targets_tick=targets,
                                  inference_and_transform_ms=(time.monotonic() - read_end) * 1000)
                    # Recheck freshness AFTER inference, before any actuator command.
                    states_by_id(rows, time.monotonic(), cfg['max_age_s'])
                    if args.mode == 'live':
                        if time.monotonic() - start >= cfg['period_s']:
                            raise Fault('Read/inference exceeded command deadline')
                        commands = safety.command(targets, rows, time.monotonic(), cfg['period_s'])
                        # Nonselected motors remain torque-off and do not advance command history.
                        for r in rows:
                            if r['id'] not in selected:
                                safety.previous[r['id']] = r['position_tick']
                        emit({'type': 'command_attempt', 'cycle': cycle,
                              'targets': {i: commands[i] for i in selected}})
                        bus.command(commands, selected, rows, cfg['max_age_s'])
                        record.update(commands_sent=True, targets_tick={i: commands[i] for i in selected})
                elapsed = time.monotonic() - start
                record['cycle_ms'] = elapsed * 1000
                record['overrun'] = elapsed > cfg['period_s']
                emit(record)
                print(f'cycle={cycle} time={elapsed*1000:.1f}ms sent={record["commands_sent"]}')
                if args.mode == 'live' and record['overrun']:
                    raise Fault('Loop overrun: stop, no catch-up commands')
                if cycle + 1 < args.cycles:
                    time.sleep(max(0, cfg['period_s'] - elapsed))
        print(f'Log: {log_path}')
    except (Exception, KeyboardInterrupt) as exc:
        exit_code = 1
        print(f'STOP: {type(exc).__name__}: {exc}')
        if log:
            try:
                log.write(json.dumps({'type': 'fault', 'error': str(exc)}) + '\n'); log.flush()
            except OSError:
                pass
    finally:
        if bus:
            try:
                errors = bus.stop()
                for error in errors:
                    print(error)
                if errors:
                    exit_code = 1
            finally:
                bus.close()
        if log:
            log.close()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
