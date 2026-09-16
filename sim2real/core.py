"""Pure transformations and safety checks. No serial or policy imports."""
import math

ORDER = ('right', 'front', 'left', 'back')
CTRL = ((-0.0873, 2.0071), (0., 2.0944), (-0.0441, 2.0503), (0., 2.0944))


class Fault(RuntimeError):
    pass


def finite(x):
    if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
        raise Fault(f'Expected finite number, got {x!r}')
    return x


def positive(x):
    if finite(x) <= 0:
        raise Fault('Expected positive number')
    return x


def validate(cfg, policy=False, live=False, allow_provisional=False):
    if live and allow_provisional:
        raise Fault("Provisional calibration is only allowed without motor commands")
    motors = cfg['motors']
    if len(motors) != 4 or {m['id'] for m in motors} != {1, 2, 3, 4}:
        raise Fault('Exactly IDs 1-4 required')
    if tuple(m['finger'] for m in motors) != ORDER:
        raise Fault('Configuration must be in right/front/left/back order')
    positive(cfg['period_s']); positive(cfg['max_age_s'])
    if policy:
        for m in motors:
            c = m['calibration']
            if (not c['verified'] or not c['evidence']) and not (allow_provisional and c.get('status') == 'user_approved_provisional'):
                raise Fault(f"ID {m['id']}: motor/tendon calibration needs verification")
            finite(c['zero_tick']); finite(c['tendon_at_zero'])
            if finite(c['tendon_per_tick']) >= 0:
                raise Fault('Expected negative slope for winding direction')
        if not cfg['mapping_verified'] and not (allow_provisional and cfg.get('mapping_status') == 'user_approved_provisional'):
            raise Fault('Simulation finger mapping is not verified')
    if live:
        validate(cfg, policy=True)
        if not cfg['timing_verified'] or not cfg['stop_verified'] or not cfg['policy']['hardware_validated']:
            raise Fault('Timing, torque-off stop and actual policy require verification')
        if not cfg['policy']['sha256']:
            raise Fault('Policy hash required')
        if type(cfg['watchdog_ticks']) is not int or not 1 <= cfg['watchdog_ticks'] <= 127:
            raise Fault('Watchdog must be 1..127 (20ms units)')
        positive(cfg['smoothing_tau_s'])
        for m in motors:
            if m['model_number'] not in (1190, 1200):
                raise Fault('Only confirmed XL330-M077/M288 supported')
            l = m['limits']
            if not l['verified'] or not l['evidence']:
                raise Fault(f"ID {m['id']}: verified limits required")
            lo, hi = l['position_tick']
            if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= 4095:
                raise Fault('Position limits must be integer single-turn ticks')
            for key in ('velocity_tick_s', 'current_ma', 'temperature_c', 'step_tick', 'tracking_error_tick'):
                positive(l[key])
            if type(l['current_ma']) is not int or l['current_ma'] > 1750:
                raise Fault('Invalid current limit')
            if type(l['profile_velocity_raw']) is not int or not 1 <= l['profile_velocity_raw'] <= 445:
                raise Fault('Nonzero velocity profile required')


def states_by_id(rows, now, max_age):
    if len(rows) != 4 or {r['id'] for r in rows} != {1, 2, 3, 4}:
        raise Fault('Missing/duplicate motor states')
    for r in rows:
        for k in ('position_tick', 'velocity_tick_s', 'current_ma', 'temperature_c', 'sample_start_s'):
            finite(r[k])
        age = finite(now) - r['sample_start_s']
        if age < 0 or age > max_age:
            raise Fault(f"ID {r['id']}: stale/future state ({age}s)")
    return {r['id']: r for r in rows}


def observation(cfg, rows, now, allow_provisional=False):
    validate(cfg, policy=True, allow_provisional=allow_provisional)
    by_id = states_by_id(rows, now, cfg['max_age_s'])
    positions, velocities = [], []
    for m in cfg['motors']:
        r, c = by_id[m['id']], m['calibration']
        positions.append(c['tendon_at_zero'] + (r['position_tick'] - c['zero_tick']) * c['tendon_per_tick'])
        velocities.append(r['velocity_tick_s'] * c['tendon_per_tick'])
    return [finite(x) for x in positions + velocities]


def action_targets(cfg, action):
    if len(action) != 4:
        raise Fault('Action must contain 4 values')
    targets = {}
    for m, a, (lo, hi) in zip(cfg['motors'], action, CTRL):
        a = finite(a)
        if abs(a) > 1.000001:
            raise Fault('Policy output outside normalized action range')
        a = max(-1., min(1., a))
        target = lo + (a + 1.) * .5 * (hi - lo)
        c = m['calibration']
        targets[m['id']] = finite(c['zero_tick'] + (target - c['tendon_at_zero']) / c['tendon_per_tick'])
    return targets


class Safety:
    def __init__(self, cfg, rows, now):
        validate(cfg, live=True)
        self.cfg, self.faulted = cfg, False
        self.previous = {r['id']: r['position_tick'] for r in rows}
        self.check(rows, now)

    def check(self, rows, now):
        try:
            if self.faulted:
                raise Fault('Latched fault: restart and revalidate required')
            by_id = states_by_id(rows, now, self.cfg['max_age_s'])
            for m in self.cfg['motors']:
                r, l = by_id[m['id']], m['limits']
                lo, hi = l['position_tick']
                if not lo <= r['position_tick'] <= hi:
                    raise Fault(f"ID {m['id']}: position limit")
                for k, lim in [('velocity_tick_s', l['velocity_tick_s']), ('current_ma', l['current_ma']), ('temperature_c', l['temperature_c'])]:
                    if abs(r[k]) > lim:
                        raise Fault(f"ID {m['id']}: {k} limit")
                if abs(r['position_tick'] - self.previous[m['id']]) > l['tracking_error_tick']:
                    raise Fault(f"ID {m['id']}: tracking error")
        except Exception:
            self.faulted = True
            raise

    def command(self, targets, rows, now, dt):
        try:
            self.check(rows, now)
            positive(dt)
            if dt > self.cfg['period_s'] * 1.5:
                raise Fault('Command deadline exceeded')
            if set(targets) != {1, 2, 3, 4}:
                raise Fault('Incomplete targets')
            result = {}
            alpha = 1 - math.exp(-dt / self.cfg['smoothing_tau_s'])
            for m in self.cfg['motors']:
                i, l = m['id'], m['limits']
                prev = self.previous[i]
                target = prev + alpha * (finite(targets[i]) - prev)
                step = min(l['step_tick'], l['velocity_tick_s'] * dt)
                lo, hi = l['position_tick']
                # Integer feasible interval prevents rounding from exceeding slew limits.
                low = math.ceil(max(lo, prev - step))
                high = math.floor(min(hi, prev + step))
                result[i] = max(low, min(high, round(target)))
            self.previous = result
            return result
        except Exception:
            self.faulted = True
            raise
