"""Pure risk calculations; no API requests or trade execution."""
import math


def calculate_trade(account, risk_pct, entry, stop, target, direction='Long', buying_power=None):
    vals = (account, risk_pct, entry, stop, target)
    if not all(math.isfinite(float(x)) for x in vals):
        raise ValueError('Inputs must be finite numbers.')
    account, risk_pct, entry, stop, target = map(float, vals)
    if account <= 0 or not (0 < risk_pct <= 100) or min(entry, stop, target) <= 0:
        raise ValueError('Account, prices and risk must be positive; risk cannot exceed 100%.')
    if direction not in ('Long', 'Short'):
        raise ValueError('Direction must be Long or Short.')
    if direction == 'Long' and not (stop < entry < target):
        raise ValueError('For long positions: stop < entry < target.')
    if direction == 'Short' and not (target < entry < stop):
        raise ValueError('For short positions: target < entry < stop.')
    if buying_power is None:
        buying_power = account  # Conservative, no leverage assumption
    if not math.isfinite(float(buying_power)) or buying_power < 0:
        raise ValueError('Buying power must be a nonnegative finite number.')
    budget = account * risk_pct / 100
    risk_per_share = abs(entry - stop)
    shares = min(math.floor(budget / risk_per_share), math.floor(float(buying_power) / entry))
    return {
        'shares': shares, 'risk_budget': budget,
        'planned_loss': shares * risk_per_share,
        'potential_profit': shares * abs(target - entry),
        'position_value': shares * entry,
        'stop_distance_pct': risk_per_share / entry * 100,
        'reward_risk': abs(target - entry) / risk_per_share,
    }
