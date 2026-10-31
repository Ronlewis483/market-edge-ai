"""Review exact offered props. No network calls, model fitting, or assumed platform payouts."""
import math
PICKEM = {'underdog', 'prizepicks'}


def single_value(row):
    """Conditional on a decided (non-push) result, like the supplied model chance."""
    try:
        p, price = float(row['chance']), float(row['decimal_odds'])
    except (TypeError, ValueError, KeyError):
        return None
    if not math.isfinite(p) or not math.isfinite(price) or not 0 < p < 1 or price <= 1:
        return None
    return {'break_even': 1/price, 'edge': p-1/price,
            'return_per_unit': p*price-1}


def rank_offers(rows, book):
    result = []
    for row in rows:
        if row.get('book') != book:
            continue
        value = None if book in PICKEM else single_value(row)
        if book not in PICKEM and (value is None or value['edge'] <= 0):
            continue
        result.append(dict(row, value_review=value))
    return sorted(result, key=lambda r: (
        -(r['value_review']['return_per_unit'] if r['value_review'] else r['chance']),
        -r['chance'], str(r['start']), str(r['player'])))


def review_ticket(legs, book, requested, total_return=None):
    """Scenario calculations, never a verified joint-probability or calibration claim."""
    singles = sorted([dict(r, **single_value(r)) for r in legs
                      if book not in PICKEM and single_value(r) is not None],
                     key=lambda r: -r['return_per_unit'])
    report = {'complete': len(legs) == requested, 'singles': singles,
              'status': 'incomplete', 'combined_probability': None,
              'return_per_unit': None, 'total_return': None}
    if not report['complete']:
        return report
    games = [(r['sport'], r['game'], r['start']) for r in legs]
    players = [(r['sport'], str(r['player']).strip().casefold()) for r in legs]
    if len(set(games)) != len(games) or len(set(players)) != len(players):
        report['status'] = 'correlation_not_assessed'
        return report
    if any(r.get('game_phase') == 'live' for r in legs):
        report['status'] = 'live_joint_not_assessed'
        return report
    if book in PICKEM:
        try:
            payout = float(total_return)
        except (ValueError, TypeError):
            payout = 0
        if not math.isfinite(payout) or payout <= 1:
            report['status'] = 'actual_payout_required'
            return report
    else:
        if any(single_value(r) is None for r in legs):
            report['status'] = 'price_missing'
            return report
        payout = math.prod(float(r['decimal_odds']) for r in legs)
    p = math.prod(float(r['chance']) for r in legs)
    report.update(combined_probability=p, total_return=payout,
                  break_even=1/payout, return_per_unit=p*payout-1,
                  status='positive_estimated_edge' if p*payout > 1 else 'no_estimated_edge',
                  assumptions='Independent legs, all win or full loss, no pushes/voids, unchanged quoted payout. Model chances are not established sportsbook accuracy.')
    # Compare removing each leg with the offered full ticket, under the same scenario.
    report['leg_effects'] = []
    if book not in PICKEM:
        for r in legs:
            factor = float(r['chance'])*float(r['decimal_odds'])
            report['leg_effects'].append({'player': r['player'],
                'without_leg_return_per_unit': (p*payout/factor)-1,
                'with_leg_return_per_unit': p*payout-1})
    return report
