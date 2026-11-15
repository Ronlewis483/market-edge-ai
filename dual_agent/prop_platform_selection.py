"""Select exact collected offers without fetching data or changing forecasts."""
from copy import deepcopy
import math
from dual_agent.preferred_sportsbooks import PLATFORMS, PICKEM

OPTIONS = ['DraftKings', 'FanDuel', 'PrizePicks', 'Underdog', 'Polymarket', 'All platforms']


def platform_key(label):
    if label == 'All platforms':
        return None
    return next((key for key, name in PLATFORMS.items() if name == label), str(label).lower())


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def decimal(offer):
    value = number(offer.get('decimal_odds'))
    if value is not None and value > 1:
        return value
    value = number(offer.get('american_odds'))
    if value is not None and abs(value) >= 100:
        return 1+value/100 if value > 0 else 1+100/abs(value)
    return None


def expected_return(row):
    """Model return per unit stake, conditional-cover probability plus pushes."""
    book = row.get('Best sportsbook')
    if book in PICKEM or row.get('Probability source') == 'sportsbook_implied':
        return None
    probability = number(row.get('Estimated chance'))
    price = number(row.get('Best decimal odds'))
    push = number(row.get('Estimated push chance', 0))
    if probability is None or not 0 <= probability <= 1 or price is None or price <= 1:
        return None
    if push is None or not 0 <= push < 1:
        return None
    return (1-push)*(probability*price-1)


def select_offer(row, platform='DraftKings'):
    """An offer list belongs only to this row's exact game/line/direction."""
    if not row.get('_offered', True):
        return None
    wanted = platform_key(platform)
    offers = list(row.get('Sportsbook offers') or [])
    # Older snapshots can prove their saved best offer only.
    best = row.get('Best sportsbook')
    if best and not any(o.get('book', o.get('bookmaker_key')) == best for o in offers):
        offers.append(dict(book=best, decimal_odds=row.get('Best decimal odds'),
                           updated_at=row.get('Captured UTC')))
    valid = []
    for offer in offers:
        book = str(offer.get('book', offer.get('bookmaker_key', ''))).lower()
        if book not in PLATFORMS or (wanted is not None and book != wanted):
            continue
        # Market-implied forecasts are tied to their original price source.
        if row.get('Probability source') == 'sportsbook_implied' and book != best:
            continue
        price = decimal(offer)
        if price is None and book not in PICKEM:
            continue
        valid.append(dict(offer, book=book, decimal_odds=price))
    if not valid:
        return None
    offer = max(valid, key=lambda item:item['decimal_odds'] or 0)
    selected = deepcopy(row)
    selected.update({'Best sportsbook':offer['book'], 'Best decimal odds':offer['decimal_odds'],
                     'Sportsbook offers':valid, '_offered':True,
                     'Selected quote UTC':offer.get('updated_at', offer.get('last_update', row.get('Captured UTC')))})
    selected['Estimated return per unit'] = expected_return(selected)
    if selected.get('_nfl_original'):
        original = dict(selected['_nfl_original'])
        price = offer['decimal_odds']
        original.update(best_sportsbook=offer['book'], best_decimal_odds=price,
                        sportsbook_offers=valid)
        if price is not None:
            original['best_american_odds'] = (price-1)*100 if price >= 2 else -100/(price-1)
        selected['_nfl_original'] = original
    return selected


def side_rank(row, ranking='Probability and price', platform='DraftKings'):
    probability = number(row.get('Estimated chance')) or 0
    value = expected_return(row)
    # Pick'em and cross-platform returns cannot be compared without entry rules.
    comparable = platform_key(platform) not in PICKEM and platform_key(platform) is not None
    if ranking == 'Probability and price' and comparable:
        return (0 if value is not None and value > 0 else 1,
                -probability, -(value if value is not None else -math.inf))
    return (0, -probability, 0)


def rank_pairs(pairs, ranking='Probability and price', platform='DraftKings'):
    for pair in pairs:
        offered = [(side,row) for side,row in pair['sides'].items() if row.get('_offered')]
        side, row = min(offered,key=lambda item:side_rank(item[1],ranking,platform))
        pair['recommended_side'] = side
        pair['_platform_rank'] = side_rank(row,ranking,platform)
    return sorted(pairs,key=lambda pair:(pair['_platform_rank'],str(pair['player']),
                                        str(pair['market']),pair['line']))
