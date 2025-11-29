def calculate_ev(prob, odds=2.0):
    return prob * odds - 1

def kelly_stake(ev, bankroll):
    return max(0, ev * bankroll)
