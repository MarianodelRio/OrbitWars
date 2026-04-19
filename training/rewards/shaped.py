def shaped_reward(prev_obs, curr_obs, player):
    """Dense reward: delta_planets + delta_ships*0.01 + delta_production*0.1"""
    def count(obs, attr_idx, condition):
        return sum(p[attr_idx] for p in obs.get("planets", []) if condition(p))

    prev_planets = sum(1 for p in prev_obs.get("planets", []) if p[1] == player)
    curr_planets = sum(1 for p in curr_obs.get("planets", []) if p[1] == player)
    delta_planets = curr_planets - prev_planets

    prev_ships = count(prev_obs, 5, lambda p: p[1] == player)
    curr_ships = count(curr_obs, 5, lambda p: p[1] == player)
    delta_ships = curr_ships - prev_ships

    prev_prod = count(prev_obs, 6, lambda p: p[1] == player)
    curr_prod = count(curr_obs, 6, lambda p: p[1] == player)
    delta_production = curr_prod - prev_prod

    return delta_planets + 0.01 * delta_ships + 0.1 * delta_production
