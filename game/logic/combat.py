def simulate_combat(garrison, owner, attackers_dict):
    """
    attackers_dict: {player_id: ship_count}
    Returns (surviving_ships, new_owner)
    Replicates game rules: attacker with most ships wins ties go to defender.
    """
    current_ships = garrison
    current_owner = owner
    for attacker_id, attack_ships in attackers_dict.items():
        if attacker_id == current_owner:
            current_ships += attack_ships
        else:
            if attack_ships > current_ships:
                current_ships = attack_ships - current_ships
                current_owner = attacker_id
            else:
                current_ships -= attack_ships
    return current_ships, current_owner
