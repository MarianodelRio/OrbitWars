def simulate_combat(garrison, owner, attackers_dict):
    """
    attackers_dict: {player_id: ship_count}
    Returns (surviving_ships, new_owner)

    Official rules:
    1. Group arriving fleets by owner (already done via attackers_dict).
    2. Largest attacking force fights second largest; difference survives.
       If tied, both are destroyed. Repeat until one attacker remains.
    3. Surviving attacker vs garrison:
       - Same owner as planet → ships added to garrison.
       - Different owner → subtract from garrison; if exceeds, planet flips.
       - Ties favour the defender (attacker must strictly exceed garrison).
    """
    if not attackers_dict:
        return garrison, owner

    # Sort descending by ship count
    attackers = sorted(attackers_dict.items(), key=lambda x: x[1], reverse=True)

    # Resolve attacker-vs-attacker until at most one remains
    while len(attackers) >= 2:
        top_id, top_ships = attackers[0]
        sec_id, sec_ships = attackers[1]
        if top_ships == sec_ships:
            # Tie: both factions destroyed
            attackers = attackers[2:]
        else:
            surviving = top_ships - sec_ships
            attackers = [(top_id, surviving)] + list(attackers[2:])
        attackers.sort(key=lambda x: x[1], reverse=True)

    if not attackers:
        # All attacking ships destroyed; garrison unchanged
        return garrison, owner

    attacker_id, attack_ships = attackers[0]
    if attacker_id == owner:
        # Friendly reinforcement
        return garrison + attack_ships, owner
    elif attack_ships > garrison:
        # Planet captured
        return attack_ships - garrison, attacker_id
    else:
        # Defender holds (ties favour defender)
        return garrison - attack_ships, owner
