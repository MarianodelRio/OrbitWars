from training.rewards.shaped import shaped_reward

def self_play_loop(bot_factory, n_episodes=10, steps=200):
    """
    Scaffold for self-play RL. Runs episodes and collects (obs, action, reward) tuples.
    Does NOT update gradients — for future RL integration.
    """
    from game.env.runner import run_match

    for episode in range(n_episodes):
        bot1 = bot_factory()
        bot2 = bot_factory()
        result = run_match(bot1, bot2, steps=steps, render=False)
        print(f"Episode {episode+1}/{n_episodes}: winner={result['winner']}, rewards={result['rewards']}")
    print("Self-play scaffold complete. Gradient updates not implemented.")
