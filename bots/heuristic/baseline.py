from bots.interface import Bot

class BaselineBot(Bot):
    def act(self, obs, config=None):
        return []

agent_fn = BaselineBot()
