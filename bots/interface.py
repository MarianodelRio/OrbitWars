from abc import ABC, abstractmethod

class Bot(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def act(self, obs, config) -> list:
        ...

    def __call__(self, obs, config=None):
        return self.act(obs, config)

def make_agent(bot_instance):
    def agent_fn(obs, config=None):
        return bot_instance(obs, config)
    return agent_fn
