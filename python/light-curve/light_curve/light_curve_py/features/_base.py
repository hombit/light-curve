from abc import ABC, abstractmethod


class BaseFeature(ABC):
    @abstractmethod
    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        pass
