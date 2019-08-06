from abc import ABC, abstractmethod

class AbstractMetric(ABC):

    @abstractmethod
    def calculate_metric(self, *args, **kwargs):
        pass

    @abstractmethod
    def __str__(self, *args, **kwargs):
        pass