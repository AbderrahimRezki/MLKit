from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X, y): pass


class Transformer(BaseEstimator):
    @abstractmethod
    def transform(self, X): pass

    @abstractmethod
    def fit_transform(self, X): pass


class Predictor(BaseEstimator):
    @abstractmethod
    def predict(self, X): pass