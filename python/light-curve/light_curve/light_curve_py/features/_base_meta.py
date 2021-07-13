from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Collection, Union

from ._base import BaseFeature
from .extractor import Extractor, _PyExtractor
from light_curve.light_curve_ext import Extractor as _RustExtractor, _FeatureEvaluator as _RustBaseFeature


@dataclass
class BaseMetaFeature(BaseFeature):
    features: Collection[Union[BaseFeature, _RustBaseFeature]] = ()
    extractor: Union[_RustExtractor, _PyExtractor] = field(init=False)

    def __post_init__(self):
        self.extractor = Extractor(self.features)

    @abstractmethod
    def transform(self, t, m, sigma):
        """Must return temporarily sorted arrays (t, m, sigma)"""
        pass

    def __call__(self, t, m, sigma=None, sorted=None, fill_value=None):
        t, m, sigma = self.transform(t, m, sigma)
        return self.extractor(t, m, sigma, sorted=True, fill_value=fill_value)
