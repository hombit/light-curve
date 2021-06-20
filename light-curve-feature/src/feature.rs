use crate::evaluator::*;
use crate::extractor::FeatureExtractor;
use crate::features::antifeatures::*;
use crate::features::*;
use crate::float_trait::Float;
use crate::time_series::TimeSeries;

use enum_dispatch::enum_dispatch;
use std::fmt::Debug;

#[enum_dispatch(FeatureEvaluator<T>)]
#[derive(Clone, Debug)]
pub enum Feature<T>
where
    T: Float,
{
    // extractor
    FeatureExtractor(FeatureExtractor<T, Feature<T>>),
    // features
    Amplitude,
    AndersonDarlingNormal,
    #[cfg(feature = "gsl")]
    BazinFit,
    BeyondNStd(BeyondNStd<T>),
    Bins(Bins<T, Feature<T>>),
    Cusum,
    Eta,
    EtaE,
    ExcessVariance,
    InterPercentileRange,
    Kurtosis,
    LinearFit,
    LinearTrend,
    MagnitudePercentageRatio,
    MaximumSlope,
    Mean,
    MeanVariance,
    Median,
    MedianAbsoluteDeviation,
    MedianBufferRangePercentage(MedianBufferRangePercentage<T>),
    PercentAmplitude,
    PercentDifferenceMagnitudePercentile,
    Periodogram(Periodogram<T, Feature<T>>),
    _PeriodogramPeaks,
    ReducedChi2,
    Skew,
    StandardDeviation,
    StetsonK,
    WeightedMean,
    // antifeatures
    Duration,
    MaximumTimeInterval,
    MinimumTimeInterval,
    ObservationCount,
    TimeMean,
    TimeStandardDeviation,
}
