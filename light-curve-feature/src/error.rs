#[derive(Debug, thiserror::Error, PartialEq)]
pub enum EvaluatorError {
    #[error("time-series' length {actual} is smaller than the minimum required length {minimum}")]
    ShortTimeSeries { actual: usize, minimum: usize },

    #[error("feature value is undefined for a flat time series")]
    FlatTimeSeries,

    #[error("zero division: {0}")]
    ZeroDivision(&'static str),
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SortedArrayError {
    #[error("SortedVec constructors accept only sorted arrays")]
    Unsorted,
    #[error("SortedVec constructors accept only contiguous arrays")]
    NonContiguous,
}
