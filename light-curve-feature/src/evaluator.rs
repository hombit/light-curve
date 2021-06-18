pub use crate::error::EvaluatorError;
pub use crate::float_trait::Float;
pub use crate::time_series::TimeSeries;

use dyn_clonable::*;
pub use lazy_static::lazy_static;
use ndarray::Array1;
pub use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatorInfo {
    pub size: usize,
    pub min_ts_length: usize,
    pub t_required: bool,
    pub m_required: bool,
    pub w_required: bool,
    pub sorting_required: bool,
}

/// The trait each feature should implement
#[clonable]
pub trait FeatureEvaluator<T: Float>: Send + Clone + Debug {
    /// Should return the vector of feature values or `EvaluatorError`
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError>;

    /// Should return the vector of feature values and fill invalid components with given value
    fn eval_or_fill(&self, ts: &mut TimeSeries<T>, fill_value: T) -> Vec<T> {
        match self.eval(ts) {
            Ok(v) => v,
            Err(_) => vec![fill_value; self.size_hint()],
        }
    }

    /// Get feature evaluator meta-information
    fn get_info(&self) -> &EvaluatorInfo;

    /// Should return the vector of feature names. The length and feature order should
    /// correspond to `eval()` output
    fn get_names(&self) -> Vec<&str>;

    /// Shoud return the vector of feature descriptions. The length and feature order should
    /// correspond to `eval()` output
    fn get_descriptions(&self) -> Vec<&str>;

    /// Should return the size of vectors returned by `eval()` and `get_names()`
    fn size_hint(&self) -> usize {
        self.get_info().size
    }

    /// Should return minimum time series length to successfully find feature value
    fn min_ts_length(&self) -> usize {
        self.get_info().min_ts_length
    }

    fn is_t_required(&self) -> bool {
        self.get_info().t_required
    }

    fn is_m_required(&self) -> bool {
        self.get_info().m_required
    }

    fn is_w_required(&self) -> bool {
        self.get_info().w_required
    }

    fn is_sorting_required(&self) -> bool {
        self.get_info().sorting_required
    }

    fn check_ts_length(&self, ts: &TimeSeries<T>) -> Result<usize, EvaluatorError> {
        let length = ts.lenu();
        if length < self.min_ts_length() {
            Err(EvaluatorError::ShortTimeSeries {
                actual: length,
                minimum: self.min_ts_length(),
            })
        } else {
            Ok(length)
        }
    }
}

pub type VecFe<T> = Vec<Box<dyn FeatureEvaluator<T>>>;

pub fn get_nonzero_m_std<T: Float>(ts: &mut TimeSeries<T>) -> Result<T, EvaluatorError> {
    let std = ts.m.get_std();
    if std.is_zero() || ts.is_plateau() {
        Err(EvaluatorError::FlatTimeSeries)
    } else {
        Ok(std)
    }
}

pub fn get_nonzero_m_std2<T: Float>(ts: &mut TimeSeries<T>) -> Result<T, EvaluatorError> {
    let std2 = ts.m.get_std2();
    if std2.is_zero() || ts.is_plateau() {
        Err(EvaluatorError::FlatTimeSeries)
    } else {
        Ok(std2)
    }
}

pub fn get_nonzero_reduced_chi2<T: Float>(ts: &mut TimeSeries<T>) -> Result<T, EvaluatorError> {
    let reduced_chi2 = ts.get_m_reduced_chi2();
    if reduced_chi2.is_zero() || ts.is_plateau() {
        Err(EvaluatorError::FlatTimeSeries)
    } else {
        Ok(reduced_chi2)
    }
}

pub trait OwnedArrays<T>
where
    T: Float,
{
    fn ts(self) -> TimeSeries<'static, T>;
}

pub struct TmArrays<T> {
    pub t: Array1<T>,
    pub m: Array1<T>,
}

impl<T> OwnedArrays<T> for TmArrays<T>
where
    T: Float,
{
    fn ts(self) -> TimeSeries<'static, T> {
        TimeSeries::new_without_weight(self.t, self.m)
    }
}

pub struct TmwArrays<T> {
    pub t: Array1<T>,
    pub m: Array1<T>,
    pub w: Array1<T>,
}

impl<T> OwnedArrays<T> for TmwArrays<T>
where
    T: Float,
{
    fn ts(self) -> TimeSeries<'static, T> {
        TimeSeries::new(self.t, self.m, self.w)
    }
}
