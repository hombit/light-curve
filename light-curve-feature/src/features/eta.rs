use crate::evaluator::*;
use itertools::Itertools;

/// Von Neummann $\eta$
///
/// $$
/// \eta \equiv \frac1{(N - 1)\\,\sigma_m^2} \sum_{i=0}^{N-2}(m_{i+1} - m_i)^2,
/// $$
/// where $N$ is the number of observations,
/// $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// Kim et al. 2014, [DOI:10.1051/0004-6361/201323252](https://doi.org/10.1051/0004-6361/201323252)
#[derive(Clone, Default, Debug)]
pub struct Eta {}

impl Eta {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    ETA_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: true,
);

impl<T> FeatureEvaluator<T> for Eta
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_std2 = get_nonzero_m_std2(ts)?;
        let value =
            ts.m.as_slice()
                .iter()
                .tuple_windows()
                .map(|(&a, &b)| (b - a).powi(2))
                .sum::<T>()
                / (ts.lenf() - T::one())
                / m_std2;
        Ok(vec![value])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &ETA_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["eta"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["Von Neummann eta-coefficient for magnitude sample"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(eta_info, Eta::default());

    feature_test!(
        eta,
        [Box::new(Eta::new())],
        [1.11338],
        [1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 109.0],
    );
}
