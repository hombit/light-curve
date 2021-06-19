use crate::evaluator::*;
use itertools::Itertools;

/// Maximum slope between two sub-sequential observations
///
/// $$
/// \mathrm{maximum~slope} \equiv \max_{i=0..N-2}\left|\frac{m_{i+1} - m_i}{t_{i+1} - t_i}\right|
/// $$
///
/// - Depends on: **time**, **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
#[derive(Clone, Default, Debug)]
pub struct MaximumSlope {}

lazy_info!(
    MAXIMUM_SLOPE_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: true,
    m_required: true,
    w_required: false,
    sorting_required: true,
);

impl MaximumSlope {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for MaximumSlope
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let result =
            ts.t.as_slice()
                .iter()
                .tuple_windows()
                .map(|(&t1, &t2)| t2 - t1)
                .zip(
                    ts.m.as_slice()
                        .iter()
                        .tuple_windows()
                        .map(|(&m1, &m2)| m2 - m1),
                )
                .map(|(dt, dm)| T::abs(dm / dt))
                .filter(|&x| x.is_finite())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .expect("All points of the light curve have the same time");
        Ok(vec![result])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &MAXIMUM_SLOPE_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["maximum_slope"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["maximum slope of time-series"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(maximum_slope_info, MaximumSlope::default());

    feature_test!(
        maximum_slope_positive,
        [Box::new(MaximumSlope::new())],
        [1.0],
        [0.0_f32, 2.0, 4.0, 5.0, 7.0, 9.0],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
    );

    feature_test!(
        maximum_slope_negative,
        [Box::new(MaximumSlope::new())],
        [1.0],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0_f32, 0.5, 1.0, 0.0, 0.5, 1.0],
    );
}
