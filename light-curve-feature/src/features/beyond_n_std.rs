use crate::evaluator::*;

use conv::ConvUtil;

/// Fraction of observations beyond $n\\,\sigma\_m$ from the mean magnitude $\langle m \rangle$
///
/// $$
/// \mathrm{beyond}~n\\,\sigma\_m \equiv \frac{\sum\_i I\_{|m - \langle m \rangle| > n\\,\sigma\_m}(m_i)}{N},
/// $$
/// where $I$ is the [indicator function](https://en.wikipedia.org/wiki/Indicator_function),
/// $N$ is the number of observations,
/// $\langle m \rangle$ is the mean magnitude
/// and $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
/// ```
/// use light_curve_feature::*;
/// use light_curve_common::all_close;
/// use std::f64::consts::SQRT_2;
///
/// let fe = feat_extr!(BeyondNStd::default(), BeyondNStd::new(2.0));
/// let time = [0.0; 21];  // Doesn't depend on time
/// let mut magn = vec![0.0; 17];
/// magn.extend_from_slice(&[SQRT_2, -SQRT_2, 2.0 * SQRT_2, -2.0 * SQRT_2]);
/// let mut ts = TimeSeries::new_without_weight(&time[..], &magn[..]);
/// assert_eq!(0.0, ts.m.get_mean());
/// assert!((1.0 - ts.m.get_std()).abs() < 1e-15);
/// assert_eq!(vec![4.0 / 21.0, 2.0 / 21.0], fe.eval(&mut ts).unwrap());
/// ```
#[derive(Clone, Debug)]
pub struct BeyondNStd<T> {
    nstd: T,
    name: String,
    description: String,
}

impl<T> BeyondNStd<T>
where
    T: Float,
{
    pub fn new(nstd: T) -> Self {
        assert!(nstd > T::zero(), "nstd should be positive");
        Self {
            nstd,
            name: format!("beyond_{:.0}_std", nstd),
            description: format!(
                "fraction of observations which magnitudes are beyond {:.3e} standard deviations \
                from the mean magnitude",
                nstd
            ),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    #[inline]
    pub fn default_nstd() -> T {
        T::one()
    }
}

lazy_info!(
    BEYOND_N_STD_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl<T> Default for BeyondNStd<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(Self::default_nstd())
    }
}

impl<T> FeatureEvaluator<T> for BeyondNStd<T>
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_mean = ts.m.get_mean();
        let threshold = ts.m.get_std() * self.nstd;
        Ok(vec![
            ts.m.sample
                .iter()
                .filter(|&&y| T::abs(y - m_mean) > threshold)
                .count()
                .value_as::<T>()
                .unwrap()
                / ts.lenf(),
        ])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &BEYOND_N_STD_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![self.description.as_str()]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(beyond_n_std_info, BeyondNStd::default());

    feature_test!(
        beyond_n_std,
        [
            Box::new(BeyondNStd::default()),
            Box::new(BeyondNStd::new(1.0)), // should be the same as the previous one
            Box::new(BeyondNStd::new(2.0)),
        ],
        [0.2, 0.2, 0.0],
        [1.0_f32, 2.0, 3.0, 4.0, 100.0],
    );
}
