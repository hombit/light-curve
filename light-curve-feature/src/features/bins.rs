use crate::evaluator::*;
use crate::extractor::FeatureExtractor;

use itertools::Itertools;
use unzip3::Unzip3;

/// Bins — sampled time series
///
/// Binning time series to bins with width $\mathrm{window}$ with respect to some $\mathrm{offset}$.
/// $j-th$ bin interval is
/// $[j \cdot \mathrm{window} + \mathrm{offset}; (j + 1) \cdot \mathrm{window} + \mathrm{offset})$.
/// Binned time series is defined by
/// $$
/// t_j^* = (j + \frac12) \cdot \mathrm{window} + \mathrm{offset},
/// $$
/// $$
/// m_j^* = \frac{\sum{m_i / \delta_i^2}}{\sum{\delta_i^{-2}}},
/// $$
/// $$
/// \delta_j^* = \frac{N_j}{\sum{\delta_i^{-2}}},
/// $$
/// where $N_j$ is a number of sampling observations and all sums are over observations inside
/// considering bin
///
/// - Depends on: **time**, **magnitude**, **magnitude error**
/// - Minimum number of observations: **1** (or as required by sub-features)
/// - Number of features: **$...$**
#[derive(Clone, Debug)]
pub struct Bins<T: Float> {
    window: T,
    offset: T,
    info: EvaluatorInfo,
    feature_names: Vec<String>,
    feature_descriptions: Vec<String>,
    feature_extractor: FeatureExtractor<T>,
}

impl<T> Bins<T>
where
    T: Float,
{
    pub fn new(window: T, offset: T) -> Self {
        assert!(window.is_sign_positive(), "window must be positive");
        Self {
            window,
            offset,
            info: EvaluatorInfo {
                size: 0,
                min_ts_length: 1,
                t_required: true,
                m_required: true,
                w_required: true,
                sorting_required: true,
            },
            feature_names: vec![],
            feature_descriptions: vec![],
            feature_extractor: feat_extr!(),
        }
    }

    pub fn set_window(&mut self, window: T) -> &mut Self {
        assert!(window.is_sign_positive(), "window must be positive");
        self.window = window;
        self
    }

    pub fn set_offset(&mut self, offset: T) -> &mut Self {
        self.offset = offset;
        self
    }

    /// Extend a feature to extract from binned time series
    pub fn add_feature(&mut self, feature: Box<dyn FeatureEvaluator<T>>) -> &mut Self {
        let window = self.window;
        let offset = self.offset;
        self.info.size += feature.size_hint();
        self.feature_names.extend(
            feature
                .get_names()
                .iter()
                .map(|name| format!("bins_window{:.1}_offset{:.1}_{}", window, offset, name)),
        );
        self.feature_descriptions
            .extend(feature.get_descriptions().iter().map(|desc| {
                format!(
                    "{desc} for binned time-series with window {window} and offset {offset}",
                    desc = desc,
                    window = window,
                    offset = offset,
                )
            }));
        self.feature_extractor.add_feature(feature);
        self
    }

    fn transform_ts(&self, ts: &mut TimeSeries<T>) -> Result<TmwArrays<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let (t, m, w): (Vec<_>, Vec<_>, Vec<_>) =
            ts.t.as_slice()
                .iter()
                .copied()
                .zip(ts.m.as_slice().iter().copied())
                .zip(ts.w.as_slice().iter().copied())
                .map(|((t, m), w)| (t, m, w))
                .group_by(|(t, _, _)| ((*t - self.offset) / self.window).floor())
                .into_iter()
                .map(|(x, group)| {
                    let bin_t = (x + T::half()) * self.window;
                    let (n, bin_m, norm) = group
                        .fold((T::zero(), T::zero(), T::zero()), |acc, (_, m, w)| {
                            (acc.0 + T::one(), acc.1 + m * w, acc.2 + w)
                        });
                    let bin_m = bin_m / norm;
                    let bin_w = norm / n;
                    (bin_t, bin_m, bin_w)
                })
                .unzip3();
        Ok(TmwArrays {
            t: t.into(),
            m: m.into(),
            w: w.into(),
        })
    }

    #[inline]
    pub fn default_window() -> T {
        T::one()
    }

    #[inline]
    pub fn default_offset() -> T {
        T::zero()
    }
}

impl<T> Default for Bins<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(Self::default_window(), Self::default_offset())
    }
}

impl<T> FeatureEvaluator<T> for Bins<T>
where
    T: Float,
{
    transformer_eval!();
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::features::Amplitude;
    use crate::tests::*;

    eval_info_test!(bins_info, {
        let mut bins = Bins::default();
        bins.add_feature(Box::new(Amplitude::default()));
        bins
    });

    #[test]
    fn bins() {
        let t = [0.0_f32, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 5.0];
        let m = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let w = [10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0];
        let mut ts = TimeSeries::new(&t, &m, &w);

        let desired_t = [0.5, 1.5, 2.5, 5.5];
        let desired_m = [0.0, 2.0, 6.333333333333333, 10.0];
        let desired_w = [10.0, 6.666666666666667, 7.5, 10.0];

        let bins = Bins::new(1.0, 0.0);
        let actual_tmw = bins.transform_ts(&mut ts).unwrap();

        assert_eq!(actual_tmw.t.len(), actual_tmw.m.len());
        assert_eq!(actual_tmw.t.len(), actual_tmw.w.len());
        all_close(&actual_tmw.t.as_slice().unwrap(), &desired_t, 1e-6);
        all_close(&actual_tmw.m.as_slice().unwrap(), &desired_m, 1e-6);
        all_close(&actual_tmw.w.as_slice().unwrap(), &desired_w, 1e-6);
    }

    #[test]
    fn bins_windows_and_offsets() {
        let t = [0.0_f32, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 5.0];
        let m = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut ts = TimeSeries::new_without_weight(&t, &m);

        let mut len = |window, offset| {
            let tmw = Bins::new(window, offset).transform_ts(&mut ts).unwrap();
            assert_eq!(tmw.t.len(), tmw.m.len());
            assert_eq!(tmw.m.len(), tmw.w.len());
            tmw.t.len()
        };

        assert_eq!(len(2.0, 0.0), 3);
        assert_eq!(len(3.0, 0.0), 2);
        assert_eq!(len(10.0, 0.0), 1);
        assert_eq!(len(1.0, 0.1), 5);
        assert_eq!(len(1.0, 0.5), 5);
        assert_eq!(len(2.0, 1.0), 3);
    }

    // Add more Bins::get_info() tests for non-trivial cases
}
