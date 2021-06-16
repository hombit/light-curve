use crate::float_trait::Float;
use crate::sorted_array::SortedArray;
use crate::statistics::Statistics;

use conv::prelude::*;
use itertools::{Either, Itertools};
use std::iter;

#[derive(Clone, Debug)]
pub struct DataSample<'a, T>
where
    T: Float,
{
    pub sample: &'a [T],
    sorted: Option<SortedArray<T>>,
    min: Option<T>,
    max: Option<T>,
    mean: Option<T>,
    median: Option<T>,
    std: Option<T>,
    std2: Option<T>,
}

macro_rules! data_sample_getter {
    ($attr: ident, $getter: ident, $method: ident) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some(self.sample.$method());
                    self.$attr.unwrap()
                }
            }
        }
    };
    ($attr: ident, $getter: ident, $method: ident, $method_sorted: ident) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some(match self.sorted.as_ref() {
                        Some(sorted) => sorted.$method_sorted(),
                        None => self.sample.$method(),
                    });
                    self.$attr.unwrap()
                }
            }
        }
    };
    ($attr: ident, $getter: ident, $func: expr) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some($func(self));
                    self.$attr.unwrap()
                }
            }
        }
    };
}

impl<'a, T> DataSample<'a, T>
where
    T: Float,
    [T]: Statistics<T>,
{
    fn new(sample: &'a [T]) -> Self {
        Self {
            sample,
            sorted: None,
            min: None,
            max: None,
            mean: None,
            median: None,
            std: None,
            std2: None,
        }
    }

    pub fn get_sorted(&mut self) -> &SortedArray<T> {
        if self.sorted.is_none() {
            self.sorted = Some(self.sample.into());
        }
        self.sorted.as_ref().unwrap()
    }

    data_sample_getter!(min, get_min, minimum, minimum);
    data_sample_getter!(max, get_max, maximum, maximum);
    data_sample_getter!(mean, get_mean, mean);
    data_sample_getter!(median, get_median, |ds: &mut DataSample<'a, T>| {
        ds.get_sorted().median()
    });
    data_sample_getter!(std, get_std, |ds: &mut DataSample<'a, T>| {
        ds.get_std2().sqrt()
    });
    data_sample_getter!(std2, get_std2, |ds: &mut DataSample<'a, T>| {
        let mean = ds.get_mean();
        ds.sample.iter().map(|&x| (x - mean).powi(2)).sum::<T>()
            / (ds.sample.len() - 1).value_as::<T>().unwrap()
    });

    pub fn signal_to_noise(&mut self, value: T) -> T {
        if self.get_std().is_zero() {
            T::zero()
        } else {
            (value - self.get_mean()) / self.get_std()
        }
    }
}

#[derive(Clone, Debug)]
pub struct TimeSeries<'a, T>
where
    T: Float,
{
    pub t: DataSample<'a, T>,
    pub m: DataSample<'a, T>,
    pub w: Option<DataSample<'a, T>>,
    m_weighted_mean: Option<T>,
    m_reduced_chi2: Option<T>,
    t_max_m: Option<T>,
    t_min_m: Option<T>,
}

macro_rules! time_series_getter {
    ($attr: ident, $getter: ident, $func: expr) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some($func(self));
                    self.$attr.unwrap()
                }
            }
        }
    };
}

impl<'a, T> TimeSeries<'a, T>
where
    T: Float,
{
    pub fn new(t: &'a [T], m: &'a [T], w: Option<&'a [T]>) -> Self {
        assert_eq!(t.len(), m.len(), "t and m should have the same size");
        Self {
            t: DataSample::new(t),
            m: DataSample::new(m),
            w: w.map(|w| {
                assert_eq!(m.len(), w.len(), "m and err should have the same size");
                DataSample::new(w)
            }),
            m_weighted_mean: None,
            m_reduced_chi2: None,
            t_max_m: None,
            t_min_m: None,
        }
    }

    #[inline]
    pub fn lenu(&self) -> usize {
        self.t.sample.len()
    }

    pub fn lenf(&self) -> T {
        self.lenu().value_as::<T>().unwrap()
    }

    /// Weight iterator, if weight is None, returns iterator with ones
    pub fn w_iter(&self) -> impl Iterator<Item = T> + 'a {
        match self.w.as_ref() {
            Some(w) => Either::Left(w.sample.iter().copied()),
            None => Either::Right(iter::repeat_with(T::one).take(self.lenu())),
        }
    }

    /// (t, m) pair iter
    pub fn tm_iter(&self) -> impl Iterator<Item = (T, T)> + 'a {
        self.t
            .sample
            .iter()
            .copied()
            .zip(self.m.sample.iter().copied())
    }

    /// (t, w) pair iter
    pub fn tw_iter(&self) -> impl Iterator<Item = (T, T)> + 'a {
        self.t.sample.iter().copied().zip(self.w_iter())
    }

    /// (m, w) pair iterator
    pub fn mw_iter(&self) -> impl Iterator<Item = (T, T)> + 'a {
        self.m.sample.iter().copied().zip(self.w_iter())
    }

    pub fn tmw_iter(&self) -> impl Iterator<Item = (T, T, T)> + 'a {
        self.t
            .sample
            .iter()
            .zip(self.m.sample.iter().zip(self.w_iter()))
            .map(|(&t, (&m, w))| (t, m, w))
    }

    time_series_getter!(
        m_weighted_mean,
        get_m_weighted_mean,
        |ts: &mut TimeSeries<T>| {
            ts.mw_iter().map(|(y, w)| y * w).sum::<T>() / ts.w_iter().sum()
        }
    );

    time_series_getter!(m_reduced_chi2, get_m_reduced_chi2, |ts: &mut TimeSeries<
        T,
    >| {
        let m_weighed_mean = ts.get_m_weighted_mean();
        ts.mw_iter()
            .map(|(y, w)| (y - m_weighed_mean).powi(2) * w)
            .sum::<T>()
            / (ts.lenf() - T::one())
    });

    fn set_t_min_max_m(&mut self) {
        let (i_min, i_max) = self
            .m
            .sample
            .iter()
            .position_minmax()
            .into_option()
            .expect("time series must to be non-empty");
        self.t_min_m = Some(self.t.sample[i_min]);
        self.t_max_m = Some(self.t.sample[i_max]);
    }

    pub fn get_t_min_m(&mut self) -> T {
        if self.t_min_m.is_none() {
            self.set_t_min_max_m()
        }
        self.t_min_m.unwrap()
    }

    pub fn get_t_max_m(&mut self) -> T {
        if self.t_max_m.is_none() {
            self.set_t_min_max_m()
        }
        self.t_max_m.unwrap()
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    use light_curve_common::all_close;

    macro_rules! data_sample_test {
        ($name: ident, $method: ident, $desired: tt, $x: tt $(,)?) => {
            #[test]
            fn $name() {
                let x = $x;
                let desired = $desired;

                let mut ds = DataSample::new(&x[..]);
                all_close(&[ds.$method()], &desired[..], 1e-6);
                all_close(&[ds.$method()], &desired[..], 1e-6);

                let mut ds = DataSample::new(&x[..]);
                ds.get_sorted();
                all_close(&[ds.$method()], &desired[..], 1e-6);
                all_close(&[ds.$method()], &desired[..], 1e-6);
            }
        };
    }

    data_sample_test!(
        data_sample_min,
        get_min,
        [-7.79420906],
        [3.92948846, 3.28436964, 6.73375373, -7.79420906, -7.23407407],
    );

    data_sample_test!(
        data_sample_max,
        get_max,
        [6.73375373],
        [3.92948846, 3.28436964, 6.73375373, -7.79420906, -7.23407407],
    );

    data_sample_test!(
        data_sample_mean,
        get_mean,
        [-0.21613426],
        [3.92948846, 3.28436964, 6.73375373, -7.79420906, -7.23407407],
    );

    data_sample_test!(
        data_sample_median_odd,
        get_median,
        [3.28436964],
        [3.92948846, 3.28436964, 6.73375373, -7.79420906, -7.23407407],
    );

    data_sample_test!(
        data_sample_median_even,
        get_median,
        [5.655794743124782],
        [9.47981408, 3.86815751, 9.90299294, -2.986894, 7.44343197, 1.52751816],
    );

    data_sample_test!(
        data_sample_std,
        get_std,
        [6.7900544035968435],
        [3.92948846, 3.28436964, 6.73375373, -7.79420906, -7.23407407],
    );

    #[test]
    fn time_series_m_weighted_mean() {
        let t: Vec<_> = (0..5).map(|i| i as f64).collect();
        let m = [
            12.77883145,
            18.89988406,
            17.55633632,
            18.36073996,
            11.83854198,
        ];
        let w = [0.1282489, 0.10576467, 0.32102692, 0.12962352, 0.10746144];
        let mut ts = TimeSeries::new(&t[..], &m[..], Some(&w[..]));
        // np.average(m, weights=w)
        let desired = [16.31817047752941];
        all_close(&[ts.get_m_weighted_mean()], &desired[..], 1e-6);
    }

    #[test]
    fn time_series_m_reduced_chi2() {
        let t: Vec<_> = (0..5).map(|i| i as f64).collect();
        let m = [
            12.77883145,
            18.89988406,
            17.55633632,
            18.36073996,
            11.83854198,
        ];
        let w = [0.1282489, 0.10576467, 0.32102692, 0.12962352, 0.10746144];
        let mut ts = TimeSeries::new(&t[..], &m[..], Some(&w[..]));
        let desired = [1.3752251301435465];
        all_close(&[ts.get_m_reduced_chi2()], &desired[..], 1e-6);
    }
}
