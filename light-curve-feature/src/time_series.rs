use crate::float_trait::Float;
use crate::sorted_array::SortedArray;
use crate::types::CowArray1;

use conv::prelude::*;
use itertools::Itertools;
use ndarray::{s, Array1, ArrayView1, Zip};
use ndarray_stats::{QuantileExt, SummaryStatisticsExt};

#[derive(Clone, Debug)]
pub struct DataSample<'a, T>
where
    T: Float,
{
    pub sample: CowArray1<'a, T>,
    sorted: Option<SortedArray<T>>,
    min: Option<T>,
    max: Option<T>,
    mean: Option<T>,
    median: Option<T>,
    std: Option<T>,
    std2: Option<T>,
}

macro_rules! data_sample_getter {
    ($attr: ident, $getter: ident, $func: expr, $method_sorted: ident) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some(match self.sorted.as_ref() {
                        Some(sorted) => sorted.$method_sorted(),
                        None => $func(self),
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
{
    fn new(sample: CowArray1<'a, T>) -> Self {
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

    pub fn as_slice(&mut self) -> &[T] {
        if !self.sample.is_standard_layout() {
            let owned: Array1<_> = self.sample.iter().copied().collect::<Vec<_>>().into();
            self.sample = owned.into();
        }
        self.sample.as_slice().unwrap()
    }

    pub fn get_sorted(&mut self) -> &SortedArray<T> {
        if self.sorted.is_none() {
            self.sorted = Some(self.sample.to_vec().into());
        }
        self.sorted.as_ref().unwrap()
    }

    data_sample_getter!(
        min,
        get_min,
        |ds: &mut DataSample<'a, T>| { *ds.sample.min().expect("time series must be non-empty") },
        minimum
    );
    data_sample_getter!(
        max,
        get_max,
        |ds: &mut DataSample<'a, T>| { *ds.sample.max().expect("time series must be non-empty") },
        maximum
    );
    data_sample_getter!(mean, get_mean, |ds: &mut DataSample<'a, T>| {
        ds.sample.mean().expect("time series must be non-empty")
    });
    data_sample_getter!(median, get_median, |ds: &mut DataSample<'a, T>| {
        ds.get_sorted().median()
    });
    data_sample_getter!(std, get_std, |ds: &mut DataSample<'a, T>| {
        ds.get_std2().sqrt()
    });
    data_sample_getter!(std2, get_std2, |ds: &mut DataSample<'a, T>| {
        ds.sample.var(T::one())
    });

    pub fn signal_to_noise(&mut self, value: T) -> T {
        if self.get_std().is_zero() {
            T::zero()
        } else {
            (value - self.get_mean()) / self.get_std()
        }
    }
}

impl<'a, T, Slice: ?Sized> From<&'a Slice> for DataSample<'a, T>
where
    T: Float,
    Slice: AsRef<[T]>,
{
    fn from(s: &'a Slice) -> Self {
        ArrayView1::from(s).into()
    }
}

impl<'a, T> From<Vec<T>> for DataSample<'a, T>
where
    T: Float,
{
    fn from(v: Vec<T>) -> Self {
        Array1::from(v).into()
    }
}

impl<'a, T> From<ArrayView1<'a, T>> for DataSample<'a, T>
where
    T: Float,
{
    fn from(a: ArrayView1<'a, T>) -> Self {
        Self::new(a.into())
    }
}

impl<'a, T> From<Array1<T>> for DataSample<'a, T>
where
    T: Float,
{
    fn from(a: Array1<T>) -> Self {
        Self::new(a.into())
    }
}

#[derive(Clone, Debug)]
pub struct TimeSeries<'a, T>
where
    T: Float,
{
    pub t: DataSample<'a, T>,
    pub m: DataSample<'a, T>,
    pub w: DataSample<'a, T>,
    m_weighted_mean: Option<T>,
    m_reduced_chi2: Option<T>,
    t_max_m: Option<T>,
    t_min_m: Option<T>,
    plateau: Option<bool>,
}

macro_rules! time_series_getter {
    ($t: ty, $attr: ident, $getter: ident, $func: expr) => {
        pub fn $getter(&mut self) -> $t {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some($func(self));
                    self.$attr.unwrap()
                }
            }
        }
    };

    ($attr: ident, $getter: ident, $func: expr) => {
        time_series_getter!(T, $attr, $getter, $func);
    };
}

impl<'a, T> TimeSeries<'a, T>
where
    T: Float,
{
    pub fn new(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
        w: impl Into<DataSample<'a, T>>,
    ) -> Self {
        let t = t.into();
        let m = m.into();
        let w = w.into();

        assert_eq!(
            t.sample.len(),
            m.sample.len(),
            "t and m should have the same size"
        );
        assert_eq!(
            m.sample.len(),
            w.sample.len(),
            "m and err should have the same size"
        );

        Self {
            t,
            m,
            w,
            m_weighted_mean: None,
            m_reduced_chi2: None,
            t_max_m: None,
            t_min_m: None,
            plateau: None,
        }
    }

    pub fn new_without_weight(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
    ) -> Self {
        let t = t.into();
        let m = m.into();

        assert_eq!(
            t.sample.len(),
            m.sample.len(),
            "t and m should have the same size"
        );

        let w = T::array0_unity().broadcast(t.sample.len()).unwrap().into();

        Self {
            t,
            m,
            w,
            m_weighted_mean: None,
            m_reduced_chi2: None,
            t_max_m: None,
            t_min_m: None,
            plateau: None,
        }
    }

    #[inline]
    pub fn lenu(&self) -> usize {
        self.t.sample.len()
    }

    pub fn lenf(&self) -> T {
        self.lenu().value_as::<T>().unwrap()
    }

    time_series_getter!(
        m_weighted_mean,
        get_m_weighted_mean,
        |ts: &mut TimeSeries<T>| { ts.m.sample.weighted_mean(&ts.w.sample).unwrap() }
    );

    time_series_getter!(m_reduced_chi2, get_m_reduced_chi2, |ts: &mut TimeSeries<
        T,
    >| {
        let m_weighed_mean = ts.get_m_weighted_mean();
        let m_reduced_chi2 = Zip::from(&ts.m.sample)
            .and(&ts.w.sample)
            .fold(T::zero(), |chi2, &m, &w| {
                chi2 + (m - m_weighed_mean).powi(2) * w
            })
            / (ts.lenf() - T::one());
        if m_reduced_chi2.is_zero() {
            ts.plateau = Some(true);
        }
        m_reduced_chi2
    });

    time_series_getter!(bool, plateau, is_plateau, |ts: &mut TimeSeries<T>| {
        if ts.m.max.is_some() && ts.m.max == ts.m.min {
            return true;
        }
        if ts.m.std2 == Some(T::zero()) {
            return true;
        }
        let m0 = ts.m.sample[0];
        // all() returns true for the empty slice, i.e. one-point time series
        Zip::from(ts.m.sample.slice(s![1..])).all(|&m| m == m0)
    });

    fn set_t_min_max_m(&mut self) {
        let (i_min, i_max) = self
            .m
            .as_slice()
            .iter()
            .position_minmax()
            .into_option()
            .expect("time series must be non-empty");
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

                let mut ds: DataSample<_> = DataSample::from(&x);
                all_close(&[ds.$method()], &desired[..], 1e-6);
                all_close(&[ds.$method()], &desired[..], 1e-6);

                let mut ds: DataSample<_> = DataSample::from(&x);
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
        let mut ts = TimeSeries::new(&t, &m, &w);
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
        let mut ts = TimeSeries::new(&t, &m, &w);
        let desired = [1.3752251301435465];
        all_close(&[ts.get_m_reduced_chi2()], &desired[..], 1e-6);
    }
}
