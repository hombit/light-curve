use conv::prelude::*;

use crate::float_trait::Float;
use crate::statistics::Statistics;

pub struct DataSample<'a, T> {
    pub(super) sample: &'a [T],
    sorted: Vec<T>,
    min: Option<T>,
    max: Option<T>,
    mean: Option<T>,
    median: Option<T>,
    std: Option<T>,
}

macro_rules! data_sample_getter {
    ($attr: ident, $getter: ident, $method: ident) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some(self.sample.$method());
                    self.$attr.unwrap()
                },
            }
        }
    };
    ($attr: ident, $getter: ident, $method: ident, $method_sorted: ident) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some(
                        if self.sorted.is_empty() {
                            self.sample.$method()
                        } else {
                            self.sorted[..].$method_sorted()
                        }
                    );
                    self.$attr.unwrap()
                },
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
                },
            }
        }
    };
}

impl<'a, T> DataSample<'a, T>
where
    T: Float,
    [T]: Statistics<T>,
{
    pub fn new(sample: &'a [T]) -> Self {
        assert!(
            sample.len() > 1,
            "DataSample should has at least two points"
        );
        Self {
            sample,
            sorted: vec![],
            min: None,
            max: None,
            mean: None,
            median: None,
            std: None,
        }
    }

    pub(super) fn get_sorted<'c>(&'c mut self) -> &'c [T] {
        if self.sorted.is_empty() {
            self.sorted.extend(self.sample.sorted());
        }
        &self.sorted[..]
    }

    data_sample_getter!(min, get_min, minimum, min_from_sorted);
    data_sample_getter!(max, get_max, maximum, max_from_sorted);
    data_sample_getter!(mean, get_mean, mean);
    data_sample_getter!(median, get_median, |ds: &mut DataSample<'a, T>| {
        ds.get_sorted().median_from_sorted()
    });
    data_sample_getter!(std, get_std, |ds: &mut DataSample<'a, T>| {
        let mean = ds.get_mean();
        T::sqrt(
            ds.sample.iter().map(|&x| (x - mean).powi(2)).sum::<T>()
                / (ds.sample.len() - 1).value_as::<T>().unwrap(),
        )
    });

    pub fn signal_to_noise(&mut self, value: T) -> T {
        (value - self.get_mean()) / self.get_std()
    }
}

pub struct TimeSeries<'a, T> {
    pub(super) t: DataSample<'a, T>,
    pub(super) m: DataSample<'a, T>,
    pub(super) err2: Option<DataSample<'a, T>>,
    weight_sum: Option<T>,
    m_weighted_mean: Option<T>,
    m_reduced_chi2: Option<T>,
}

macro_rules! time_series_getter {
    ($attr: ident, $getter: ident, $func: expr) => {
        pub fn $getter(&mut self) -> Option<T> {
            match self.err2 {
                Some(_) => Some(match self.$attr {
                    Some(x) => x,
                    None => {
                        self.$attr = Some($func(self));
                        self.$attr.unwrap()
                    }
                }),
                None => None,
            }
        }
    };
}

impl<'a, T> TimeSeries<'a, T>
where
    T: Float,
{
    pub fn new(t: &'a [T], m: &'a [T], err2: Option<&'a [T]>) -> Self {
        assert_eq!(t.len(), m.len(), "t and m should have the same size");
        Self {
            t: DataSample::new(t),
            m: DataSample::new(m),
            err2: err2.map(|err2| {
                assert_eq!(m.len(), err2.len(), "m and err should have the same size");
                DataSample::new(err2)
            }),
            weight_sum: None,
            m_weighted_mean: None,
            m_reduced_chi2: None,
        }
    }

    pub fn lenu(&self) -> usize {
        self.t.sample.len()
    }

    pub fn lenf(&self) -> T {
        self.lenu().value_as::<T>().unwrap()
    }

    pub fn iter_time_value(&'a self) -> TimeSeriesIteratorTM<'a, T> {
        TimeSeriesIteratorTM::new(self)
    }

    pub fn iter_time_value_sqerror(&'a self) -> Option<TimeSeriesIteratorTME<'a, T>> {
        if self.err2.is_some() {
            Some(TimeSeriesIteratorTME::new(self))
        } else {
            None
        }
    }

    pub fn iter_value_sqerror(&'a self) -> Option<TimeSeriesIteratorME<'a, T>> {
        if self.err2.is_some() {
            Some(TimeSeriesIteratorME::new(self))
        } else {
            None
        }
    }

    pub fn max_by_m(&self) -> (T, T) {
        self.iter_time_value()
            .max_by(|(_t_a, m_a), (_t_b, m_b)| m_a.partial_cmp(m_b).unwrap())
            .unwrap()
    }

    time_series_getter!(weight_sum, get_weight_sum, |ts: &mut TimeSeries<T>| {
        ts.err2
            .as_ref()
            .unwrap()
            .sample
            .iter()
            .map(|&x| x.recip())
            .sum::<T>()
    });

    time_series_getter!(
        m_weighted_mean,
        get_m_weighted_mean,
        |ts: &mut TimeSeries<T>| {
            ts.iter_value_sqerror()
                .unwrap()
                .map(|(y, err2)| y / err2)
                .sum::<T>()
                / ts.get_weight_sum().unwrap()
        }
    );

    time_series_getter!(m_reduced_chi2, get_m_reduced_chi2, |ts: &mut TimeSeries<
        T,
    >| {
        let m_weighed_mean = ts.get_m_weighted_mean().unwrap();
        ts.iter_value_sqerror()
            .unwrap()
            .map(|(y, err2)| (y - m_weighed_mean).powi(2) / err2)
            .sum::<T>()
            / (ts.lenf() - T::one())
    });
}

pub struct TimeSeriesIteratorTM<'a, T: Float> {
    ts: &'a TimeSeries<'a, T>,
    position: usize,
}

impl<'a, T: Float> TimeSeriesIteratorTM<'a, T> {
    fn new(ts: &'a TimeSeries<'a, T>) -> Self {
        Self { ts, position: 0 }
    }
}

impl<'a, T: Float> Iterator for TimeSeriesIteratorTM<'a, T> {
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.ts.lenu() {
            self.position += 1;
            Some((
                self.ts.t.sample[self.position - 1],
                self.ts.m.sample[self.position - 1],
            ))
        } else {
            None
        }
    }
}

pub struct TimeSeriesIteratorTME<'a, T: Float> {
    ts: &'a TimeSeries<'a, T>,
    position: usize,
}

impl<'a, T: Float> TimeSeriesIteratorTME<'a, T> {
    fn new(ts: &'a TimeSeries<'a, T>) -> Self {
        assert!(ts.err2.is_some());
        Self { ts, position: 0 }
    }
}

impl<'a, T: Float> Iterator for TimeSeriesIteratorTME<'a, T> {
    type Item = (T, T, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.ts.lenu() {
            self.position += 1;
            Some((
                self.ts.t.sample[self.position - 1],
                self.ts.m.sample[self.position - 1],
                self.ts.err2.as_ref().unwrap().sample[self.position - 1],
            ))
        } else {
            None
        }
    }
}

pub struct TimeSeriesIteratorME<'a, T: Float> {
    ts: &'a TimeSeries<'a, T>,
    position: usize,
}

impl<'a, T: Float> TimeSeriesIteratorME<'a, T> {
    fn new(ts: &'a TimeSeries<'a, T>) -> Self {
        assert!(ts.err2.is_some());
        Self { ts, position: 0 }
    }
}

impl<'a, T: Float> Iterator for TimeSeriesIteratorME<'a, T> {
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.ts.lenu() {
            self.position += 1;
            Some((
                self.ts.m.sample[self.position - 1],
                self.ts.err2.as_ref().unwrap().sample[self.position - 1],
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
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
        let err2 = [7.7973377, 9.45495344, 3.11500361, 7.71464925, 9.30566326];
        let mut ts = TimeSeries::new(&t[..], &m[..], Some(&err2[..]));
        let desired = [16.318170478908428];
        all_close(&[ts.get_m_weighted_mean().unwrap()], &desired[..], 1e-6);
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
        let err2 = [7.7973377, 9.45495344, 3.11500361, 7.71464925, 9.30566326];
        let mut ts = TimeSeries::new(&t[..], &m[..], Some(&err2[..]));
        let desired = [1.3752251300760054];
        all_close(&[ts.get_m_reduced_chi2().unwrap()], &desired[..], 1e-6);
    }
}
