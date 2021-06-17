use crate::float_trait::Float;
use crate::periodogram::fft::*;
use crate::periodogram::freq::FreqGrid;
use crate::periodogram::power::*;
use crate::time_series::TimeSeries;

use conv::{ConvAsUtil, RoundToNearest};
use ndarray::Zip;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use thread_local::ThreadLocal;

/// "Fast" (FFT-based) periodogram executor
///
/// This algorithm spreads observer time series into uniform time grid using linear interpolation
/// and then uses FFT to obtain periodogram sums. This implementation returns estimation of
/// Lomb-Scargle periodogram that derives to the exact values while `max_freq_factor` grows.
/// Asymptotic time is $O(N \log N)$, it is faster then
/// [PeriodogramPowerDirect](crate::periodogram::PeriodogramPowerDirect) even for $N \gtrsim 10$.
/// Note that current implementation uses two-powered time grids and requires to estimate the best
/// FFT algorithm for each pair of grid size and working thread that can take several seconds,
/// especially for large grids.
///
/// The implementation is inspired by Numerical Recipes, Press et al., 1997, Section 13.8
#[derive(Clone)]
pub struct PeriodogramPowerFft<T>
where
    T: Float,
{
    fft: Arc<ThreadLocal<RefCell<Fft<T>>>>,
    arrays: Arc<ThreadLocal<RefCell<PeriodogramArraysMap<T>>>>,
}

impl<T> PeriodogramPowerFft<T>
where
    T: Float,
{
    pub fn new() -> Self {
        Self {
            fft: Arc::new(ThreadLocal::new()),
            arrays: Arc::new(ThreadLocal::new()),
        }
    }
}

impl<T> fmt::Debug for PeriodogramPowerFft<T>
where
    T: Float,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", std::any::type_name::<Self>())
    }
}

impl<T> Default for PeriodogramPowerFft<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PeriodogramPower<T> for PeriodogramPowerFft<T>
where
    T: Float,
{
    fn power(&self, freq: &FreqGrid<T>, ts: &mut TimeSeries<T>) -> Vec<T> {
        let m_std2 = ts.m.get_std().powi(2);

        if m_std2.is_zero() {
            return vec![T::zero(); freq.size.next_power_of_two()];
        }

        let grid = TimeGrid::from_freq_grid(&freq);

        let mut periodogram_arrays_map = self
            .arrays
            .get_or(|| RefCell::new(PeriodogramArraysMap::new()))
            .borrow_mut();
        let PeriodogramArrays {
            x_sch: m_for_sch,
            y_sch: sum_sin_cos_h,
            x_sc2: m_for_sc2,
            y_sc2: sum_sin_cos_2,
        } = periodogram_arrays_map.get(grid.size);

        spread_arrays_for_fft(m_for_sch, m_for_sc2, &grid, ts);

        {
            let mut fft = self.fft.get_or(|| RefCell::new(Fft::new())).borrow_mut();

            fft.fft(m_for_sch, sum_sin_cos_h).unwrap();
            fft.fft(m_for_sc2, sum_sin_cos_2).unwrap();
        }

        let ts_size = ts.lenf();

        sum_sin_cos_h
            .iter()
            .zip(sum_sin_cos_2.iter())
            .skip(1) // skip zero frequency
            .map(|(sch, sc2)| {
                let sum_cos_h = sch.get_re();
                let sum_sin_h = -sch.get_im();
                let sum_cos_2 = sc2.get_re();
                let sum_sin_2 = -sc2.get_im();

                let cos_2wtau = if T::is_zero(&sum_cos_2) && T::is_zero(&sum_sin_2) {
                    // Set tau to zero
                    T::one()
                } else {
                    sum_cos_2 / T::hypot(sum_cos_2, sum_sin_2)
                };

                let cos_wtau = T::sqrt(T::half() * (T::one() + cos_2wtau));
                let sin_wtau = T::signum(sum_sin_2) * T::sqrt(T::half() * (T::one() - cos_2wtau));

                let sum_h_cos = sum_cos_h * cos_wtau + sum_sin_h * sin_wtau;
                let sum_h_sin = sum_sin_h * cos_wtau - sum_cos_h * sin_wtau;

                let sum_cos2_wt_tau =
                    T::half() * (ts_size + sum_cos_2 * cos_wtau + sum_sin_2 * sin_wtau);
                let sum_sin2_wt_tau = ts_size - sum_cos2_wt_tau;

                let frac_cos = if T::is_zero(&sum_cos2_wt_tau) {
                    T::zero()
                } else {
                    sum_h_cos.powi(2) / sum_cos2_wt_tau
                };
                let frac_sin = if T::is_zero(&sum_sin2_wt_tau) {
                    T::zero()
                } else {
                    sum_h_sin.powi(2) / sum_sin2_wt_tau
                };

                let sum_frac = if T::is_zero(&frac_cos) {
                    T::two() * frac_sin
                } else if T::is_zero(&frac_sin) {
                    T::two() * frac_cos
                } else {
                    frac_sin + frac_cos
                };

                T::half() / m_std2 * sum_frac
            })
            .collect()
    }
}

struct PeriodogramArrays<T>
where
    T: Float,
{
    x_sch: AlignedVec<T>,
    y_sch: AlignedVec<T::FftwComplex>,
    x_sc2: AlignedVec<T>,
    y_sc2: AlignedVec<T::FftwComplex>,
}

impl<T> PeriodogramArrays<T>
where
    T: Float,
{
    fn new(n: usize) -> Self {
        let c_n = n / 2 + 1;
        Self {
            x_sch: AlignedVec::new(n),
            y_sch: AlignedVec::new(c_n),
            x_sc2: AlignedVec::new(n),
            y_sc2: AlignedVec::new(c_n),
        }
    }
}

impl<T> fmt::Debug for PeriodogramArrays<T>
where
    T: Float,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PeriodogramArrays(n = {})", self.x_sch.len())
    }
}

#[derive(Debug)]
struct PeriodogramArraysMap<T>
where
    T: Float,
{
    arrays: HashMap<usize, PeriodogramArrays<T>>,
}

impl<T> PeriodogramArraysMap<T>
where
    T: Float,
{
    fn new() -> Self {
        Self {
            arrays: HashMap::new(),
        }
    }

    fn get(&mut self, n: usize) -> &mut PeriodogramArrays<T> {
        self.arrays
            .entry(n)
            .or_insert_with(|| PeriodogramArrays::new(n))
    }
}

struct TimeGrid<T> {
    dt: T,
    size: usize,
}

impl<T: Float> TimeGrid<T> {
    fn from_freq_grid(freq: &FreqGrid<T>) -> Self {
        let size = freq.size.next_power_of_two() << 1;
        Self {
            dt: T::two() * T::PI() / (freq.step * size.approx().unwrap()),
            size,
        }
    }

    #[cfg(test)]
    fn freq_grid(&self) -> FreqGrid<T> {
        FreqGrid {
            step: T::two() * T::PI() / (self.dt * self.size.approx().unwrap()),
            size: self.size >> 1,
        }
    }
}

fn spread<T: Float>(v: &mut [T], x: T, y: T) {
    let x_lo = x.floor();
    let x_hi = x.ceil();
    let i_lo: usize = x_lo.approx_by::<RoundToNearest>().unwrap() % v.len();
    let i_hi: usize = x_hi.approx_by::<RoundToNearest>().unwrap() % v.len();

    if i_lo == i_hi {
        v[i_lo] += y;
        return;
    }

    v[i_lo] += (x_hi - x) * y;
    v[i_hi] += (x - x_lo) * y;
}

fn spread_arrays_for_fft<T: Float>(
    x_sch: &mut [T],
    x_sc2: &mut [T],
    grid: &TimeGrid<T>,
    ts: &mut TimeSeries<T>,
) {
    x_sch.fill(T::zero());
    x_sc2.fill(T::zero());

    let t0 = ts.t.sample[0];
    let m_mean = ts.m.get_mean();

    Zip::from(&ts.t.sample)
        .and(&ts.m.sample)
        .for_each(|&t, &m| {
            let x = (t - t0) / grid.dt;
            spread(x_sch, x, m - m_mean);
            let double_x = T::two() * x;
            spread(x_sc2, double_x, T::one());
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodogram::freq::{AverageNyquistFreq, NyquistFreq};
    use light_curve_common::{all_close, linspace};
    use rand::prelude::*;

    #[test]
    fn time_grid_from_freq_grid_power_of_two_size() {
        const FREQ: FreqGrid<f32> = FreqGrid {
            size: 1 << 4,
            step: 3.0,
        };
        let time_grid = TimeGrid::from_freq_grid(&FREQ);
        let freq_grid = time_grid.freq_grid();
        assert_eq!(freq_grid.size, FREQ.size);
        assert!(f32::abs(freq_grid.step - FREQ.step) < 1e-10);
    }

    #[test]
    fn time_grid_from_freq_grid_not_power_of_two_size() {
        const FREQ: FreqGrid<f32> = FreqGrid {
            size: (1 << 4) + 1,
            step: 3.0,
        };
        let time_grid = TimeGrid::from_freq_grid(&FREQ);
        let freq_grid = time_grid.freq_grid();
        assert!(freq_grid.size >= FREQ.size);
        assert!(f32::abs(freq_grid.step - FREQ.step) < 1e-10);
    }

    #[test]
    fn spread_arrays_for_fft_one_to_one() {
        const N: usize = 32;

        let mut rng = StdRng::seed_from_u64(0);

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<f64> = (0..N).map(|_| rng.gen()).collect();
        let mut ts = TimeSeries::new_without_weight(&t[..], &m[..]);

        let nyquist: Box<dyn NyquistFreq<f64>> = Box::new(AverageNyquistFreq);
        let freq_grid = FreqGrid::from_t(&t, 1.0, 1.0, &nyquist);
        let time_grid = TimeGrid::from_freq_grid(&freq_grid);

        let (mh, m2) = {
            let mut mh = vec![0.0; time_grid.size];
            let mut m2 = vec![0.0; time_grid.size];
            spread_arrays_for_fft(&mut mh, &mut m2, &time_grid, &mut ts);
            (mh, m2)
        };

        let desired_mh: Vec<_> = m.iter().map(|&x| x - ts.m.get_mean()).collect();
        all_close(&mh, &desired_mh, 1e-10);

        let desired_m2: Vec<_> = (0..N).map(|i| ((i + 1) % 2 * 2) as f64).collect();
        assert_eq!(&m2[..], &desired_m2[..]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn spread_arrays_for_fft_one_to_one_resolution() {
        const N: usize = 8;
        const RESOLUTION: usize = 4;

        let mut rng = StdRng::seed_from_u64(0);

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<f64> = (0..N).map(|_| rng.gen()).collect();
        let mut ts = TimeSeries::new_without_weight(&t[..], &m[..]);

        let nyquist: Box<dyn NyquistFreq<f64>> = Box::new(AverageNyquistFreq);
        let freq_grid = FreqGrid::from_t(&t, RESOLUTION as f32, 1.0, &nyquist);
        let time_grid = TimeGrid::from_freq_grid(&freq_grid);
        let (mh, m2) = {
            let mut mh = vec![0.0; time_grid.size];
            let mut m2 = vec![0.0; time_grid.size];
            spread_arrays_for_fft(&mut mh, &mut m2, &time_grid, &mut ts);
            (mh, m2)
        };

        let desired_mh: Vec<_> = m.iter().map(|&x| x - ts.m.get_mean()).collect();
        all_close(&mh[..N], &desired_mh, 1e-10);
        assert_eq!(&mh[N..], &[0.0; (RESOLUTION - 1) * N]);

        let desired_m2: Vec<_> = (0..2 * N).map(|i| ((i + 1) % 2) as f64).collect();
        assert_eq!(&m2[..2 * N], &desired_m2[..]);
        assert_eq!(&m2[2 * N..], &[0.0; (RESOLUTION - 2) * N]);
    }
}
