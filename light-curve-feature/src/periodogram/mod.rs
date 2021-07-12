use crate::float_trait::Float;
use crate::time_series::TimeSeries;

use conv::ConvAsUtil;

mod fft;
pub use fft::FftwFloat;

mod freq;
pub use freq::FreqGrid;
pub use freq::{AverageNyquistFreq, MedianNyquistFreq, NyquistFreq, QuantileNyquistFreq};

mod power;
pub use power::PeriodogramPower;

mod power_fft;
pub use power_fft::PeriodogramPowerFft;

mod power_direct;
pub use power_direct::PeriodogramPowerDirect;

pub mod recurrent_sin_cos;

/// Lamb-Scargle periodogram calculator on uniform frequency grid
///
/// Frequencies are given by $\omega = \{\min\omega..\max\omega\}$: $N_\omega$ nodes with step
/// $\Delta\omega$: $\min\omega = \Delta\omega$, $\max\omega = N_\omega \Delta\omega$.
///
/// Parameters of the grid can be derived from time series properties: typical time interval
/// $\delta t$ and duration of observation. The idea is to set maximum frequency to Nyquist
/// value $\pi / \Delta t$ and minimum frequency to $2\pi / \mathrm{duration}$, while `nyquist` and
/// `resolution` factors are used to widen this interval:
/// $$
/// \max\omega = N_\omega \Delta\omega = \frac{\pi}{\Delta t},
/// $$
/// $$
/// \min\omega = \Delta\omega = \frac{2\pi}{\mathrm{resolution} \times \mathrm{duration}}.
/// $$
pub struct Periodogram<T> {
    freq_grid: FreqGrid<T>,
    periodogram_power: Box<dyn PeriodogramPower<T>>,
}

impl<T> Periodogram<T>
where
    T: Float,
{
    pub fn new(periodogram_power: Box<dyn PeriodogramPower<T>>, freq_grid: FreqGrid<T>) -> Self {
        assert!(freq_grid.step.is_sign_positive() && freq_grid.step.is_finite());
        assert!(freq_grid.size > 0);

        Self {
            freq_grid,
            periodogram_power,
        }
    }

    pub fn set_periodogram_power(
        &mut self,
        periodogram_power: Box<dyn PeriodogramPower<T>>,
    ) -> &mut Self {
        self.periodogram_power = periodogram_power;
        self
    }

    #[allow(clippy::borrowed_box)] // https://github.com/rust-lang/rust-clippy/issues/4305
    pub fn from_t(
        periodogram_power: Box<dyn PeriodogramPower<T>>,
        t: &[T],
        resolution: f32,
        max_freq_factor: f32,
        nyquist: &Box<dyn NyquistFreq<T>>,
    ) -> Self {
        Self::new(
            periodogram_power,
            FreqGrid::from_t(t, resolution, max_freq_factor, nyquist),
        )
    }

    pub fn freq(&self, i: usize) -> T {
        self.freq_grid.step * (i + 1).approx().unwrap()
    }

    pub fn power(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        self.periodogram_power.power(&self.freq_grid, ts)
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    use crate::peak_indices::peak_indices_reverse_sorted;
    use crate::sorted_array::SortedArray;

    use light_curve_common::{all_close, linspace};
    use rand::prelude::*;

    #[test]
    fn compr_direct_with_scipy() {
        const OMEGA_SIN: f64 = 0.07;
        const N: usize = 100;
        let t = linspace(0.0, 99.0, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA_SIN * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let mut periodogram = Periodogram::new(
            Box::new(PeriodogramPowerDirect),
            FreqGrid {
                step: OMEGA_SIN,
                size: 1,
            },
        );
        periodogram.set_periodogram_power(Box::new(PeriodogramPowerDirect));
        all_close(
            &[periodogram.power(&mut ts)[0] * 2.0 / (N as f64 - 1.0)],
            &[1.0],
            1.0 / (N as f64),
        );

        // import numpy as np
        // from scipy.signal import lombscargle
        //
        // t = np.arange(100)
        // m = np.sin(0.07 * t)
        // y = (m - m.mean()) / m.std(ddof=1)
        // freq = np.linspace(0.01, 0.05, 5)
        // print(lombscargle(t, y, freq, precenter=True, normalize=False))

        let freq_grid = FreqGrid {
            step: 0.01,
            size: 5,
        };
        let periodogram = Periodogram::new(Box::new(PeriodogramPowerDirect), freq_grid.clone());
        all_close(
            &linspace(
                freq_grid.step,
                freq_grid.step * freq_grid.size as f64,
                freq_grid.size,
            ),
            &(0..freq_grid.size)
                .map(|i| periodogram.freq(i))
                .collect::<Vec<_>>(),
            1e-12,
        );
        let desired = [
            16.99018018,
            18.57722516,
            21.96049738,
            28.15056806,
            36.66519435,
        ];
        let actual = periodogram.power(&mut ts);
        all_close(&actual[..], &desired[..], 1e-6);
    }

    #[test]
    fn direct_vs_fft_one_to_one() {
        const OMEGA: f64 = 0.472;
        const N: usize = 64;
        const RESOLUTION: f32 = 1.0;
        const MAX_FREQ_FACTOR: f32 = 1.0;

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let nyquist: Box<dyn NyquistFreq<f64>> = Box::new(AverageNyquistFreq);

        let direct = Periodogram::from_t(
            Box::new(PeriodogramPowerDirect),
            &t,
            RESOLUTION,
            MAX_FREQ_FACTOR,
            &nyquist,
        )
        .power(&mut ts);
        let fft = Periodogram::from_t(
            Box::new(PeriodogramPowerFft::new()),
            &t,
            RESOLUTION,
            MAX_FREQ_FACTOR,
            &nyquist,
        )
        .power(&mut ts);
        all_close(&fft[..direct.len() - 1], &direct[..direct.len() - 1], 1e-8);
    }

    #[test]
    fn direct_vs_fft_uniform_sin_cos() {
        const OMEGA1: f64 = 0.472;
        const OMEGA2: f64 = 1.222;
        const AMPLITUDE2: f64 = 2.0;
        const N: usize = 100;
        const RESOLUTION: f32 = 4.0;
        const MAX_FREQ_FACTOR: f32 = 1.0;

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<_> = t
            .iter()
            .map(|&x| f64::sin(OMEGA1 * x) + AMPLITUDE2 * f64::cos(OMEGA2 * x))
            .collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let nyquist: Box<dyn NyquistFreq<f64>> = Box::new(AverageNyquistFreq);

        let direct = Periodogram::from_t(
            Box::new(PeriodogramPowerDirect),
            &t,
            RESOLUTION,
            MAX_FREQ_FACTOR,
            &nyquist,
        )
        .power(&mut ts);
        let fft = Periodogram::from_t(
            Box::new(PeriodogramPowerFft::new()),
            &t,
            RESOLUTION,
            MAX_FREQ_FACTOR,
            &nyquist,
        )
        .power(&mut ts);

        assert_eq!(
            peak_indices_reverse_sorted(&fft)[..2],
            peak_indices_reverse_sorted(&direct)[..2]
        );
    }

    #[test]
    fn direct_vs_fft_unevenly_sin_cos() {
        const OMEGA1: f64 = 0.472;
        const OMEGA2: f64 = 1.222;
        const AMPLITUDE2: f64 = 2.0;
        const NOISE_AMPLITUDE: f64 = 1.0;
        const N: usize = 100;
        const RESOLUTION: f32 = 6.0;
        const MAX_FREQ_FACTOR: f32 = 1.0;

        let mut rng = StdRng::seed_from_u64(0);
        let t: SortedArray<_> = (0..N)
            .map(|_| rng.gen::<f64>() * (N - 1) as f64)
            .collect::<Vec<_>>()
            .into();
        let m: Vec<_> = t
            .iter()
            .map(|&x| {
                f64::sin(OMEGA1 * x)
                    + AMPLITUDE2 * f64::cos(OMEGA2 * x)
                    + NOISE_AMPLITUDE * rng.gen::<f64>()
            })
            .collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let nyquist: Box<dyn NyquistFreq<f64>> = Box::new(MedianNyquistFreq);

        let direct = Periodogram::from_t(
            Box::new(PeriodogramPowerDirect),
            &t,
            RESOLUTION,
            MAX_FREQ_FACTOR,
            &nyquist,
        )
        .power(&mut ts);
        let fft = Periodogram::from_t(
            Box::new(PeriodogramPowerFft::new()),
            &t,
            RESOLUTION,
            MAX_FREQ_FACTOR,
            &nyquist,
        )
        .power(&mut ts);

        assert_eq!(
            peak_indices_reverse_sorted(&fft)[..2],
            peak_indices_reverse_sorted(&direct)[..2]
        );
    }
}
