use crate::float_trait::Float;
use crate::periodogram::freq::FreqGrid;
use crate::periodogram::power::*;
use crate::periodogram::recurrent_sin_cos::*;
use crate::time_series::TimeSeries;

/// Direct periodogram executor
///
/// This algorithm evaluate direct calculation of Lomb-Scargle periodogram. Asymptotic time is
/// $O(N^2)$, so it is recommended to use
/// [PeriodogramPowerFft](crate::periodogram::PeriodogramPowerFft) instead
///
/// The implementation is inspired by Numerical Recipes, Press et al., 1997, Section 13.8
#[derive(Debug, Clone)]
pub struct PeriodogramPowerDirect;

impl<T> PeriodogramPower<T> for PeriodogramPowerDirect
where
    T: Float,
{
    fn power(&self, freq: &FreqGrid<T>, ts: &mut TimeSeries<T>) -> Vec<T> {
        let m_mean = ts.m.get_mean();

        let sin_cos_omega_tau = SinCosOmegaTau::new(freq.step, &ts.t.sample.as_slice().unwrap());
        let mut sin_cos_omega_x: Vec<_> =
            ts.t.sample
                .iter()
                .map(|&x| RecurrentSinCos::new(freq.step * x))
                .collect();

        sin_cos_omega_tau
            .take(freq.size)
            .map(|(sin_omega_tau, cos_omega_tau)| {
                let mut sum_m_sin = T::zero();
                let mut sum_m_cos = T::zero();
                let mut sum_sin2 = T::zero();
                for (s_c_omega_x, &y) in sin_cos_omega_x.iter_mut().zip(ts.m.sample.iter()) {
                    let (sin_omega_x, cos_omega_x) = s_c_omega_x.next().unwrap();
                    // sine and cosine of omega * (x - tau)
                    let sin = sin_omega_x * cos_omega_tau - cos_omega_x * sin_omega_tau;
                    let cos = cos_omega_x * cos_omega_tau + sin_omega_x * sin_omega_tau;
                    sum_m_sin += (y - m_mean) * sin;
                    sum_m_cos += (y - m_mean) * cos;
                    sum_sin2 += sin.powi(2);
                }
                let sum_cos2 = ts.lenf() - sum_sin2;

                if (sum_m_sin.is_zero() & sum_sin2.is_zero())
                    | (sum_m_cos.is_zero() & sum_cos2.is_zero())
                    | ts.m.get_std().is_zero()
                {
                    T::zero()
                } else {
                    T::half() * (sum_m_sin.powi(2) / sum_sin2 + sum_m_cos.powi(2) / sum_cos2)
                        / ts.m.get_std().powi(2)
                }
            })
            .collect()
    }
}

#[derive(Clone)]
struct PeriodogramSums<T> {
    m_sin: T,
    m_cos: T,
    sin2: T,
}

impl<T: Float> Default for PeriodogramSums<T> {
    fn default() -> Self {
        Self {
            m_sin: T::zero(),
            m_cos: T::zero(),
            sin2: T::zero(),
        }
    }
}

struct SinCosOmegaTau<T> {
    sin_cos_2omega_x: Vec<RecurrentSinCos<T>>,
}

impl<T: Float> SinCosOmegaTau<T> {
    fn new(freq0: T, t: &[T]) -> Self {
        let sin_cos_2omega_x = t
            .iter()
            .map(|&x| RecurrentSinCos::new(T::two() * freq0 * x))
            .collect();
        Self { sin_cos_2omega_x }
    }
}

impl<T: Float> Iterator for SinCosOmegaTau<T> {
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        let mut sum_sin = T::zero();
        let mut sum_cos = T::zero();
        for s_c in self.sin_cos_2omega_x.iter_mut() {
            let (sin, cos) = s_c.next().unwrap();
            sum_sin += sin;
            sum_cos += cos;
        }
        let cos2 = sum_cos / T::hypot(sum_sin, sum_cos);
        let sin = T::signum(sum_sin) * T::sqrt(T::half() * (T::one() - cos2));
        let cos = T::sqrt(T::half() * (T::one() + cos2));
        Some((sin, cos))
    }
}
