use crate::float_trait::Float;
use crate::time_series::TimeSeries;

use ndarray::Zip;

// See Press et al. sec. 15.2 Fitting Data to a Straight Line, p. 661

#[allow(clippy::many_single_char_names)]
pub fn fit_straight_line<T: Float>(
    ts: &TimeSeries<T>,
    known_errors: bool,
) -> StraightLineFitterResult<T> {
    let n = ts.lenf();
    let (s, sx, sy) = if known_errors {
        Zip::from(&ts.t.sample)
            .and(&ts.m.sample)
            .and(&ts.w.sample)
            .fold(
                (T::zero(), T::zero(), T::zero()),
                |(s, sx, sy), &x, &y, &w| (s + w, sx + w * x, sy + w * y),
            )
    } else {
        let (sx, sy) = Zip::from(&ts.t.sample)
            .and(&ts.m.sample)
            .fold((T::zero(), T::zero()), |(sx, sy), &x, &y| (sx + x, sy + y));
        (n, sx, sy)
    };
    let (stt, sty) = if known_errors {
        Zip::from(&ts.t.sample)
            .and(&ts.m.sample)
            .and(&ts.w.sample)
            .fold((T::zero(), T::zero()), |(stt, sty), &x, &y, &w| {
                let t = x - sx / s;
                (stt + w * t.powi(2), sty + w * t * y)
            })
    } else {
        Zip::from(&ts.t.sample).and(&ts.m.sample).fold(
            (T::zero(), T::zero()),
            |(stt, sty), &x, &y| {
                let t = x - sx / s;
                (stt + t.powi(2), sty + t * y)
            },
        )
    };
    let slope = sty / stt;
    let intercept = (sy - sx * slope) / s;
    let mut slope_sigma2 = stt.recip();
    let chi2: T = if known_errors {
        Zip::from(&ts.t.sample)
            .and(&ts.m.sample)
            .and(&ts.w.sample)
            .fold(T::zero(), |chi2, &x, &y, &w| {
                chi2 + (y - intercept - slope * x).powi(2) * w
            })
    } else {
        Zip::from(&ts.t.sample)
            .and(&ts.m.sample)
            .fold(T::zero(), |chi2, &x, &y| {
                chi2 + (y - intercept - slope * x).powi(2)
            })
    };
    let reduced_chi2 = chi2 / (n - T::two());
    if !known_errors {
        slope_sigma2 *= reduced_chi2;
    }
    StraightLineFitterResult {
        slope,
        slope_sigma2,
        reduced_chi2,
    }
}

pub struct StraightLineFitterResult<T> {
    pub slope: T,
    pub slope_sigma2: T,
    pub reduced_chi2: T,
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use light_curve_common::all_close;

    #[test]
    fn straight_line_fitter() {
        let x = [0.5, 1.5, 2.5, 5.0, 7.0, 16.0];
        let y = [-1.0, 3.0, 2.0, 6.0, 10.0, 25.0];
        let ts = TimeSeries::new_without_weight(&x, &y);
        // scipy.optimize.curve_fit(absolute_sigma=False)
        let desired_slope = 1.63021767;
        let desired_slope_sigma2 = 0.0078127;
        let desired_reduced_chi2 = 1.271190781049937;
        let result = fit_straight_line(&ts, false);
        all_close(&[result.slope], &[desired_slope], 1e-6);
        all_close(&[result.slope_sigma2], &[desired_slope_sigma2], 1e-6);
        all_close(&[result.reduced_chi2], &[desired_reduced_chi2], 1e-6);
    }

    #[test]
    fn noisy_straight_line_fitter() {
        let x = [0.5, 1.5, 2.5, 5.0, 7.0, 16.0];
        let y = [-1.0, 3.0, 2.0, 6.0, 10.0, 25.0];
        let w = [2.0, 1.0, 3.0, 10.0, 1.0, 0.4];
        let ts = TimeSeries::new(&x, &y, &w);
        // curve_fit(lambda x, a, b: a + b*x, xdata=x, ydata=y, sigma=w**-0.5, absolute_sigma=True)
        let desired_slope = 1.6023644;
        let desired_slope_sigma2 = 0.00882845;
        let desired_reduced_chi2 = 1.7927152569891913;
        let result = fit_straight_line(&ts, true);
        all_close(&[result.slope], &[desired_slope], 1e-6);
        all_close(&[result.slope_sigma2], &[desired_slope_sigma2], 1e-6);
        all_close(&[result.reduced_chi2], &[desired_reduced_chi2], 1e-6);
    }
}
