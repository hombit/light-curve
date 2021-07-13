use crate::float_trait::Float;
use crate::time_series::TimeSeries;

// See Press et al. sec. 15.2 Fitting Data to a Straight Line, p. 661

#[allow(clippy::many_single_char_names)]
pub fn fit_straight_line<T: Float>(
    ts: &TimeSeries<T>,
    known_errors: bool,
) -> StraightLineFitterResult<T> {
    let n = ts.lenf();
    let (s, sx, sy) = if known_errors & ts.w.is_some() {
        let (mut s, mut sx, mut sy) = (T::zero(), T::zero(), T::zero());
        for (x, y, w) in ts.tmw_iter() {
            s += w;
            sx += w * x;
            sy += w * y;
        }
        (s, sx, sy)
    } else {
        let (mut sx, mut sy) = (T::zero(), T::zero());
        for (x, y) in ts.tm_iter() {
            sx += x;
            sy += y;
        }
        (n, sx, sy)
    };
    let (stt, sty) = {
        let (mut stt, mut sty) = (T::zero(), T::zero());
        if known_errors & ts.w.is_some() {
            for (x, y, w) in ts.tmw_iter() {
                let t = x - sx / s;
                stt += w * t.powi(2);
                sty += w * t * y;
            }
        } else {
            for (x, y) in ts.tm_iter() {
                let t = x - sx / s;
                stt += t.powi(2);
                sty += t * y;
            }
        }
        (stt, sty)
    };
    let slope = sty / stt;
    let intercept = (sy - sx * slope) / s;
    let mut slope_sigma2 = stt.recip();
    let chi2: T = if known_errors & ts.w.is_some() {
        ts.tmw_iter()
            .map(|(x, y, w)| (y - intercept - slope * x).powi(2) * w)
            .sum()
    } else {
        ts.tm_iter()
            .map(|(x, y)| (y - intercept - slope * x).powi(2))
            .sum()
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
        let ts = TimeSeries::new(&x, &y, None);
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
        let ts = TimeSeries::new(&x, &y, Some(&w));
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
