use std::iter::Sum;
use std::ops::{Add, Mul, Sub};

use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead;
use argmin::solver::quasinewton::BFGS;
use serde::{Deserialize, Serialize};

use crate::float_trait::Float;
use crate::time_series::{DataSample, TimeSeries};

fn fit_car_neldermead(car: CAR<f64>) -> Theta<f64> {
    let duration = car.t[car.length - 1] - car.t[0];
    let initial_tau = duration / f64::sqrt(car.length as f64);
    let mut x = DataSample::new(&car.x[..]);
    let initial_b = x.get_mean() / initial_tau;
    let initial_sigma2 = 2.0 * x.get_std().powi(2) * (car.length as f64 - 1.0) / duration;
    let solver = NelderMead::new()
        .with_initial_params(vec![
            Theta {
                b: initial_b,
                sigma2: initial_sigma2,
                tau: initial_tau,
            }
            .to_vec(),
            Theta {
                b: initial_b - x.get_std() / duration,
                sigma2: 0.5 * initial_sigma2,
                tau: 2.0 * initial_tau,
            }
            .to_vec(),
            Theta {
                b: initial_b + x.get_std() / duration,
                sigma2: 2.0 * initial_sigma2,
                tau: 0.5 * initial_tau,
            }
            .to_vec(),
        ])
        .sd_tolerance(1e-4);
    let res = Executor::new(car, solver, Theta::default().to_vec())
        .max_iters(100)
        .run()
        .unwrap();
    Theta::from_slice(&res.state.param[..])
}

fn fit_car_bfgs(car: CAR<f64>) -> Theta<f64> {
    let duration = car.t[car.length - 1] - car.t[0];
    let initial_tau = duration / f64::sqrt(car.length as f64);
    let mut x = DataSample::new(&car.x[..]);
    let initial_b = x.get_mean() / initial_tau;
    let initial_sigma2 = 2.0 * x.get_std().powi(2) * (car.length as f64 - 1.0) / duration;
    let linesearch = MoreThuenteLineSearch::new().c(1e-4, 0.9).unwrap();
    let solver = BFGS::new(
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ],
        linesearch,
    );
    let res = Executor::new(
        car,
        solver,
        Theta {
            b: initial_b,
            sigma2: initial_sigma2,
            tau: initial_tau,
        }
        .to_vec(),
    )
    .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
    .max_iters(100)
    .run()
    .unwrap();
    Theta::from_slice(&res.state.param[..])
}

#[derive(Clone, Serialize)]
struct CAR<T> {
    t: Vec<T>,
    x: Vec<T>,
    delta2: Vec<T>,
    length: usize,
}

impl<T: Float> CAR<T> {
    fn new(t: Vec<T>, x: Vec<T>, delta2: Vec<T>) -> Self {
        assert_eq!(t.len(), x.len());
        assert_eq!(x.len(), delta2.len());
        let lenu = t.len();
        assert!(lenu > 1);
        Self {
            t,
            x,
            delta2,
            length: lenu,
        }
    }

    fn from_time_series(ts: &TimeSeries<T>) -> Self {
        assert!(ts.err2.is_some(), "Errors should be specified for CAR");
        Self::new(
            ts.t.sample.to_vec(),
            ts.m.sample.to_vec(),
            ts.err2.as_ref().unwrap().sample.to_vec(),
        )
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (T, T, T)> + 'a {
        return CARIterator::new(self);
    }
}

impl<T: Float> ArgminOp for CAR<T> {
    type Param = Vec<T>;
    type Output = T;
    type Hessian = Vec<Vec<T>>;
    type Jacobian = ();

    /// Function to minimize: -2 logL + const
    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let theta = Theta::from_slice(&param[..]);
        let mut it = self
            .iter()
            // Iterator over observations
            .map(|(t, x, delta2)| Observation { t, x, delta2 });
        let obs = it.next().unwrap();
        let coeffs = CoeffsForTarget::new(obs, &theta);
        Ok(coeffs.target_function_part()
            + it.scan(coeffs, |coeffs, obs| {
                coeffs.step(obs, &theta);
                Some(coeffs.target_function_part())
            })
            .sum())
    }

    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        let theta = Theta::from_slice(&param[..]);
        let mut it = self
            .iter()
            // Iterator over observations
            .map(|(t, x, delta2)| Observation { t, x, delta2 });
        let obs = it.next().unwrap();
        let coeffs = CoeffsForGradient::new(obs, &theta);
        Ok((coeffs.target_gradient_part()
            + it.scan(coeffs, |coeffs, obs| {
                coeffs.step(obs, &theta);
                Some(coeffs.target_gradient_part())
            })
            .sum::<Theta<T>>())
        .to_vec())
    }
}

struct CARIterator<'a, T> {
    car: &'a CAR<T>,
    position: usize,
}

impl<'a, T> CARIterator<'a, T> {
    fn new(car: &'a CAR<T>) -> Self {
        Self { car, position: 0 }
    }
}

impl<'a, T: Float> Iterator for CARIterator<'a, T> {
    type Item = (T, T, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.car.length {
            self.position += 1;
            Some((
                self.car.t[self.position - 1],
                self.car.x[self.position - 1],
                self.car.delta2[self.position - 1],
            ))
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Theta<T> {
    b: T,
    sigma2: T,
    tau: T,
}

impl<T: Float> Theta<T> {
    fn from_slice(v: &[T]) -> Self {
        assert_eq!(v.len(), 3);
        Self {
            b: v[0],
            sigma2: v[1],
            tau: v[2],
        }
    }

    fn to_vec(&self) -> Vec<T> {
        vec![self.b, self.sigma2, self.tau]
    }
}

impl<T: Float> Default for Theta<T> {
    fn default() -> Self {
        Self {
            b: T::zero(),
            sigma2: T::zero(),
            tau: T::zero(),
        }
    }
}

impl<T: Float> Add<Theta<T>> for Theta<T> {
    type Output = Self;

    fn add(self, rhs: Theta<T>) -> Self::Output {
        Self {
            b: self.b + rhs.b,
            sigma2: self.sigma2 + rhs.sigma2,
            tau: self.tau + rhs.tau,
        }
    }
}

impl<T: Float> Sub<Theta<T>> for Theta<T> {
    type Output = Self;

    fn sub(self, rhs: Theta<T>) -> Self::Output {
        Self {
            b: self.b - rhs.b,
            sigma2: self.sigma2 - rhs.sigma2,
            tau: self.tau - rhs.tau,
        }
    }
}

impl<T: Float> Mul<T> for Theta<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            b: self.b * rhs,
            sigma2: self.sigma2 * rhs,
            tau: self.tau * rhs,
        }
    }
}

impl<T: Float> Sum for Theta<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Theta::default(), |a, b| a + b)
    }
}

#[derive(Clone)]
struct Observation<T> {
    t: T,
    x: T,
    delta2: T,
}

#[derive(Clone)]
struct CoeffsForTarget<T> {
    obs: Observation<T>,
    star: T,
    hat: T,
    a: T,
    omega: T,
    omega_0: T,
}

impl<T: Float> CoeffsForTarget<T> {
    // Coefficients for the first observation
    fn new(obs: Observation<T>, theta: &Theta<T>) -> Self {
        let star = obs.x - theta.b * theta.tau;
        let hat = T::zero();
        let omega = T::half() * theta.tau * theta.sigma2;
        Self {
            obs,
            star,
            hat,
            a: T::one(), // isn't used
            omega,
            omega_0: omega,
        }
    }

    fn step(&mut self, obs: Observation<T>, theta: &Theta<T>) {
        let star = obs.x - theta.b * theta.tau;
        let a = T::exp(-(obs.t - self.obs.t) / theta.tau);
        let omega = self.omega_0 * (T::one() - a.powi(2))
            + a.powi(2) * self.omega * (T::one() - self.omega / (self.omega + obs.delta2));
        let hat =
            a * self.hat + a * self.omega / (self.omega + obs.delta2) * (self.star + self.hat);
        *self = Self {
            obs,
            star,
            hat,
            a,
            omega,
            omega_0: self.omega_0,
        };
    }

    fn target_function_part(&self) -> T {
        let omega_delta2 = self.omega + self.obs.delta2;
        T::ln(omega_delta2) + (self.hat - self.star).powi(2) / (omega_delta2)
    }
}

struct CoeffsForGradient<T> {
    coeff: CoeffsForTarget<T>,
    d_hat: Theta<T>,
    d_star: Theta<T>,
    d_omega: Theta<T>,
    d_omega_0: Theta<T>,
}

impl<T: Float> CoeffsForGradient<T> {
    fn new(obs: Observation<T>, theta: &Theta<T>) -> Self {
        let d_omega_0 = Theta {
            b: T::zero(),
            sigma2: T::half() * theta.tau,
            tau: T::half() * theta.sigma2,
        };
        Self {
            coeff: CoeffsForTarget::new(obs, theta),
            d_hat: Theta {
                b: T::zero(),
                sigma2: T::zero(),
                tau: T::zero(),
            },
            d_star: Theta {
                b: -theta.tau,
                sigma2: T::zero(),
                tau: -theta.b,
            },
            d_omega: d_omega_0.clone(),
            d_omega_0,
        }
    }

    fn step(&mut self, obs: Observation<T>, theta: &Theta<T>) {
        let prev_coeff = self.coeff.clone();
        self.coeff.step(obs, theta);
        let da = Theta {
            b: T::zero(),
            sigma2: T::zero(),
            tau: (self.coeff.obs.t - prev_coeff.obs.t) * self.coeff.a / theta.tau.powi(2),
        };
        let omega_delta2_prev = prev_coeff.omega + prev_coeff.obs.delta2;
        let omega_over_omega_delta2_prev = prev_coeff.omega / omega_delta2_prev;
        let delta2_over_omega_delta2_prev = prev_coeff.obs.delta2 / omega_delta2_prev;
        let d_omega = self.d_omega_0.clone() * (T::one() - self.coeff.a.powi(2))
            + da.clone()
                * T::two()
                * self.coeff.a
                * (-self.coeff.omega_0
                    + prev_coeff.omega * (T::one() - omega_over_omega_delta2_prev))
            + self.d_omega.clone() * (self.coeff.a * delta2_over_omega_delta2_prev).powi(2);
        let d_hat = da.clone()
            * (prev_coeff.hat + omega_over_omega_delta2_prev * (prev_coeff.star + prev_coeff.hat))
            + self.d_hat.clone() * self.coeff.a * (T::one() + omega_over_omega_delta2_prev)
            + self.d_omega.clone()
                * self.coeff.a
                * delta2_over_omega_delta2_prev
                * (prev_coeff.star + prev_coeff.hat)
            + self.d_star.clone() * self.coeff.a * omega_over_omega_delta2_prev;

        self.d_hat = d_hat;
        self.d_omega = d_omega;
    }

    fn target_gradient_part(&self) -> Theta<T> {
        let omega_delta2 = self.coeff.omega + self.coeff.obs.delta2;
        let hat_min_star = self.coeff.hat - self.coeff.star;
        self.d_omega.clone() * ((T::one() - (hat_min_star / omega_delta2).powi(2)) / omega_delta2)
            + (self.d_hat.clone() - self.d_star.clone()) * (T::two() * hat_min_star / omega_delta2)
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;
    use rand_distr::Normal;
    use rand_pcg::Pcg64Mcg;

    use super::*;
    use light_curve_common::linspace;

    fn generate_process(
        x0: f64,
        t: &[f64],
        delta2: &[f64],
        theta: &Theta<f64>,
        lag: usize,
    ) -> Vec<f64> {
        assert!(lag > 0);
        assert_eq!(t.len(), delta2.len());
        let mut process_noise = Normal::new(0.0, theta.sigma2.sqrt())
            .unwrap()
            .sample_iter(Pcg64Mcg::seed_from_u64(0));
        let mut standard_noise = Normal::new(0.0, 1.0)
            .unwrap()
            .sample_iter(Pcg64Mcg::seed_from_u64(1));
        let long_dt_it =
            (1..t.len()).flat_map(|i| (0..lag).map(move |_| (t[i] - t[i - 1]) / (lag as f64)));
        [0.0]
            // Append zero dt in front of long_dt
            .iter()
            .cloned()
            .chain(long_dt_it)
            // produce signal
            .scan(x0, |x, dt| {
                *x += (theta.b - *x / theta.tau) * dt + process_noise.next().unwrap() * dt.sqrt();
                Some(*x)
            })
            // get signal only at original time points
            .step_by(lag)
            // add error
            .zip(delta2.iter())
            .map(|(x, &delta2)| x + delta2.sqrt() * standard_noise.next().unwrap())
            .collect()
    }

    #[test]
    fn car() {
        let t = linspace(0.0, 99.0, 20);
        let delta2: Vec<_> = vec![0.0; 20];
        let desired_b = 0.0;
        let desired_sigma2 = 1e-4 * (t[t.len() - 1] - t[0]);
        let desired_tau = 5.0;
        let x = generate_process(
            0.0,
            &t[..],
            &delta2[..],
            &Theta {
                b: desired_b,
                sigma2: desired_sigma2,
                tau: desired_tau,
            },
            100,
        );
        println!("{:?}", x);
        let car = CAR::new(t, x, delta2);
        let theta = fit_car_bfgs(car);
        println!("{:?}", theta);
        assert!(false);
    }
}
