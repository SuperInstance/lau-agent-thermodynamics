//! Jarzynski Equality: extracting free energy from non-equilibrium trajectories.
//!
//! ⟨e^(-βW)⟩ = e^(-βΔF)
//! The exponential average of work over many trajectories yields the free energy
//! difference, regardless of how far from equilibrium the process is.

use crate::constants::BOLTZMANN;
use serde::{Deserialize, Serialize};

/// A single non-equilibrium trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    /// Work done along this trajectory in Joules.
    pub work: f64,
    /// Probability weight (for biased sampling).
    pub weight: f64,
}

impl Trajectory {
    /// Create a new trajectory.
    pub fn new(work: f64) -> Self {
        Self { work, weight: 1.0 }
    }

    /// Create with weight.
    pub fn weighted(work: f64, weight: f64) -> Self {
        Self { work, weight }
    }

    /// Boltzmann factor: e^(-βW).
    pub fn boltzmann_factor(&self, beta: f64) -> f64 {
        (-beta * self.work).exp() * self.weight
    }
}

/// Jarzynski estimator for free energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JarzynskiEstimator {
    /// Inverse temperature β = 1/(kT).
    pub beta: f64,
    /// Collected trajectories.
    pub trajectories: Vec<Trajectory>,
}

impl JarzynskiEstimator {
    /// Create a new estimator at given temperature.
    pub fn new(temperature: f64) -> Self {
        Self {
            beta: 1.0 / (BOLTZMANN * temperature),
            trajectories: Vec::new(),
        }
    }

    /// Add a trajectory.
    pub fn add(&mut self, trajectory: Trajectory) {
        self.trajectories.push(trajectory);
    }

    /// Add multiple trajectories from work values.
    pub fn add_works(&mut self, works: &[f64]) {
        for &w in works {
            self.trajectories.push(Trajectory::new(w));
        }
    }

    /// Number of trajectories.
    pub fn n_trajectories(&self) -> usize {
        self.trajectories.len()
    }

    /// Jarzynski average: ⟨e^(-βW)⟩.
    pub fn jarzynski_average(&self) -> f64 {
        if self.trajectories.is_empty() {
            return 1.0;
        }
        let total_weight: f64 = self.trajectories.iter().map(|t| t.weight).sum();
        let weighted_sum: f64 = self
            .trajectories
            .iter()
            .map(|t| t.boltzmann_factor(self.beta))
            .sum();
        weighted_sum / total_weight
    }

    /// Estimated free energy difference: ΔF = -kT ln(⟨e^(-βW)⟩).
    pub fn free_energy_estimate(&self) -> f64 {
        let avg = self.jarzynski_average();
        if avg <= 0.0 {
            return f64::INFINITY;
        }
        -(1.0 / self.beta) * avg.ln()
    }

    /// Mean work.
    pub fn mean_work(&self) -> f64 {
        if self.trajectories.is_empty() {
            return 0.0;
        }
        let total_weight: f64 = self.trajectories.iter().map(|t| t.weight).sum();
        self.trajectories
            .iter()
            .map(|t| t.work * t.weight)
            .sum::<f64>()
            / total_weight
    }

    /// Variance of work.
    pub fn work_variance(&self) -> f64 {
        if self.trajectories.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_work();
        let total_weight: f64 = self.trajectories.iter().map(|t| t.weight).sum();
        let var = self
            .trajectories
            .iter()
            .map(|t| t.weight * (t.work - mean).powi(2))
            .sum::<f64>()
            / total_weight;
        var
    }

    /// Standard deviation of work.
    pub fn work_std(&self) -> f64 {
        self.work_variance().sqrt()
    }

    /// Check Jarzynski equality: ⟨e^(-βW)⟩ ≈ e^(-βΔF).
    pub fn verify_equality(&self, delta_f: f64, tolerance: f64) -> bool {
        let lhs = self.jarzynski_average();
        let rhs = (-self.beta * delta_f).exp();
        (lhs - rhs).abs() < tolerance
    }

    /// Cumulant expansion estimate (2nd order).
    /// ΔF ≈ ⟨W⟩ - βσ²(W)/2
    pub fn cumulant_estimate(&self) -> f64 {
        self.mean_work() - self.beta * self.work_variance() / 2.0
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.trajectories.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_trajectory_boltzmann_factor() {
        let t = Trajectory::new(1e-20);
        let beta = 1.0 / (BOLTZMANN * 300.0);
        let bf = t.boltzmann_factor(beta);
        assert!(bf > 0.0);
    }

    #[test]
    fn test_jarzynski_empty() {
        let est = JarzynskiEstimator::new(300.0);
        assert_relative_eq!(est.jarzynski_average(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_jarzynski_equilibrium() {
        // If all trajectories do the same work = ΔF, then equality holds exactly
        let mut est = JarzynskiEstimator::new(300.0);
        let delta_f = 1e-20;
        for _ in 0..100 {
            est.add(Trajectory::new(delta_f));
        }
        let avg = est.jarzynski_average();
        let expected = (-est.beta * delta_f).exp();
        assert_relative_eq!(avg, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_free_energy_estimate() {
        let mut est = JarzynskiEstimator::new(300.0);
        let w = BOLTZMANN * 300.0 * 0.01; // Small work
        est.add_works(&[w, w, w, w, w]);
        let df = est.free_energy_estimate();
        assert!(df.is_finite());
    }

    #[test]
    fn test_mean_work() {
        let mut est = JarzynskiEstimator::new(300.0);
        est.add_works(&[1e-20, 2e-20, 3e-20]);
        assert_relative_eq!(est.mean_work(), 2e-20, epsilon = 1e-25);
    }

    #[test]
    fn test_work_variance() {
        let mut est = JarzynskiEstimator::new(300.0);
        est.add_works(&[1e-20, 3e-20]);
        let var = est.work_variance();
        assert_relative_eq!(var, 1e-40, epsilon = 1e-50);
    }

    #[test]
    fn test_cumulant_estimate() {
        let mut est = JarzynskiEstimator::new(300.0);
        est.add_works(&[1e-20, 1e-20]);
        let c = est.cumulant_estimate();
        // Zero variance, so cumulant = mean
        assert_relative_eq!(c, 1e-20, epsilon = 1e-25);
    }

    #[test]
    fn test_verify_equality() {
        let mut est = JarzynskiEstimator::new(300.0);
        let delta_f = 1e-20;
        for _ in 0..1000 {
            // Gaussian around delta_f
            let noise = (rand_work() - 0.5) * 1e-22;
            est.add(Trajectory::new(delta_f + noise));
        }
        // Should approximately hold
        assert!(est.verify_equality(delta_f, 0.01));
    }

    #[test]
    fn test_reset() {
        let mut est = JarzynskiEstimator::new(300.0);
        est.add_works(&[1e-20, 2e-20]);
        est.reset();
        assert_eq!(est.n_trajectories(), 0);
    }

    #[test]
    fn test_weighted_trajectories() {
        let mut est = JarzynskiEstimator::new(300.0);
        est.add(Trajectory::weighted(1e-20, 2.0));
        est.add(Trajectory::weighted(3e-20, 2.0));
        let mean = est.mean_work();
        assert_relative_eq!(mean, 2e-20, epsilon = 1e-25);
    }

    fn rand_work() -> f64 {
        // Simple deterministic "random" for testing
        static mut SEED: u64 = 42;
        unsafe {
            SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
            (SEED >> 33) as f64 / (1u64 << 31) as f64
        }
    }
}
