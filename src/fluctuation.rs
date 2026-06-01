//! Fluctuation Theorem: probability of entropy decrease in small agents.
//!
//! P(σ) / P(-σ) = e^(σ τ)
//! The ratio of probability of forward to backward entropy production
//! rate σ over time τ is exponential. For small systems, entropy can
//! temporarily decrease, but exponentially rarely.

use crate::constants::BOLTZMANN;
use serde::{Deserialize, Serialize};

/// Fluctuation theorem calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluctuationTheorem {
    /// Duration of observation in seconds.
    pub tau: f64,
    /// Temperature in Kelvin.
    pub temperature: f64,
}

impl FluctuationTheorem {
    /// Create a new fluctuation theorem calculator.
    pub fn new(tau: f64, temperature: f64) -> Self {
        Self { tau, temperature }
    }

    /// Entropy production rate ratio: P(σ)/P(-σ) = e^(σ τ).
    pub fn probability_ratio(&self, sigma: f64) -> f64 {
        (sigma * self.tau).exp()
    }

    /// Log-probability ratio: ln(P(σ)/P(-σ)) = σ τ.
    pub fn log_probability_ratio(&self, sigma: f64) -> f64 {
        sigma * self.tau
    }

    /// Probability of observing entropy decrease ΔS in time τ.
    /// P(ΔS < 0) ≈ e^(-|ΔS| τ / k)
    pub fn probability_of_decrease(&self, delta_s: f64) -> f64 {
        if delta_s >= 0.0 {
            return 1.0; // Entropy increase is common
        }
        (-delta_s.abs() * self.tau / BOLTZMANN).exp().min(1.0)
    }

    /// Probability of observing a specific entropy production rate.
    /// Uses the Gallavotti-Cohen form: P(σ) ∝ e^(-I(σ) τ)
    /// where I(σ) is the large deviation rate function.
    pub fn probability_of_rate(&self, sigma: f64, mean_sigma: f64) -> f64 {
        // Simplified large deviation: I(σ) ≈ (σ - mean)² / (2 * var)
        let var = 2.0 * mean_sigma.abs().max(0.01); // Variance scales with mean
        let rate_function = (sigma - mean_sigma).powi(2) / (2.0 * var);
        (-rate_function * self.tau).exp()
    }

    /// Check if a fluctuation violates the second law.
    /// Returns true if ΔS < 0 but it's within fluctuation bounds.
    pub fn is_allowed_fluctuation(&self, delta_s: f64, confidence: f64) -> bool {
        if delta_s >= 0.0 {
            return true;
        }
        let p = self.probability_of_decrease(delta_s);
        p >= 1.0 - confidence
    }

    /// Minimum observable entropy production (resolution limit).
    pub fn resolution_limit(&self) -> f64 {
        BOLTZMANN / self.tau
    }
}

/// Detailed fluctuation theorem: for individual trajectories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedFluctuation {
    /// Entropy production in forward trajectory.
    pub forward_entropy: f64,
    /// Entropy production in reverse trajectory.
    pub reverse_entropy: f64,
    /// Temperature.
    pub temperature: f64,
}

impl DetailedFluctuation {
    /// Create a new detailed fluctuation.
    pub fn new(forward_entropy: f64, reverse_entropy: f64, temperature: f64) -> Self {
        Self {
            forward_entropy,
            reverse_entropy,
            temperature,
        }
    }

    /// Crooks fluctuation theorem: P_F(W) / P_R(-W) = e^(β(W - ΔF)).
    pub fn crooks_ratio(&self, delta_f: f64) -> f64 {
        let beta = 1.0 / (BOLTZMANN * self.temperature);
        let work = self.forward_entropy * self.temperature; // W = T * ΔS
        (beta * (work - delta_f)).exp()
    }

    /// Net entropy production.
    pub fn net_entropy_production(&self) -> f64 {
        self.forward_entropy - self.reverse_entropy
    }

    /// Verify the relation: ΔS_total ≥ 0 on average.
    pub fn verify_on_average(&self) -> bool {
        self.net_entropy_production() >= -f64::EPSILON
    }
}

/// Agent-scale fluctuation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFluctuation {
    /// The fluctuation theorem.
    pub theorem: FluctuationTheorem,
    /// Observed entropy productions.
    pub entropy_productions: Vec<f64>,
}

impl AgentFluctuation {
    /// Create a new agent fluctuation analysis.
    pub fn new(tau: f64, temperature: f64) -> Self {
        Self {
            theorem: FluctuationTheorem::new(tau, temperature),
            entropy_productions: Vec::new(),
        }
    }

    /// Record an entropy production observation.
    pub fn record(&mut self, delta_s: f64) {
        self.entropy_productions.push(delta_s);
    }

    /// Mean entropy production.
    pub fn mean_entropy_production(&self) -> f64 {
        if self.entropy_productions.is_empty() {
            return 0.0;
        }
        self.entropy_productions.iter().sum::<f64>() / self.entropy_productions.len() as f64
    }

    /// Count of negative entropy productions (second law violations).
    pub fn negative_count(&self) -> usize {
        self.entropy_productions
            .iter()
            .filter(|&&s| s < 0.0)
            .count()
    }

    /// Fraction of trajectories with negative entropy production.
    pub fn negative_fraction(&self) -> f64 {
        if self.entropy_productions.is_empty() {
            return 0.0;
        }
        self.negative_count() as f64 / self.entropy_productions.len() as f64
    }

    /// Verify integral fluctuation theorem: ⟨e^(-ΔS/k)⟩ = 1.
    pub fn verify_integral_theorem(&self) -> f64 {
        if self.entropy_productions.is_empty() {
            return 1.0;
        }
        let avg: f64 = self
            .entropy_productions
            .iter()
            .map(|&ds| (-ds / BOLTZMANN).exp())
            .sum::<f64>()
            / self.entropy_productions.len() as f64;
        avg
    }

    /// Verify the theorem is satisfied (average close to 1).
    pub fn is_satisfied(&self, tolerance: f64) -> bool {
        (self.verify_integral_theorem() - 1.0).abs() < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_probability_ratio_positive() {
        let ft = FluctuationTheorem::new(1.0, 300.0);
        let r = ft.probability_ratio(1.0);
        assert!(r > 1.0); // Forward > backward
    }

    #[test]
    fn test_probability_ratio_symmetric() {
        let ft = FluctuationTheorem::new(1.0, 300.0);
        let r1 = ft.probability_ratio(1.0);
        let r2 = ft.probability_ratio(-1.0);
        assert_relative_eq!(r1 * r2, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_probability_of_decrease() {
        let ft = FluctuationTheorem::new(1.0, 300.0);
        let p = ft.probability_of_decrease(-BOLTZMANN);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn test_probability_of_increase() {
        let ft = FluctuationTheorem::new(1.0, 300.0);
        let p = ft.probability_of_decrease(BOLTZMANN);
        assert_relative_eq!(p, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_log_ratio() {
        let ft = FluctuationTheorem::new(2.0, 300.0);
        assert_relative_eq!(ft.log_probability_ratio(3.0), 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_detailed_fluctuation_net() {
        let df = DetailedFluctuation::new(1e-22, -1e-22, 300.0);
        let net = df.net_entropy_production();
        assert!(net > 0.0);
    }

    #[test]
    fn test_detailed_fluctuation_verify() {
        let df = DetailedFluctuation::new(2e-22, 1e-22, 300.0);
        assert!(df.verify_on_average());
    }

    #[test]
    fn test_crooks_ratio() {
        let df = DetailedFluctuation::new(1e-20, 0.0, 300.0);
        let r = df.crooks_ratio(0.0);
        assert!(r > 1.0);
    }

    #[test]
    fn test_agent_fluctuation_mean() {
        let mut af = AgentFluctuation::new(1.0, 300.0);
        af.record(1e-22);
        af.record(2e-22);
        af.record(3e-22);
        assert_relative_eq!(af.mean_entropy_production(), 2e-22, epsilon = 1e-30);
    }

    #[test]
    fn test_agent_fluctuation_negative_count() {
        let mut af = AgentFluctuation::new(1.0, 300.0);
        af.record(1e-22);
        af.record(-1e-22);
        af.record(1e-22);
        assert_eq!(af.negative_count(), 1);
    }

    #[test]
    fn test_resolution_limit() {
        let ft = FluctuationTheorem::new(1.0, 300.0);
        let limit = ft.resolution_limit();
        assert!(limit > 0.0);
        assert_relative_eq!(limit, BOLTZMANN, epsilon = 1e-30);
    }

    #[test]
    fn test_allowed_fluctuation_positive() {
        let ft = FluctuationTheorem::new(1.0, 300.0);
        assert!(ft.is_allowed_fluctuation(1e-22, 0.99));
    }
}
