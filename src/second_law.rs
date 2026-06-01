//! Second Law of Thermodynamics applied to agent beliefs.
//!
//! Entropy of agent beliefs never decreases without external work.
//! ΔS_agent ≥ 0 (spontaneous) or ΔS_agent = W_ext / T (with work).

use crate::constants::BOLTZMANN;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Belief state as a probability distribution over discrete states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefState {
    /// Probability vector (must sum to 1).
    pub probabilities: Vec<f64>,
}

impl BeliefState {
    /// Create a new belief state from probabilities (normalizes automatically).
    pub fn new(probabilities: Vec<f64>) -> Self {
        let sum: f64 = probabilities.iter().sum();
        let normalized: Vec<f64> = probabilities.iter().map(|p| p / sum).collect();
        Self {
            probabilities: normalized,
        }
    }

    /// Create a uniform belief over n states.
    pub fn uniform(n: usize) -> Self {
        let p = 1.0 / n as f64;
        Self {
            probabilities: vec![p; n],
        }
    }

    /// Create a deterministic (peaked) belief at state i.
    pub fn deterministic(n: usize, i: usize) -> Self {
        let mut probs = vec![0.0; n];
        probs[i] = 1.0;
        Self { probabilities: probs }
    }

    /// Shannon entropy in bits.
    pub fn shannon_entropy(&self) -> f64 {
        -self
            .probabilities
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.log2())
            .sum::<f64>()
    }

    /// Thermodynamic entropy in J/K at temperature T.
    pub fn thermodynamic_entropy(&self, _temperature: f64) -> f64 {
        // S = k_B * H * ln(2) where H is Shannon entropy in bits
        self.shannon_entropy() * BOLTZMANN * crate::constants::LN2
    }

    /// Kullback-Leibler divergence from another belief state.
    pub fn kl_divergence(&self, other: &BeliefState) -> f64 {
        self.probabilities
            .iter()
            .zip(other.probabilities.iter())
            .filter(|(p, _)| **p > 0.0)
            .map(|(p, q)| p * (p / q).ln())
            .sum()
    }

    /// Convert to nalgebra vector.
    pub fn to_vector(&self) -> DVector<f64> {
        DVector::from_vec(self.probabilities.clone())
    }

    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.probabilities.len()
    }

    /// Update beliefs with new evidence (Bayesian update).
    pub fn bayesian_update(&self, likelihoods: &[f64]) -> BeliefState {
        let unnormalized: Vec<f64> = self
            .probabilities
            .iter()
            .zip(likelihoods.iter())
            .map(|(p, l)| p * l)
            .collect();
        BeliefState::new(unnormalized)
    }

    /// Work required to reduce entropy from current state to target.
    pub fn work_to_reduce_entropy(&self, target: &BeliefState, temperature: f64) -> f64 {
        let ds = target.shannon_entropy() - self.shannon_entropy();
        if ds < 0.0 {
            -ds * BOLTZMANN * temperature
        } else {
            0.0
        }
    }
}

/// Tracker for entropy changes in an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyTracker {
    /// History of belief states.
    pub history: Vec<BeliefState>,
    /// Temperature in Kelvin.
    pub temperature: f64,
    /// External work applied at each step.
    pub work_applied: Vec<f64>,
}

impl EntropyTracker {
    /// Create a new entropy tracker.
    pub fn new(initial: BeliefState, temperature: f64) -> Self {
        Self {
            history: vec![initial],
            temperature,
            work_applied: vec![0.0],
        }
    }

    /// Record a belief update with associated work.
    pub fn update(&mut self, new_belief: BeliefState, work: f64) {
        self.history.push(new_belief);
        self.work_applied.push(work);
    }

    /// Check if second law is satisfied at step i.
    /// Entropy can decrease only if external work was applied.
    pub fn check_second_law(&self, step: usize) -> bool {
        if step == 0 || step >= self.history.len() {
            return true;
        }
        let ds = self.history[step].shannon_entropy() - self.history[step - 1].shannon_entropy();
        if ds >= 0.0 {
            return true; // Entropy increased or stayed same — always OK
        }
        // Entropy decreased — requires work
        let work_required = -ds * BOLTZMANN * self.temperature;
        self.work_applied[step] >= work_required
    }

    /// Total entropy change over all steps.
    pub fn total_entropy_change(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        self.history.last().unwrap().shannon_entropy() - self.history.first().unwrap().shannon_entropy()
    }

    /// Total external work applied.
    pub fn total_work(&self) -> f64 {
        self.work_applied.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_uniform_entropy() {
        let b = BeliefState::uniform(2);
        assert_relative_eq!(b.shannon_entropy(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_deterministic_entropy() {
        let b = BeliefState::deterministic(4, 0);
        assert_relative_eq!(b.shannon_entropy(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_4_states() {
        let b = BeliefState::uniform(4);
        assert_relative_eq!(b.shannon_entropy(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy_never_decreases_spontaneous() {
        let mut tracker = EntropyTracker::new(BeliefState::uniform(4), 300.0);
        // Uniform → deterministic is entropy decrease, but with no work
        let peaked = BeliefState::deterministic(4, 0);
        tracker.update(peaked, 0.0);
        assert!(!tracker.check_second_law(1)); // Should violate
    }

    #[test]
    fn test_entropy_decrease_with_work() {
        let mut tracker = EntropyTracker::new(BeliefState::uniform(4), 300.0);
        let peaked = BeliefState::deterministic(4, 0);
        let ds = peaked.shannon_entropy() - BeliefState::uniform(4).shannon_entropy();
        let work = -ds * BOLTZMANN * 300.0;
        tracker.update(peaked, work * 2.0); // Apply more than needed
        assert!(tracker.check_second_law(1));
    }

    #[test]
    fn test_kl_divergence_same() {
        let b = BeliefState::uniform(4);
        assert_relative_eq!(b.kl_divergence(&b), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_positive() {
        let b1 = BeliefState::deterministic(2, 0);
        let b2 = BeliefState::uniform(2);
        assert!(b1.kl_divergence(&b2) > 0.0);
    }

    #[test]
    fn test_bayesian_update() {
        let prior = BeliefState::uniform(2);
        let likelihood = vec![0.9, 0.1];
        let posterior = prior.bayesian_update(&likelihood);
        assert!(posterior.probabilities[0] > 0.5);
    }

    #[test]
    fn test_work_to_reduce_entropy() {
        let uniform = BeliefState::uniform(2);
        let peaked = BeliefState::deterministic(2, 0);
        let w = uniform.work_to_reduce_entropy(&peaked, 300.0);
        assert!(w > 0.0);
    }

    #[test]
    fn test_no_work_needed_for_entropy_increase() {
        let peaked = BeliefState::deterministic(2, 0);
        let uniform = BeliefState::uniform(2);
        let w = peaked.work_to_reduce_entropy(&uniform, 300.0);
        assert_eq!(w, 0.0);
    }

    #[test]
    fn test_normalization() {
        let b = BeliefState::new(vec![3.0, 1.0]);
        let sum: f64 = b.probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_total_entropy_change() {
        let mut tracker = EntropyTracker::new(BeliefState::deterministic(4, 0), 300.0);
        tracker.update(BeliefState::uniform(4), 0.0);
        assert_relative_eq!(tracker.total_entropy_change(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_to_vector() {
        let b = BeliefState::uniform(3);
        let v = b.to_vector();
        assert_eq!(v.len(), 3);
    }
}
