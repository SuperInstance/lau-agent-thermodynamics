//! Carnot efficiency applied to agent learning.
//!
//! Maximum efficiency of an agent learning process = 1 - T_cold/T_hot.
//! Treats learning as a heat engine between belief reservoirs.

use crate::constants::BOLTZMANN;
use crate::second_law::BeliefState;
use serde::{Deserialize, Serialize};

/// Carnot engine for agent learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CarnotEngine {
    /// Hot reservoir temperature (high-entropy, exploratory regime).
    pub t_hot: f64,
    /// Cold reservoir temperature (low-entropy, exploitative regime).
    pub t_cold: f64,
}

impl CarnotEngine {
    /// Create a new Carnot engine between two temperatures.
    pub fn new(t_hot: f64, t_cold: f64) -> Self {
        Self { t_hot, t_cold }
    }

    /// Carnot efficiency: η = 1 - T_cold / T_hot.
    pub fn efficiency(&self) -> f64 {
        if self.t_hot < f64::EPSILON {
            return 0.0;
        }
        1.0 - self.t_cold / self.t_hot
    }

    /// Maximum work extractable from Q_hot units of heat.
    pub fn max_work(&self, q_hot: f64) -> f64 {
        q_hot * self.efficiency()
    }

    /// Heat rejected to cold reservoir for given Q_hot.
    pub fn heat_rejected(&self, q_hot: f64) -> f64 {
        q_hot - self.max_work(q_hot)
    }

    /// Work required to pump heat from cold to hot (learning = heat pump).
    pub fn work_for_pump(&self, q_cold: f64) -> f64 {
        let eff = self.efficiency();
        if eff < f64::EPSILON {
            return f64::INFINITY;
        }
        q_cold * self.t_hot / (self.t_hot - self.t_cold) - q_cold
    }

    /// Coefficient of performance for the heat pump (learning) mode.
    pub fn cop(&self) -> f64 {
        let diff = self.t_hot - self.t_cold;
        if diff < f64::EPSILON {
            return f64::INFINITY;
        }
        self.t_cold / diff
    }
}

/// Learning engine: maps Carnot cycle to agent learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEngine {
    /// The Carnot engine.
    pub carnot: CarnotEngine,
    /// Number of belief states.
    pub n_states: usize,
}

impl LearningEngine {
    /// Create a new learning engine.
    pub fn new(t_hot: f64, t_cold: f64, n_states: usize) -> Self {
        Self {
            carnot: CarnotEngine::new(t_hot, t_cold),
            n_states,
        }
    }

    /// Maximum information gain (in bits) achievable per cycle.
    pub fn max_info_gain(&self) -> f64 {
        // Entropy difference between hot (uniform) and cold (Boltzmann) beliefs
        let hot_belief = BeliefState::uniform(self.n_states);
        let cold_agent = crate::third_law::CooledAgent::new(
            crate::third_law::GroundState::new(0, self.n_states, BOLTZMANN * self.carnot.t_cold),
            self.carnot.t_cold,
        );
        hot_belief.shannon_entropy() - cold_agent.belief().shannon_entropy()
    }

    /// Energy cost per bit of learning.
    pub fn cost_per_bit(&self) -> f64 {
        let eff = self.carnot.efficiency();
        if eff < f64::EPSILON {
            return f64::INFINITY;
        }
        BOLTZMANN * self.carnot.t_hot * crate::constants::LN2 / eff
    }

    /// Optimal learning rate given available energy budget.
    pub fn optimal_learning_rate(&self, energy_budget: f64, total_bits: f64) -> f64 {
        if total_bits < f64::EPSILON {
            return 0.0;
        }
        let max_rate = energy_budget / (self.cost_per_bit() * total_bits);
        max_rate.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_carnot_efficiency() {
        let engine = CarnotEngine::new(600.0, 300.0);
        assert_relative_eq!(engine.efficiency(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_carnot_max_work() {
        let engine = CarnotEngine::new(600.0, 300.0);
        assert_relative_eq!(engine.max_work(100.0), 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_carnot_heat_rejected() {
        let engine = CarnotEngine::new(600.0, 300.0);
        assert_relative_eq!(engine.heat_rejected(100.0), 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_carnot_zero_efficiency() {
        let engine = CarnotEngine::new(300.0, 300.0);
        assert_relative_eq!(engine.efficiency(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_work_for_pump() {
        let engine = CarnotEngine::new(600.0, 300.0);
        let w = engine.work_for_pump(100.0);
        assert!(w > 0.0);
    }

    #[test]
    fn test_cop() {
        let engine = CarnotEngine::new(600.0, 300.0);
        assert_relative_eq!(engine.cop(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_learning_engine_creation() {
        let le = LearningEngine::new(600.0, 300.0, 4);
        assert_eq!(le.n_states, 4);
    }

    #[test]
    fn test_learning_cost_per_bit() {
        let le = LearningEngine::new(600.0, 300.0, 4);
        let cost = le.cost_per_bit();
        assert!(cost > 0.0);
    }

    #[test]
    fn test_optimal_learning_rate() {
        let le = LearningEngine::new(600.0, 300.0, 4);
        let rate = le.optimal_learning_rate(1e-18, 10.0);
        assert!(rate >= 0.0 && rate <= 1.0);
    }

    #[test]
    fn test_equal_temps_zero_efficiency() {
        let engine = CarnotEngine::new(300.0, 300.0);
        assert_relative_eq!(engine.efficiency(), 0.0, epsilon = 1e-10);
    }
}
