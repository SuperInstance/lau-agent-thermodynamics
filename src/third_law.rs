//! Third Law of Thermodynamics applied to agents.
//!
//! As T → 0, the agent approaches its ground state (optimal policy) with zero entropy.
//! In the zero-temperature limit, all uncertainty is eliminated and the agent
//! converges to deterministic optimal behavior.

use crate::constants::BOLTZMANN;
use crate::second_law::BeliefState;
use serde::{Deserialize, Serialize};

/// Ground state representation: the optimal deterministic policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundState {
    /// Index of the optimal action/state.
    pub optimal_state: usize,
    /// Number of possible states.
    pub n_states: usize,
    /// Energy gap to next-best state in Joules.
    pub energy_gap: f64,
}

impl GroundState {
    /// Create a new ground state.
    pub fn new(optimal_state: usize, n_states: usize, energy_gap: f64) -> Self {
        Self {
            optimal_state,
            n_states,
            energy_gap,
        }
    }

    /// Belief state corresponding to this ground state (fully concentrated).
    pub fn belief(&self) -> BeliefState {
        BeliefState::deterministic(self.n_states, self.optimal_state)
    }

    /// Residual entropy at temperature T due to thermal fluctuations.
    /// Boltzmann distribution entropy: S = k * sum(p_i * ln(1/p_i))
    /// In the low-T limit, S ~ k * exp(-ΔE / kT)
    pub fn residual_entropy(&self, temperature: f64) -> f64 {
        if temperature < f64::EPSILON {
            return 0.0;
        }
        // Boltzmann weights
        let z = 1.0 + (self.n_states - 1) as f64
            * (-self.energy_gap / (BOLTZMANN * temperature)).exp();
        let p0 = 1.0 / z;
        let p_other = (-self.energy_gap / (BOLTZMANN * temperature)).exp() / z;

        let mut s = 0.0;
        if p0 > 0.0 {
            s -= p0 * p0.ln();
        }
        for _ in 1..self.n_states {
            if p_other > 0.0 {
                s -= p_other * p_other.ln();
            }
        }
        BOLTZMANN * s
    }
}

/// Temperature-dependent agent state converging to ground state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CooledAgent {
    /// Ground state the agent converges to.
    pub ground: GroundState,
    /// Current temperature.
    pub temperature: f64,
}

impl CooledAgent {
    /// Create a cooled agent.
    pub fn new(ground: GroundState, temperature: f64) -> Self {
        Self { ground, temperature }
    }

    /// Current belief at this temperature.
    pub fn belief(&self) -> BeliefState {
        if self.temperature < f64::EPSILON {
            return self.ground.belief();
        }
        let beta = 1.0 / (BOLTZMANN * self.temperature);
        let mut probs = vec![0.0; self.ground.n_states];

        probs[self.ground.optimal_state] = 1.0;
        let boltz_other = (-self.ground.energy_gap * beta).exp();
        for i in 0..self.ground.n_states {
            if i != self.ground.optimal_state {
                probs[i] = boltz_other;
            }
        }
        BeliefState::new(probs)
    }

    /// Current entropy.
    pub fn entropy(&self) -> f64 {
        self.ground.residual_entropy(self.temperature)
    }

    /// Cool the agent to a new temperature.
    pub fn cool(&self, new_temperature: f64) -> CooledAgent {
        CooledAgent::new(self.ground.clone(), new_temperature)
    }

    /// Check if the agent is effectively at ground state.
    pub fn is_ground_state(&self, tolerance: f64) -> bool {
        self.entropy() < tolerance
    }
}

/// Cooling schedule for annealing an agent to ground state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoolingSchedule {
    /// Starting temperature.
    pub t_start: f64,
    /// Final temperature.
    pub t_end: f64,
    /// Number of steps.
    pub n_steps: usize,
    /// Schedule type.
    pub schedule_type: CoolingType,
}

/// Type of cooling schedule.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CoolingType {
    /// Linear: T = T_start - (T_start - T_end) * (step / n_steps).
    Linear,
    /// Exponential: T = T_start * (T_end / T_start)^(step / n_steps).
    Exponential,
    /// Inverse: T = T_start / (1 + alpha * step).
    Inverse,
}

impl CoolingSchedule {
    /// Temperature at a given step.
    pub fn temperature_at(&self, step: usize) -> f64 {
        if step >= self.n_steps {
            return self.t_end;
        }
        let frac = step as f64 / self.n_steps as f64;
        match self.schedule_type {
            CoolingType::Linear => self.t_start - (self.t_start - self.t_end) * frac,
            CoolingType::Exponential => {
                self.t_start * (self.t_end / self.t_start).powf(frac)
            }
            CoolingType::Inverse => {
                let alpha = (self.t_start - self.t_end) / (self.t_end * self.n_steps as f64);
                self.t_start / (1.0 + alpha * step as f64)
            }
        }
    }

    /// Full schedule of temperatures.
    pub fn all_temperatures(&self) -> Vec<f64> {
        (0..=self.n_steps).map(|i| self.temperature_at(i)).collect()
    }

    /// Compute entropy trajectory for a given ground state.
    pub fn entropy_trajectory(&self, ground: &GroundState) -> Vec<f64> {
        self.all_temperatures()
            .iter()
            .map(|&t| ground.residual_entropy(t))
            .collect()
    }

    /// Verify third law: entropy → 0 as T → 0.
    pub fn verify_third_law(&self, ground: &GroundState, tolerance: f64) -> bool {
        let final_entropy = ground.residual_entropy(self.t_end);
        final_entropy < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ground_state_zero_entropy() {
        let gs = GroundState::new(0, 4, 1e-20);
        let s = gs.residual_entropy(0.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_ground_state_belief() {
        let gs = GroundState::new(0, 4, 1e-20);
        let b = gs.belief();
        assert_eq!(b.probabilities[0], 1.0);
    }

    #[test]
    fn test_cooled_agent_zero_temp() {
        let gs = GroundState::new(0, 4, 1e-20);
        let agent = CooledAgent::new(gs, 0.0);
        let b = agent.belief();
        assert_eq!(b.probabilities[0], 1.0);
    }

    #[test]
    fn test_cooled_agent_high_temp() {
        let gs = GroundState::new(0, 4, 1e-20);
        let agent = CooledAgent::new(gs, 1e20);
        let b = agent.belief();
        // At very high temp, should be nearly uniform
        let max_diff = b
            .probabilities
            .iter()
            .map(|p| (p - 0.25).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff < 0.01);
    }

    #[test]
    fn test_entropy_decreases_with_cooling() {
        let gs = GroundState::new(0, 4, 1e-20);
        let s_hot = gs.residual_entropy(1e6);
        let s_cold = gs.residual_entropy(100.0);
        assert!(s_cold < s_hot);
    }

    #[test]
    fn test_is_ground_state() {
        let gs = GroundState::new(0, 4, 1e-20);
        let agent = CooledAgent::new(gs, 1e-3);
        assert!(agent.is_ground_state(1e-30));
    }

    #[test]
    fn test_linear_cooling() {
        let schedule = CoolingSchedule {
            t_start: 100.0,
            t_end: 0.0,
            n_steps: 100,
            schedule_type: CoolingType::Linear,
        };
        assert_relative_eq!(schedule.temperature_at(0), 100.0, epsilon = 1e-10);
        assert_relative_eq!(schedule.temperature_at(100), 0.0, epsilon = 1e-10);
        assert_relative_eq!(schedule.temperature_at(50), 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exponential_cooling() {
        let schedule = CoolingSchedule {
            t_start: 100.0,
            t_end: 1.0,
            n_steps: 100,
            schedule_type: CoolingType::Exponential,
        };
        assert_relative_eq!(schedule.temperature_at(0), 100.0, epsilon = 1e-10);
        assert_relative_eq!(schedule.temperature_at(100), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_verify_third_law() {
        let gs = GroundState::new(0, 4, 1e-20);
        let schedule = CoolingSchedule {
            t_start: 300.0,
            t_end: 0.001,
            n_steps: 1000,
            schedule_type: CoolingType::Exponential,
        };
        assert!(schedule.verify_third_law(&gs, 1e-30));
    }

    #[test]
    fn test_entropy_trajectory_monotone() {
        let gs = GroundState::new(0, 4, 1e-20);
        let schedule = CoolingSchedule {
            t_start: 300.0,
            t_end: 0.01,
            n_steps: 50,
            schedule_type: CoolingType::Exponential,
        };
        let traj = schedule.entropy_trajectory(&gs);
        for i in 1..traj.len() {
            assert!(traj[i] <= traj[i - 1] + 1e-35); // Non-increasing (with tolerance)
        }
    }

    #[test]
    fn test_cool_produces_new_temperature() {
        let gs = GroundState::new(0, 4, 1e-20);
        let agent = CooledAgent::new(gs, 300.0);
        let cooled = agent.cool(100.0);
        assert_relative_eq!(cooled.temperature, 100.0, epsilon = 1e-10);
    }
}
