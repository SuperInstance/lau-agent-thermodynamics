//! Physical constants used throughout the crate.

use serde::{Deserialize, Serialize};

/// Boltzmann constant in J/K.
pub const BOLTZMANN: f64 = 1.38064852e-23;

/// Natural log of 2, used frequently in information-theoretic calculations.
pub const LN2: f64 = std::f64::consts::LN_2;

/// Thermodynamic state of an agent at a given point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicState {
    /// Temperature in Kelvin.
    pub temperature: f64,
    /// Internal energy in Joules.
    pub internal_energy: f64,
    /// Entropy in J/K.
    pub entropy: f64,
    /// Number of bits in the agent's belief state.
    pub belief_bits: f64,
    /// Work done on the agent in Joules.
    pub work: f64,
    /// Heat exchanged with the environment in Joules.
    pub heat: f64,
}

impl ThermodynamicState {
    /// Create a new thermodynamic state at the given temperature.
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature,
            internal_energy: 0.0,
            entropy: 0.0,
            belief_bits: 0.0,
            work: 0.0,
            heat: 0.0,
        }
    }

    /// Free energy: F = U - TS.
    pub fn free_energy(&self) -> f64 {
        self.internal_energy - self.temperature * self.entropy
    }

    /// Landauer cost to maintain current beliefs at this temperature.
    pub fn landauer_cost(&self) -> f64 {
        BOLTZMANN * self.temperature * LN2 * self.belief_bits
    }

    /// Thermodynamic efficiency of this state.
    pub fn efficiency(&self) -> f64 {
        if self.work.abs() < f64::EPSILON {
            return 0.0;
        }
        (self.work - self.landauer_cost()) / self.work
    }
}

impl Default for ThermodynamicState {
    fn default() -> Self {
        Self::new(300.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_boltzmann_constant() {
        assert_relative_eq!(BOLTZMANN, 1.38064852e-23, epsilon = 1e-30);
    }

    #[test]
    fn test_ln2() {
        assert_relative_eq!(LN2, 0.6931471805599453, epsilon = 1e-15);
    }

    #[test]
    fn test_thermodynamic_state_new() {
        let s = ThermodynamicState::new(300.0);
        assert_eq!(s.temperature, 300.0);
        assert_eq!(s.internal_energy, 0.0);
        assert_eq!(s.entropy, 0.0);
    }

    #[test]
    fn test_free_energy_zero() {
        let s = ThermodynamicState::new(300.0);
        assert_eq!(s.free_energy(), 0.0);
    }

    #[test]
    fn test_free_energy_nonzero() {
        let s = ThermodynamicState {
            temperature: 300.0,
            internal_energy: 1e-20,
            entropy: 1e-23,
            ..Default::default()
        };
        let expected = 1e-20 - 300.0 * 1e-23;
        assert_relative_eq!(s.free_energy(), expected, epsilon = 1e-30);
    }

    #[test]
    fn test_landauer_cost() {
        let s = ThermodynamicState {
            temperature: 300.0,
            belief_bits: 1.0,
            ..Default::default()
        };
        let expected = BOLTZMANN * 300.0 * LN2;
        assert_relative_eq!(s.landauer_cost(), expected, epsilon = 1e-30);
    }

    #[test]
    fn test_state_serialization() {
        let s = ThermodynamicState::new(42.0);
        let json = serde_json::to_string(&s).unwrap();
        let s2: ThermodynamicState = serde_json::from_str(&json).unwrap();
        assert_eq!(s.temperature, s2.temperature);
    }
}
