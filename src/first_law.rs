//! First Law of Thermodynamics applied to agent computation.
//!
//! Energy conservation: E_input = E_computation + E_dissipation
//! All energy flowing into an agent must be accounted for as either useful
//! computation or waste heat.

use crate::constants::BOLTZMANN;
use serde::{Deserialize, Serialize};

/// Energy budget for an agent computation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyBudget {
    /// Total energy input in Joules.
    pub input_energy: f64,
    /// Energy used for computation in Joules.
    pub computation_energy: f64,
    /// Energy dissipated as heat in Joules.
    pub dissipated_energy: f64,
}

impl EnergyBudget {
    /// Create an energy budget and compute dissipation from conservation.
    pub fn new(input_energy: f64, computation_energy: f64) -> Self {
        Self {
            input_energy,
            computation_energy,
            dissipated_energy: input_energy - computation_energy,
        }
    }

    /// Verify first law: input = computation + dissipation.
    pub fn verify_conservation(&self) -> bool {
        let scale = self.input_energy.abs().max(self.computation_energy.abs()).max(self.dissipated_energy.abs()).max(1e-30);
        (self.input_energy - self.computation_energy - self.dissipated_energy).abs()
            < 1e-10 * scale
    }

    /// Fraction of input energy used for computation.
    pub fn computation_fraction(&self) -> f64 {
        if self.input_energy.abs() < f64::EPSILON {
            return 0.0;
        }
        self.computation_energy / self.input_energy
    }

    /// Minimum energy required to process n bits at temperature T.
    pub fn minimum_energy(n_bits: f64, temperature: f64) -> f64 {
        BOLTZMANN * temperature * crate::constants::LN2 * n_bits
    }
}

/// First law accounting for a sequence of agent operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstLawAccounting {
    /// Sequence of energy budgets.
    pub budgets: Vec<EnergyBudget>,
    /// Cumulative energy input.
    pub total_input: f64,
    /// Cumulative computation energy.
    pub total_computation: f64,
    /// Cumulative dissipation.
    pub total_dissipation: f64,
}

impl FirstLawAccounting {
    /// Create a new accounting tracker.
    pub fn new() -> Self {
        Self {
            budgets: Vec::new(),
            total_input: 0.0,
            total_computation: 0.0,
            total_dissipation: 0.0,
        }
    }

    /// Record a computation step.
    pub fn record(&mut self, budget: EnergyBudget) {
        self.total_input += budget.input_energy;
        self.total_computation += budget.computation_energy;
        self.total_dissipation += budget.dissipated_energy;
        self.budgets.push(budget);
    }

    /// Verify conservation over all steps.
    pub fn verify_global_conservation(&self) -> bool {
        let scale = self.total_input.abs().max(self.total_computation.abs()).max(self.total_dissipation.abs()).max(1e-30);
        (self.total_input - self.total_computation - self.total_dissipation).abs()
            < 1e-10 * scale
    }

    /// Overall efficiency across all steps.
    pub fn overall_efficiency(&self) -> f64 {
        if self.total_input.abs() < f64::EPSILON {
            return 0.0;
        }
        self.total_computation / self.total_input
    }
}

impl Default for FirstLawAccounting {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservation() {
        let budget = EnergyBudget::new(1e-18, 0.6e-18);
        assert!(budget.verify_conservation());
        assert!((budget.dissipated_energy - 0.4e-18).abs() < 1e-33);
    }

    #[test]
    fn test_computation_fraction() {
        let budget = EnergyBudget::new(1.0, 0.5);
        assert!((budget.computation_fraction() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_minimum_energy() {
        let e = EnergyBudget::minimum_energy(1.0, 300.0);
        let expected = BOLTZMANN * 300.0 * crate::constants::LN2;
        assert!((e - expected).abs() < 1e-30);
    }

    #[test]
    fn test_zero_input() {
        let budget = EnergyBudget::new(0.0, 0.0);
        assert!(budget.verify_conservation());
        assert_eq!(budget.computation_fraction(), 0.0);
    }

    #[test]
    fn test_accounting_multiple_steps() {
        let mut acc = FirstLawAccounting::new();
        acc.record(EnergyBudget::new(1e-18, 0.4e-18));
        acc.record(EnergyBudget::new(2e-18, 1.0e-18));
        assert!(acc.verify_global_conservation());
        assert!((acc.total_input - 3e-18).abs() < 1e-30);
        assert!((acc.total_computation - 1.4e-18).abs() < 1e-30);
    }

    #[test]
    fn test_overall_efficiency() {
        let mut acc = FirstLawAccounting::new();
        acc.record(EnergyBudget::new(1.0, 0.5));
        acc.record(EnergyBudget::new(1.0, 0.5));
        assert!((acc.overall_efficiency() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_serialization() {
        let budget = EnergyBudget::new(1e-18, 0.5e-18);
        let json = serde_json::to_string(&budget).unwrap();
        let b2: EnergyBudget = serde_json::from_str(&json).unwrap();
        assert_eq!(budget.input_energy, b2.input_energy);
    }
}
