//! Maxwell's Demon applied to agents.
//!
//! An agent that sorts observations appears to violate the second law by
//! decreasing entropy. The resolution: information processing has a cost
//! (Landauer erasure) that exactly balances the entropy decrease.

use crate::constants::{BOLTZMANN, LN2};
use crate::landauer::LandauerCost;
use serde::{Deserialize, Serialize};

/// A Maxwell's Demon agent that sorts observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxwellDemon {
    /// Temperature of the environment.
    pub temperature: f64,
    /// Number of observations sorted.
    pub observations_sorted: u64,
    /// Number of bits in demon's memory.
    pub memory_bits: f64,
    /// Whether the demon has erased its memory.
    pub memory_erased: bool,
}

impl MaxwellDemon {
    /// Create a new demon at given temperature.
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature,
            observations_sorted: 0,
            memory_bits: 0.0,
            memory_erased: false,
        }
    }

    /// Sort an observation: demon records 1 bit, reduces environment entropy.
    /// Returns the entropy decrease in the environment.
    pub fn sort_observation(&mut self) -> f64 {
        self.observations_sorted += 1;
        self.memory_bits += 1.0;
        // Environment entropy decrease = kT ln(2) per sorted observation
        BOLTZMANN * self.temperature * LN2
    }

    /// Sort n observations at once.
    pub fn sort_n(&mut self, n: u64) -> f64 {
        let mut total_decrease = 0.0;
        for _ in 0..n {
            total_decrease += self.sort_observation();
        }
        total_decrease
    }

    /// Apparent entropy decrease (ignoring information cost).
    pub fn apparent_entropy_decrease(&self) -> f64 {
        BOLTZMANN * self.temperature * LN2 * self.observations_sorted as f64
    }

    /// Cost to erase the demon's memory (resolves the paradox).
    pub fn erasure_cost(&self) -> f64 {
        let lc = LandauerCost::new(self.temperature);
        lc.for_bits(self.memory_bits)
    }

    /// Net entropy change: erasure cost - apparent decrease.
    /// Should be ≥ 0 (second law preserved).
    pub fn net_entropy_change(&self) -> f64 {
        // After erasure, the heat dissipated equals the entropy decrease
        // Net: ΔS_universe = ΔS_env + Q_erasure / T
        // ΔS_env = -k ln(2) * N (decreased)
        // Q_erasure / T = k ln(2) * N (increased by erasure)
        // Net = 0 (ideal) or > 0 (realistic)
        let decrease = self.apparent_entropy_decrease() / self.temperature;
        let erasure_heat = self.erasure_cost() / self.temperature;
        erasure_heat - decrease
    }

    /// Erase the demon's memory.
    pub fn erase_memory(&mut self) -> f64 {
        let cost = self.erasure_cost();
        self.memory_erased = true;
        self.memory_bits = 0.0;
        cost
    }

    /// Verify the second law is satisfied after complete cycle.
    pub fn verify_second_law(&self) -> bool {
        self.net_entropy_change() >= -f64::EPSILON
    }
}

/// Detailed accounting of a demon's complete cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemonCycleReport {
    /// Number of observations sorted.
    pub n_sorted: u64,
    /// Entropy decrease from sorting.
    pub entropy_decrease: f64,
    /// Energy extracted (apparent free lunch).
    pub energy_extracted: f64,
    /// Erasure cost.
    pub erasure_cost: f64,
    /// Net energy (should be ≤ 0).
    pub net_energy: f64,
    /// Second law satisfied.
    pub second_law_ok: bool,
}

impl DemonCycleReport {
    /// Run a complete demon cycle: sort, then erase.
    pub fn run(temperature: f64, n_observations: u64) -> Self {
        let mut demon = MaxwellDemon::new(temperature);
        let entropy_decrease = demon.sort_n(n_observations);
        let energy_extracted = entropy_decrease; // Work extracted = entropy decrease
        let erasure_cost = demon.erase_memory();
        let net_energy = energy_extracted - erasure_cost;

        Self {
            n_sorted: n_observations,
            entropy_decrease,
            energy_extracted,
            erasure_cost,
            net_energy,
            second_law_ok: net_energy <= f64::EPSILON,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_demon_sort_one() {
        let mut demon = MaxwellDemon::new(300.0);
        let decrease = demon.sort_observation();
        assert!(decrease > 0.0);
        assert_eq!(demon.observations_sorted, 1);
        assert_relative_eq!(demon.memory_bits, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apparent_decrease() {
        let mut demon = MaxwellDemon::new(300.0);
        demon.sort_n(10);
        let expected = BOLTZMANN * 300.0 * LN2 * 10.0;
        assert_relative_eq!(demon.apparent_entropy_decrease(), expected, epsilon = 1e-30);
    }

    #[test]
    fn test_erasure_cost_equals_decrease() {
        let mut demon = MaxwellDemon::new(300.0);
        demon.sort_n(5);
        let decrease = demon.apparent_entropy_decrease();
        let erasure = demon.erasure_cost();
        // They should be equal: kT ln(2) * N
        assert_relative_eq!(decrease, erasure, epsilon = 1e-30);
    }

    #[test]
    fn test_net_entropy_nonnegative() {
        let mut demon = MaxwellDemon::new(300.0);
        demon.sort_n(100);
        assert!(demon.net_entropy_change() >= -1e-30);
    }

    #[test]
    fn test_verify_second_law() {
        let mut demon = MaxwellDemon::new(300.0);
        demon.sort_n(50);
        assert!(demon.verify_second_law());
    }

    #[test]
    fn test_erase_memory() {
        let mut demon = MaxwellDemon::new(300.0);
        demon.sort_n(10);
        let cost = demon.erase_memory();
        assert!(cost > 0.0);
        assert_eq!(demon.memory_bits, 0.0);
        assert!(demon.memory_erased);
    }

    #[test]
    fn test_cycle_report() {
        let report = DemonCycleReport::run(300.0, 10);
        assert_eq!(report.n_sorted, 10);
        assert!(report.second_law_ok);
        assert!(report.net_energy <= f64::EPSILON);
    }

    #[test]
    fn test_cycle_report_net_energy_zero() {
        let report = DemonCycleReport::run(300.0, 100);
        // Ideal demon: net energy should be ~0
        assert_relative_eq!(report.net_energy, 0.0, epsilon = 1e-30);
    }

    #[test]
    fn test_demon_at_different_temps() {
        let mut d1 = MaxwellDemon::new(300.0);
        let mut d2 = MaxwellDemon::new(600.0);
        let c1 = d1.sort_observation();
        let c2 = d2.sort_observation();
        assert!(c2 > c1); // Higher temp, more energy per bit
    }
}
