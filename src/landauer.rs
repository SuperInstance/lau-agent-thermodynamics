//! Landauer's Principle: minimum energy cost of erasing information.
//!
//! Erasing one bit of information requires at least kT ln(2) of energy
//! dissipated as heat. This is the fundamental lower bound on computation.

use crate::constants::{BOLTZMANN, LN2};
use serde::{Deserialize, Serialize};

/// Landauer cost calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandauerCost {
    /// Temperature in Kelvin.
    pub temperature: f64,
}

impl LandauerCost {
    /// Create at given temperature.
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Cost to erase a single bit at this temperature.
    pub fn per_bit(&self) -> f64 {
        BOLTZMANN * self.temperature * LN2
    }

    /// Cost to erase n bits.
    pub fn for_bits(&self, n_bits: f64) -> f64 {
        self.per_bit() * n_bits
    }

    /// Cost to erase a belief state (measured by its Shannon entropy).
    pub fn for_belief(&self, entropy_bits: f64) -> f64 {
        self.per_bit() * entropy_bits
    }

    /// Number of bits that can be erased with given energy.
    pub fn bits_from_energy(&self, energy: f64) -> f64 {
        let per_bit = self.per_bit();
        if per_bit == 0.0 {
            return 0.0;
        }
        energy / per_bit
    }

    /// Temperature from a known cost and bit count.
    pub fn temperature_from_cost(cost: f64, n_bits: f64) -> f64 {
        if n_bits < f64::EPSILON {
            return 0.0;
        }
        cost / (BOLTZMANN * LN2 * n_bits)
    }
}

/// Record of an erasure operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureRecord {
    /// Bits erased.
    pub bits_erased: f64,
    /// Energy actually used.
    pub energy_used: f64,
    /// Temperature.
    pub temperature: f64,
}

impl ErasureRecord {
    /// Create a new erasure record.
    pub fn new(bits_erased: f64, energy_used: f64, temperature: f64) -> Self {
        Self {
            bits_erased,
            energy_used,
            temperature,
        }
    }

    /// Minimum energy required (Landauer bound).
    pub fn minimum_energy(&self) -> f64 {
        BOLTZMANN * self.temperature * LN2 * self.bits_erased
    }

    /// Whether this erasure satisfies the Landauer bound.
    pub fn satisfies_bound(&self) -> bool {
        self.energy_used >= self.minimum_energy()
    }

    /// Excess energy above the Landauer bound.
    pub fn excess_energy(&self) -> f64 {
        (self.energy_used - self.minimum_energy()).max(0.0)
    }

    /// Efficiency relative to the Landauer bound.
    pub fn efficiency(&self) -> f64 {
        if self.energy_used == 0.0 {
            return 0.0;
        }
        self.minimum_energy() / self.energy_used
    }

    /// Heat generated during erasure.
    pub fn heat_generated(&self) -> f64 {
        self.energy_used
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_landauer_per_bit() {
        let lc = LandauerCost::new(300.0);
        let expected = BOLTZMANN * 300.0 * LN2;
        assert_relative_eq!(lc.per_bit(), expected, epsilon = 1e-30);
    }

    #[test]
    fn test_landauer_for_bits() {
        let lc = LandauerCost::new(300.0);
        let cost = lc.for_bits(8.0);
        assert_relative_eq!(cost, 8.0 * lc.per_bit(), epsilon = 1e-30);
    }

    #[test]
    fn test_bits_from_energy() {
        let lc = LandauerCost::new(300.0);
        let energy = lc.per_bit() * 10.0;
        let bits = lc.bits_from_energy(energy);
        assert_relative_eq!(bits, 10.0, epsilon = 1e-5);
    }

    #[test]
    fn test_temperature_from_cost() {
        let t = LandauerCost::temperature_from_cost(BOLTZMANN * 300.0 * LN2, 1.0);
        assert_relative_eq!(t, 300.0, epsilon = 1e-5);
    }

    #[test]
    fn test_erasure_satisfies_bound() {
        let lc = LandauerCost::new(300.0);
        let record = ErasureRecord::new(1.0, lc.per_bit() * 2.0, 300.0);
        assert!(record.satisfies_bound());
    }

    #[test]
    fn test_erasure_violates_bound() {
        let record = ErasureRecord::new(1.0, 1e-25, 300.0);
        assert!(!record.satisfies_bound());
    }

    #[test]
    fn test_erasure_efficiency() {
        let lc = LandauerCost::new(300.0);
        let record = ErasureRecord::new(1.0, lc.per_bit() * 2.0, 300.0);
        assert_relative_eq!(record.efficiency(), 0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_excess_energy() {
        let lc = LandauerCost::new(300.0);
        let record = ErasureRecord::new(1.0, lc.per_bit() * 1.5, 300.0);
        assert_relative_eq!(record.excess_energy(), lc.per_bit() * 0.5, epsilon = 1e-30);
    }

    #[test]
    fn test_zero_bits() {
        let lc = LandauerCost::new(300.0);
        assert_relative_eq!(lc.for_bits(0.0), 0.0, epsilon = 1e-30);
    }

    #[test]
    fn test_landauer_at_different_temps() {
        let lc1 = LandauerCost::new(300.0);
        let lc2 = LandauerCost::new(600.0);
        assert!(lc2.per_bit() > lc1.per_bit());
    }
}
