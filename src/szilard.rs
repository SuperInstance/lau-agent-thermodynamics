//! Szilard Engine: extracting work from observations.
//!
//! One bit of information yields exactly kT ln(2) of extractable work.
//! This is the complementary view to Landauer: information has fuel value.

use crate::constants::{BOLTZMANN, LN2};
use serde::{Deserialize, Serialize};

/// A Szilard engine that extracts work from information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SzilardEngine {
    /// Temperature in Kelvin.
    pub temperature: f64,
    /// Total bits of information consumed.
    pub bits_consumed: f64,
    /// Total work extracted.
    pub work_extracted: f64,
    /// Whether we account for reset costs.
    pub include_reset_cost: bool,
}

impl SzilardEngine {
    /// Create a new Szilard engine.
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature,
            bits_consumed: 0.0,
            work_extracted: 0.0,
            include_reset_cost: true,
        }
    }

    /// Maximum extractable work per bit: W = kT ln(2).
    pub fn work_per_bit(&self) -> f64 {
        BOLTZMANN * self.temperature * LN2
    }

    /// Extract work from one bit of information.
    pub fn extract_one_bit(&mut self) -> f64 {
        let w = self.work_per_bit();
        self.bits_consumed += 1.0;
        self.work_extracted += w;
        w
    }

    /// Extract work from n bits.
    pub fn extract_bits(&mut self, n: f64) -> f64 {
        let w = self.work_per_bit() * n;
        self.bits_consumed += n;
        self.work_extracted += w;
        w
    }

    /// Net work after accounting for measurement reset costs.
    /// If reset cost is included, net work = 0 (no free lunch).
    pub fn net_work(&self) -> f64 {
        if self.include_reset_cost {
            0.0 // Reset cost exactly equals extracted work
        } else {
            self.work_extracted
        }
    }

    /// Efficiency: net work / theoretical maximum.
    pub fn efficiency(&self) -> f64 {
        let theoretical_max = self.work_per_bit() * self.bits_consumed;
        if theoretical_max < f64::EPSILON {
            return 0.0;
        }
        self.work_extracted / theoretical_max
    }

    /// Reset the engine for a new cycle.
    pub fn reset(&mut self) {
        self.bits_consumed = 0.0;
        self.work_extracted = 0.0;
    }
}

/// A single Szilard cycle: measure → extract → reset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SzilardCycle {
    /// Temperature.
    pub temperature: f64,
    /// Work extracted during expansion.
    pub work_extracted: f64,
    /// Cost of measurement (information acquisition).
    pub measurement_cost: f64,
    /// Cost of reset.
    pub reset_cost: f64,
    /// Net work: extracted - measurement - reset.
    pub net_work: f64,
}

impl SzilardCycle {
    /// Execute a complete Szilard cycle.
    pub fn execute(temperature: f64) -> Self {
        let w_extracted = BOLTZMANN * temperature * LN2;
        let measurement_cost = 0.0; // Measurement itself is free (reversible)
        let reset_cost = BOLTZMANN * temperature * LN2; // Erasure costs kT ln(2)

        Self {
            temperature,
            work_extracted: w_extracted,
            measurement_cost,
            reset_cost,
            net_work: w_extracted - measurement_cost - reset_cost,
        }
    }

    /// Execute cycle with n bits.
    pub fn execute_n(temperature: f64, n_bits: f64) -> Self {
        let single = Self::execute(temperature);
        Self {
            temperature,
            work_extracted: single.work_extracted * n_bits,
            measurement_cost: single.measurement_cost * n_bits,
            reset_cost: single.reset_cost * n_bits,
            net_work: single.net_work * n_bits,
        }
    }

    /// Verify no second law violation.
    pub fn verify_second_law(&self) -> bool {
        self.net_work <= f64::EPSILON
    }
}

/// Information fuel value calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationFuel {
    /// Temperature.
    pub temperature: f64,
}

impl InformationFuel {
    /// Create at given temperature.
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }

    /// Fuel value of n bits at this temperature.
    pub fn fuel_value(&self, n_bits: f64) -> f64 {
        BOLTZMANN * self.temperature * LN2 * n_bits
    }

    /// Bits needed to produce a given amount of work.
    pub fn bits_for_work(&self, work: f64) -> f64 {
        let per_bit = BOLTZMANN * self.temperature * LN2;
        if per_bit == 0.0 {
            return f64::INFINITY;
        }
        work / per_bit
    }

    /// Temperature needed to extract given work from given bits.
    pub fn temperature_for_work(work: f64, n_bits: f64) -> f64 {
        if n_bits < f64::EPSILON {
            return 0.0;
        }
        work / (BOLTZMANN * LN2 * n_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_work_per_bit() {
        let engine = SzilardEngine::new(300.0);
        let expected = BOLTZMANN * 300.0 * LN2;
        assert_relative_eq!(engine.work_per_bit(), expected, epsilon = 1e-30);
    }

    #[test]
    fn test_extract_one_bit() {
        let mut engine = SzilardEngine::new(300.0);
        let w = engine.extract_one_bit();
        assert!(w > 0.0);
        assert_relative_eq!(engine.bits_consumed, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_extract_n_bits() {
        let mut engine = SzilardEngine::new(300.0);
        let w = engine.extract_bits(10.0);
        assert_relative_eq!(w, BOLTZMANN * 300.0 * LN2 * 10.0, epsilon = 1e-30);
    }

    #[test]
    fn test_net_work_with_reset() {
        let mut engine = SzilardEngine::new(300.0);
        engine.include_reset_cost = true;
        engine.extract_bits(5.0);
        assert_relative_eq!(engine.net_work(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_net_work_without_reset() {
        let mut engine = SzilardEngine::new(300.0);
        engine.include_reset_cost = false;
        engine.extract_bits(5.0);
        assert!(engine.net_work() > 0.0);
    }

    #[test]
    fn test_cycle_execute() {
        let cycle = SzilardCycle::execute(300.0);
        assert!(cycle.work_extracted > 0.0);
        assert!(cycle.verify_second_law());
        assert_relative_eq!(cycle.net_work, 0.0, epsilon = 1e-30);
    }

    #[test]
    fn test_cycle_execute_n() {
        let cycle = SzilardCycle::execute_n(300.0, 8.0);
        assert_relative_eq!(cycle.net_work, 0.0, epsilon = 1e-29);
    }

    #[test]
    fn test_fuel_value() {
        let fuel = InformationFuel::new(300.0);
        let v = fuel.fuel_value(1.0);
        assert_relative_eq!(v, BOLTZMANN * 300.0 * LN2, epsilon = 1e-30);
    }

    #[test]
    fn test_bits_for_work() {
        let fuel = InformationFuel::new(300.0);
        let w = BOLTZMANN * 300.0 * LN2 * 5.0;
        let bits = fuel.bits_for_work(w);
        assert_relative_eq!(bits, 5.0, epsilon = 1e-5);
    }

    #[test]
    fn test_temperature_for_work() {
        let w = BOLTZMANN * 300.0 * LN2;
        let t = InformationFuel::temperature_for_work(w, 1.0);
        assert_relative_eq!(t, 300.0, epsilon = 1.0);
    }

    #[test]
    fn test_engine_reset() {
        let mut engine = SzilardEngine::new(300.0);
        engine.extract_bits(10.0);
        engine.reset();
        assert_relative_eq!(engine.bits_consumed, 0.0, epsilon = 1e-10);
        assert_relative_eq!(engine.work_extracted, 0.0, epsilon = 1e-10);
    }
}
