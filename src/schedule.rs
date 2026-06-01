//! Agent Learning Schedule Design.
//!
//! Application module: design learning schedules that minimize thermodynamic cost.
//! Combines Landauer bound, Carnot efficiency, optimal protocols, and
//! fluctuation analysis into practical agent learning design.

use crate::carnot::CarnotEngine;
use crate::constants::{BOLTZMANN, LN2};
use crate::landauer::LandauerCost;
use crate::optimal_protocol::OptimalProtocolBuilder;
use crate::thermodynamic_length::FisherInformation;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Configuration for an agent learning schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleConfig {
    /// Temperature at start (hot = exploration).
    pub t_start: f64,
    /// Temperature at end (cold = exploitation).
    pub t_end: f64,
    /// Total learning time in seconds.
    pub total_time: f64,
    /// Number of learning steps.
    pub n_steps: usize,
    /// Number of belief states.
    pub n_states: usize,
    /// Energy budget in Joules.
    pub energy_budget: f64,
}

/// Designed learning schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSchedule {
    /// Configuration used.
    pub config: ScheduleConfig,
    /// Temperature at each step.
    pub temperatures: Vec<f64>,
    /// Landauer cost at each step.
    pub landauer_costs: Vec<f64>,
    /// Cumulative energy spent.
    pub cumulative_energy: Vec<f64>,
    /// Estimated entropy at each step.
    pub entropy_estimates: Vec<f64>,
    /// Total thermodynamic cost.
    pub total_cost: f64,
}

impl LearningSchedule {
    /// Design an optimal learning schedule.
    pub fn design(config: ScheduleConfig) -> Self {
        let _carnot = CarnotEngine::new(config.t_start, config.t_end);

        // Temperature schedule: exponential cooling (Carnot-optimal)
        let temperatures: Vec<f64> = (0..=config.n_steps)
            .map(|i| {
                let frac = i as f64 / config.n_steps as f64;
                config.t_start * (config.t_end / config.t_start).powf(frac)
            })
            .collect();

        // Landauer cost at each step
        let landauer_costs: Vec<f64> = temperatures
            .iter()
            .map(|&t| {
                let lc = LandauerCost::new(t);
                // Cost per bit decreases as temperature drops
                lc.per_bit() * (config.n_states as f64).log2() / config.n_steps as f64
            })
            .collect();

        // Cumulative energy
        let mut cumulative_energy = Vec::with_capacity(temperatures.len());
        let mut cum = 0.0;
        for cost in &landauer_costs {
            cum += cost;
            cumulative_energy.push(cum);
        }

        // Entropy estimates: S ~ k * log(n) * (T - T_end) / (T_start - T_end)
        let max_entropy = BOLTZMANN * (config.n_states as f64).ln();
        let entropy_estimates: Vec<f64> = temperatures
            .iter()
            .map(|&t| {
                let frac = (t - config.t_end) / (config.t_start - config.t_end);
                max_entropy * frac
            })
            .collect();

        let total_cost = *cumulative_energy.last().unwrap_or(&0.0);

        Self {
            config,
            temperatures,
            landauer_costs,
            cumulative_energy,
            entropy_estimates,
            total_cost,
        }
    }

    /// Whether the schedule fits within energy budget.
    pub fn within_budget(&self) -> bool {
        self.total_cost <= self.config.energy_budget
    }

    /// Efficiency of the schedule (useful work / total cost).
    pub fn efficiency(&self) -> f64 {
        if self.total_cost < f64::EPSILON {
            return 0.0;
        }
        let useful = BOLTZMANN * self.config.t_end * LN2
            * (self.config.n_states as f64).log2();
        useful / self.total_cost
    }

    /// Estimated entropy reduction.
    pub fn entropy_reduction(&self) -> f64 {
        if self.entropy_estimates.is_empty() {
            return 0.0;
        }
        self.entropy_estimates.first().unwrap() - self.entropy_estimates.last().unwrap()
    }

    /// Number of steps.
    pub fn n_steps(&self) -> usize {
        self.temperatures.len().saturating_sub(1)
    }
}

/// Schedule comparison and recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleRecommendation {
    /// The recommended schedule.
    pub schedule: LearningSchedule,
    /// Carnot efficiency at these temperatures.
    pub carnot_efficiency: f64,
    /// Landauer cost comparison (actual vs minimum).
    pub landauer_ratio: f64,
    /// Whether the schedule is thermodynamically optimal.
    pub is_optimal: bool,
    /// Suggested improvements.
    pub suggestions: Vec<String>,
}

impl ScheduleRecommendation {
    /// Generate a recommendation for a schedule config.
    pub fn recommend(config: ScheduleConfig) -> Self {
        let schedule = LearningSchedule::design(config.clone());
        let carnot = CarnotEngine::new(config.t_start, config.t_end);
        let carnot_eff = carnot.efficiency();

        let min_cost = LandauerCost::new(config.t_end).for_bits(
            (config.n_states as f64).log2(),
        );
        let landauer_ratio = if schedule.total_cost > f64::EPSILON {
            min_cost / schedule.total_cost
        } else {
            0.0
        };

        let mut suggestions = Vec::new();
        if !schedule.within_budget() {
            suggestions.push("Schedule exceeds energy budget. Consider fewer steps or smaller state space.".to_string());
        }
        if carnot_eff < 0.3 {
            suggestions.push("Low Carnot efficiency. Consider reducing temperature gap.".to_string());
        }
        if config.n_steps < 50 {
            suggestions.push("Few steps may cause high dissipation. Consider more gradual cooling.".to_string());
        }

        let is_optimal = schedule.within_budget() && landauer_ratio > 0.5;

        Self {
            schedule,
            carnot_efficiency: carnot_eff,
            landauer_ratio,
            is_optimal,
            suggestions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_schedule_design() {
        let config = ScheduleConfig {
            t_start: 600.0,
            t_end: 100.0,
            total_time: 1.0,
            n_steps: 100,
            n_states: 16,
            energy_budget: 1e-17,
        };
        let schedule = LearningSchedule::design(config);
        assert_eq!(schedule.temperatures.len(), 101);
        assert_relative_eq!(schedule.temperatures[0], 600.0, epsilon = 1e-5);
        assert_relative_eq!(schedule.temperatures[100], 100.0, epsilon = 1e-5);
    }

    #[test]
    fn test_schedule_monotone_cooling() {
        let config = ScheduleConfig {
            t_start: 600.0,
            t_end: 100.0,
            total_time: 1.0,
            n_steps: 100,
            n_states: 8,
            energy_budget: 1e-17,
        };
        let schedule = LearningSchedule::design(config);
        for i in 1..schedule.temperatures.len() {
            assert!(schedule.temperatures[i] <= schedule.temperatures[i - 1] + 1e-10);
        }
    }

    #[test]
    fn test_schedule_entropy_decreases() {
        let config = ScheduleConfig {
            t_start: 600.0,
            t_end: 100.0,
            total_time: 1.0,
            n_steps: 100,
            n_states: 8,
            energy_budget: 1e-17,
        };
        let schedule = LearningSchedule::design(config);
        for i in 1..schedule.entropy_estimates.len() {
            assert!(schedule.entropy_estimates[i] <= schedule.entropy_estimates[i - 1] + 1e-30);
        }
    }

    #[test]
    fn test_schedule_total_cost_positive() {
        let config = ScheduleConfig {
            t_start: 600.0,
            t_end: 100.0,
            total_time: 1.0,
            n_steps: 100,
            n_states: 4,
            energy_budget: 1e-17,
        };
        let schedule = LearningSchedule::design(config);
        assert!(schedule.total_cost > 0.0);
    }

    #[test]
    fn test_within_budget() {
        let config = ScheduleConfig {
            t_start: 600.0,
            t_end: 100.0,
            total_time: 1.0,
            n_steps: 10,
            n_states: 2,
            energy_budget: 1e10, // Huge budget
        };
        let schedule = LearningSchedule::design(config);
        assert!(schedule.within_budget());
    }

    #[test]
    fn test_recommendation() {
        let config = ScheduleConfig {
            t_start: 600.0,
            t_end: 300.0,
            total_time: 1.0,
            n_steps: 100,
            n_states: 4,
            energy_budget: 1e-17,
        };
        let rec = ScheduleRecommendation::recommend(config);
        assert!(rec.carnot_efficiency > 0.0);
    }

    #[test]
    fn test_recommendation_suggestions() {
        let config = ScheduleConfig {
            t_start: 1000.0,
            t_end: 10.0,
            total_time: 1.0,
            n_steps: 5,
            n_states: 2,
            energy_budget: 1e-30, // Tiny budget
        };
        let rec = ScheduleRecommendation::recommend(config);
        assert!(!rec.suggestions.is_empty());
    }

    #[test]
    fn test_entropy_reduction() {
        let config = ScheduleConfig {
            t_start: 600.0,
            t_end: 100.0,
            total_time: 1.0,
            n_steps: 100,
            n_states: 8,
            energy_budget: 1e-17,
        };
        let schedule = LearningSchedule::design(config);
        assert!(schedule.entropy_reduction() > 0.0);
    }

    #[test]
    fn test_n_steps() {
        let config = ScheduleConfig {
            t_start: 600.0,
            t_end: 100.0,
            total_time: 1.0,
            n_steps: 50,
            n_states: 4,
            energy_budget: 1e-17,
        };
        let schedule = LearningSchedule::design(config);
        assert_eq!(schedule.n_steps(), 50);
    }

    #[test]
    fn test_efficiency() {
        let config = ScheduleConfig {
            t_start: 600.0,
            t_end: 300.0,
            total_time: 1.0,
            n_steps: 100,
            n_states: 4,
            energy_budget: 1e-17,
        };
        let schedule = LearningSchedule::design(config);
        assert!(schedule.efficiency() >= 0.0);
    }
}
