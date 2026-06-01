//! Optimal Protocol: minimum-dissipation learning schedule.
//!
//! Given a thermodynamic length, compute the optimal control protocol that
//! minimizes dissipation during agent learning.

use crate::thermodynamic_length::{FisherInformation, ThermodynamicPath};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A control protocol specifying the learning rate at each time step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlProtocol {
    /// Time points.
    pub times: Vec<f64>,
    /// Parameter values at each time.
    pub parameters: Vec<DVector<f64>>,
    /// Learning rates (dθ/dt) at each interval.
    pub rates: Vec<f64>,
}

impl ControlProtocol {
    /// Create a protocol from time-parameter pairs.
    pub fn new(times: Vec<f64>, parameters: Vec<DVector<f64>>) -> Self {
        let mut rates = Vec::new();
        for i in 1..times.len() {
            let dt = times[i] - times[i - 1];
            let dtheta = &parameters[i] - &parameters[i - 1];
            if dt > 0.0 {
                rates.push(dtheta.norm() / dt);
            } else {
                rates.push(0.0);
            }
        }
        if rates.len() < times.len() {
            rates.push(0.0);
        }
        Self {
            times,
            parameters,
            rates,
        }
    }

    /// Total duration.
    pub fn duration(&self) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        self.times.last().copied().unwrap_or(0.0) - self.times.first().copied().unwrap_or(0.0)
    }

    /// Number of steps.
    pub fn n_steps(&self) -> usize {
        self.times.len().saturating_sub(1)
    }
}

/// Optimal protocol calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalProtocolBuilder {
    /// Fisher information (can vary along path, but we use constant for now).
    pub fisher: FisherInformation,
    /// Total available time.
    pub total_time: f64,
    /// Number of discretization steps.
    pub n_steps: usize,
}

impl OptimalProtocolBuilder {
    /// Create a new builder.
    pub fn new(fisher: FisherInformation, total_time: f64, n_steps: usize) -> Self {
        Self {
            fisher,
            total_time,
            n_steps,
        }
    }

    /// Build the minimum-dissipation protocol between start and end.
    ///
    /// The optimal protocol distributes time proportional to the Fisher metric:
    /// dt ∝ √(dθ^T G dθ)
    pub fn build(&self, start: &DVector<f64>, end: &DVector<f64>) -> ControlProtocol {
        let fisher = &self.fisher;

        // Compute path segment lengths under Fisher metric
        let mut segment_lengths = Vec::new();
        let _total_length = (end - start).norm(); // Simplified

        for i in 0..self.n_steps {
            let t_start = i as f64 / self.n_steps as f64;
            let t_end = (i + 1) as f64 / self.n_steps as f64;
            let p_start = start + t_start * (end - start);
            let p_end = start + t_end * (end - start);
            let dtheta = &p_end - &p_start;
            let seg_len = fisher.infinitesimal_distance(&dtheta);
            segment_lengths.push(seg_len);
        }

        let total_seg: f64 = segment_lengths.iter().sum();

        // Distribute time proportional to segment length
        let mut times = vec![0.0];
        let mut parameters = vec![start.clone()];
        let mut cum_time = 0.0;

        for (i, &seg_len) in segment_lengths.iter().enumerate() {
            let frac = if total_seg > f64::EPSILON {
                seg_len / total_seg
            } else {
                1.0 / self.n_steps as f64
            };
            cum_time += frac * self.total_time;
            times.push(cum_time);

            let t = (i + 1) as f64 / self.n_steps as f64;
            parameters.push(start + t * (end - start));
        }

        ControlProtocol::new(times, parameters)
    }

    /// Compute minimum dissipation for a given path length and time.
    /// D_min = L² / (2T) where L is thermodynamic length and T is total time.
    pub fn minimum_dissipation(&self, thermodynamic_length: f64) -> f64 {
        if self.total_time < f64::EPSILON {
            return f64::INFINITY;
        }
        thermodynamic_length.powi(2) / (2.0 * self.total_time)
    }

    /// Compute the speed-up factor: how much faster than a uniform protocol.
    pub fn speedup_factor(
        &self,
        optimal_dissipation: f64,
        uniform_dissipation: f64,
    ) -> f64 {
        if optimal_dissipation < f64::EPSILON {
            return 1.0;
        }
        uniform_dissipation / optimal_dissipation
    }
}

/// Protocol comparison utilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolComparison {
    /// Uniform protocol dissipation.
    pub uniform_dissipation: f64,
    /// Optimal protocol dissipation.
    pub optimal_dissipation: f64,
    /// Improvement ratio.
    pub improvement: f64,
}

impl ProtocolComparison {
    /// Compare uniform vs optimal protocol.
    pub fn compare(
        fisher: &FisherInformation,
        start: &DVector<f64>,
        end: &DVector<f64>,
        total_time: f64,
        n_steps: usize,
    ) -> Self {
        // Uniform protocol
        let uniform_path = ThermodynamicPath::linear(start, end, n_steps);
        let uniform_diss = uniform_path.dissipation(fisher, total_time);

        // Optimal protocol
        let builder = OptimalProtocolBuilder::new(fisher.clone(), total_time, n_steps);
        let thermo_length = uniform_path.thermodynamic_length(fisher);
        let optimal_diss = builder.minimum_dissipation(thermo_length);

        Self {
            uniform_dissipation: uniform_diss,
            optimal_dissipation: optimal_diss,
            improvement: if optimal_diss > f64::EPSILON {
                uniform_diss / optimal_diss
            } else {
                1.0
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_control_protocol_duration() {
        let times = vec![0.0, 0.5, 1.0];
        let params = vec![
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![0.5]),
            DVector::from_vec(vec![1.0]),
        ];
        let proto = ControlProtocol::new(times, params);
        assert_relative_eq!(proto.duration(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_control_protocol_steps() {
        let times = vec![0.0, 0.5, 1.0];
        let params = vec![
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![0.5]),
            DVector::from_vec(vec![1.0]),
        ];
        let proto = ControlProtocol::new(times, params);
        assert_eq!(proto.n_steps(), 2);
    }

    #[test]
    fn test_optimal_protocol_build() {
        let fisher = FisherInformation::categorical(&[0.5, 0.5]);
        let builder = OptimalProtocolBuilder::new(fisher, 1.0, 100);
        let start = DVector::from_vec(vec![0.1]);
        let end = DVector::from_vec(vec![0.9]);
        let proto = builder.build(&start, &end);
        assert_eq!(proto.n_steps(), 100);
        assert_relative_eq!(proto.duration(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_minimum_dissipation() {
        let fisher = FisherInformation::categorical(&[0.5, 0.5]);
        let builder = OptimalProtocolBuilder::new(fisher, 1.0, 100);
        let d = builder.minimum_dissipation(1.0);
        assert_relative_eq!(d, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_minimum_dissipation_zero_time() {
        let fisher = FisherInformation::categorical(&[0.5, 0.5]);
        let builder = OptimalProtocolBuilder::new(fisher, 0.0, 100);
        let d = builder.minimum_dissipation(1.0);
        assert!(d.is_infinite());
    }

    #[test]
    fn test_speedup_factor() {
        let fisher = FisherInformation::categorical(&[0.5, 0.5]);
        let builder = OptimalProtocolBuilder::new(fisher, 1.0, 100);
        let sf = builder.speedup_factor(1.0, 2.0);
        assert_relative_eq!(sf, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_protocol_comparison() {
        let fisher = FisherInformation::categorical(&[0.5, 0.5]);
        let start = DVector::from_vec(vec![0.1]);
        let end = DVector::from_vec(vec![0.9]);
        let comp = ProtocolComparison::compare(&fisher, &start, &end, 1.0, 100);
        assert!(comp.improvement >= 1.0);
    }

    #[test]
    fn test_protocol_rates() {
        let times = vec![0.0, 1.0];
        let params = vec![
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![2.0]),
        ];
        let proto = ControlProtocol::new(times, params);
        assert!(proto.rates[0] > 0.0);
    }
}
