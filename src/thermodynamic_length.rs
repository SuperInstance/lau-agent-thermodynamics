//! Thermodynamic Length: geodesic distance on statistical manifold.
//!
//! The Fisher information metric defines a Riemannian manifold over belief
//! distributions. Thermodynamic length measures the minimum-dissipation path
//! between two beliefs.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Fisher information matrix for a belief state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherInformation {
    /// The Fisher information matrix.
    pub matrix: DMatrix<f64>,
    /// Dimension (number of parameters).
    pub dim: usize,
}

impl FisherInformation {
    /// Compute Fisher information for a categorical distribution.
    /// For p = (p_1, ..., p_n), the Fisher metric is g_{ij} = δ_{ij}/p_i + 1/p_n
    /// where p_n is the "free" parameter.
    pub fn categorical(probabilities: &[f64]) -> Self {
        let n = probabilities.len();
        // Free parameters: first n-1
        let k = n - 1;
        let p_n = probabilities.last().copied().unwrap_or(0.5);

        let mut g = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    g[(i, j)] = 1.0 / probabilities[i] + 1.0 / p_n;
                } else {
                    g[(i, j)] = 1.0 / p_n;
                }
            }
        }

        Self { matrix: g, dim: k }
    }

    /// Compute Fisher information for a Gaussian distribution.
    pub fn gaussian(n_params: usize, variance: f64) -> Self {
        let mut g = DMatrix::zeros(n_params, n_params);
        for i in 0..n_params {
            g[(i, i)] = 1.0 / variance;
        }
        Self { matrix: g, dim: n_params }
    }

    /// Inverse of the Fisher information matrix.
    pub fn inverse(&self) -> DMatrix<f64> {
        self.matrix.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(self.dim, self.dim))
    }

    /// Fisher-Rao distance between two nearby distributions.
    /// ds² = dθ^T * G * dθ
    pub fn infinitesimal_distance(&self, dtheta: &DVector<f64>) -> f64 {
        let quad = dtheta.transpose() * &self.matrix * dtheta;
        quad[(0, 0)].sqrt()
    }
}

/// Path between two belief states on the statistical manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicPath {
    /// Waypoints as parameter vectors (free parameters only).
    pub waypoints: Vec<DVector<f64>>,
}

impl ThermodynamicPath {
    /// Create a path from waypoints.
    pub fn new(waypoints: Vec<DVector<f64>>) -> Self {
        Self { waypoints }
    }

    /// Linear interpolation path between start and end.
    pub fn linear(start: &DVector<f64>, end: &DVector<f64>, n_steps: usize) -> Self {
        let mut waypoints = Vec::new();
        for i in 0..=n_steps {
            let t = i as f64 / n_steps as f64;
            let point = start + t * (end - start);
            waypoints.push(point);
        }
        Self { waypoints }
    }

    /// Compute the thermodynamic length of this path.
    /// Sum of infinitesimal distances weighted by Fisher metric.
    pub fn thermodynamic_length(&self, fisher: &FisherInformation) -> f64 {
        let mut total = 0.0;
        for i in 1..self.waypoints.len() {
            let dtheta = &self.waypoints[i] - &self.waypoints[i - 1];
            total += fisher.infinitesimal_distance(&dtheta);
        }
        total
    }

    /// Compute the dissipation integral for this path.
    /// Dissipation = ∫ (dθ/dt)^T * G * (dθ/dt) dt
    pub fn dissipation(&self, fisher: &FisherInformation, total_time: f64) -> f64 {
        if self.waypoints.len() < 2 {
            return 0.0;
        }
        let dt = total_time / (self.waypoints.len() - 1) as f64;
        let mut total = 0.0;
        for i in 1..self.waypoints.len() {
            let dtheta = &self.waypoints[i] - &self.waypoints[i - 1];
            let velocity = dtheta / dt;
            let quad = velocity.transpose() * &fisher.matrix * &velocity;
            total += quad[(0, 0)] * dt;
        }
        total
    }

    /// Number of waypoints.
    pub fn len(&self) -> usize {
        self.waypoints.len()
    }

    /// Whether the path is empty.
    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }
}

/// Geodesic distance between two distributions (lower bound on thermodynamic length).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeodesicDistance {
    /// Distance value.
    pub distance: f64,
    /// Starting distribution parameters.
    pub start: DVector<f64>,
    /// Ending distribution parameters.
    pub end: DVector<f64>,
}

impl GeodesicDistance {
    /// Compute the Fisher-Rao distance between two categorical distributions.
    /// For categorical distributions, this involves the Bhattacharyya angle.
    pub fn categorical(p: &[f64], q: &[f64]) -> Self {
        let start = DVector::from_vec(p.to_vec());
        let end = DVector::from_vec(q.to_vec());

        // Fisher-Rao distance for categorical: 2 * arccos(Σ√(p_i * q_i))
        let bc: f64 = p
            .iter()
            .zip(q.iter())
            .map(|(pi, qi)| (pi * qi).sqrt())
            .sum();

        let distance = 2.0 * bc.acos().max(0.0);

        Self {
            distance,
            start,
            end,
        }
    }

    /// Compute for Gaussian distributions with same variance.
    pub fn gaussian(mean_p: &DVector<f64>, mean_q: &DVector<f64>, variance: f64) -> Self {
        let diff = mean_q - mean_p;
        let dist_sq = diff.dot(&diff) / variance;
        Self {
            distance: dist_sq.sqrt(),
            start: mean_p.clone(),
            end: mean_q.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fisher_categorical_identity() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let fi = FisherInformation::categorical(&probs);
        assert_eq!(fi.dim, 3);
        // All diagonal elements should be equal for uniform
        assert_relative_eq!(fi.matrix[(0, 0)], fi.matrix[(1, 1)], epsilon = 1e-10);
    }

    #[test]
    fn test_fisher_infinitesimal_distance() {
        let probs = vec![0.5, 0.5];
        let fi = FisherInformation::categorical(&probs);
        let dtheta = DVector::from_vec(vec![0.01]);
        let d = fi.infinitesimal_distance(&dtheta);
        assert!(d > 0.0);
    }

    #[test]
    fn test_fisher_inverse() {
        let probs = vec![0.3, 0.3, 0.4];
        let fi = FisherInformation::categorical(&probs);
        let inv = fi.inverse();
        assert_eq!(inv.nrows(), 2);
    }

    #[test]
    fn test_geodesic_same_distribution() {
        let p = vec![0.5, 0.5];
        let dist = GeodesicDistance::categorical(&p, &p);
        assert_relative_eq!(dist.distance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_geodesic_orthogonal_distributions() {
        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        let dist = GeodesicDistance::categorical(&p, &q);
        assert!(dist.distance > 0.0);
    }

    #[test]
    fn test_linear_path_length() {
        let start = DVector::from_vec(vec![0.1]);
        let end = DVector::from_vec(vec![0.9]);
        let path = ThermodynamicPath::linear(&start, &end, 100);
        assert_eq!(path.len(), 101);
    }

    #[test]
    fn test_path_thermodynamic_length() {
        let probs = vec![0.5, 0.5];
        let fi = FisherInformation::categorical(&probs);
        let start = DVector::from_vec(vec![0.1]);
        let end = DVector::from_vec(vec![0.9]);
        let path = ThermodynamicPath::linear(&start, &end, 1000);
        let length = path.thermodynamic_length(&fi);
        assert!(length > 0.0);
    }

    #[test]
    fn test_path_dissipation() {
        let probs = vec![0.5, 0.5];
        let fi = FisherInformation::categorical(&probs);
        let start = DVector::from_vec(vec![0.1]);
        let end = DVector::from_vec(vec![0.9]);
        let path = ThermodynamicPath::linear(&start, &end, 100);
        let diss = path.dissipation(&fi, 1.0);
        assert!(diss > 0.0);
    }

    #[test]
    fn test_geodesic_gaussian() {
        let p = DVector::from_vec(vec![0.0, 0.0]);
        let q = DVector::from_vec(vec![1.0, 0.0]);
        let dist = GeodesicDistance::gaussian(&p, &q, 1.0);
        assert_relative_eq!(dist.distance, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fisher_gaussian() {
        let fi = FisherInformation::gaussian(3, 1.0);
        assert_eq!(fi.dim, 3);
        // For variance=1, Fisher matrix = identity
        for i in 0..3 {
            assert_relative_eq!(fi.matrix[(i, i)], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_geodesic_symmetric() {
        let p = vec![0.3, 0.7];
        let q = vec![0.6, 0.4];
        let d1 = GeodesicDistance::categorical(&p, &q);
        let d2 = GeodesicDistance::categorical(&q, &p);
        assert_relative_eq!(d1.distance, d2.distance, epsilon = 1e-10);
    }
}
