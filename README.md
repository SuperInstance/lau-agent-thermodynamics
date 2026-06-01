# lau-agent-thermodynamics

> Full thermodynamic treatment of agents — Landauer bound, Carnot efficiency, Maxwell's demon, Szilard engine, fluctuation theorems, and optimal learning schedules

## What This Does

Full thermodynamic treatment of agents — Landauer bound, Carnot efficiency, Maxwell's demon, Szilard engine, fluctuation theorems, and optimal learning schedules. Part of the PLATO/LAU ecosystem — a mathematically rigorous framework for building educational agents that learn, teach, and evolve.

## The Key Idea

This crate implements the core abstractions needed for its domain, with a focus on correctness, composability, and conservation guarantees. Every public type is serializable (serde), every algorithm is tested, and every invariant is verified.

## Install

```bash
cargo add lau-agent-thermodynamics
```

## Quick Start

See the API Reference below for complete usage. Key entry points:

```rust
use lau_agent_thermodynamics::*;
// See types and methods below for complete usage
```

## API Reference

```rust
pub struct FluctuationTheorem 
    pub fn new(tau: f64, temperature: f64) -> Self 
    pub fn probability_ratio(&self, sigma: f64) -> f64 
    pub fn log_probability_ratio(&self, sigma: f64) -> f64 
    pub fn probability_of_decrease(&self, delta_s: f64) -> f64 
    pub fn probability_of_rate(&self, sigma: f64, mean_sigma: f64) -> f64 
    pub fn is_allowed_fluctuation(&self, delta_s: f64, confidence: f64) -> bool 
    pub fn resolution_limit(&self) -> f64 
pub struct DetailedFluctuation 
    pub fn new(forward_entropy: f64, reverse_entropy: f64, temperature: f64) -> Self 
    pub fn crooks_ratio(&self, delta_f: f64) -> f64 
    pub fn net_entropy_production(&self) -> f64 
    pub fn verify_on_average(&self) -> bool 
pub struct AgentFluctuation 
    pub fn new(tau: f64, temperature: f64) -> Self 
    pub fn record(&mut self, delta_s: f64) 
    pub fn mean_entropy_production(&self) -> f64 
    pub fn negative_count(&self) -> usize 
    pub fn negative_fraction(&self) -> f64 
    pub fn verify_integral_theorem(&self) -> f64 
    pub fn is_satisfied(&self, tolerance: f64) -> bool 
pub struct BeliefState 
    pub fn new(probabilities: Vec<f64>) -> Self 
    pub fn uniform(n: usize) -> Self 
    pub fn deterministic(n: usize, i: usize) -> Self 
    pub fn shannon_entropy(&self) -> f64 
    pub fn thermodynamic_entropy(&self, _temperature: f64) -> f64 
    pub fn kl_divergence(&self, other: &BeliefState) -> f64 
    pub fn to_vector(&self) -> DVector<f64> 
    pub fn n_states(&self) -> usize 
    pub fn bayesian_update(&self, likelihoods: &[f64]) -> BeliefState 
    pub fn work_to_reduce_entropy(&self, target: &BeliefState, temperature: f64) -> f64 
pub struct EntropyTracker 
    pub fn new(initial: BeliefState, temperature: f64) -> Self 
    pub fn update(&mut self, new_belief: BeliefState, work: f64) 
    pub fn check_second_law(&self, step: usize) -> bool 
    pub fn total_entropy_change(&self) -> f64 
    pub fn total_work(&self) -> f64 
pub struct ThermodynamicState 
    pub fn new(temperature: f64) -> Self 
    pub fn free_energy(&self) -> f64 
    pub fn landauer_cost(&self) -> f64 
    pub fn efficiency(&self) -> f64 
pub struct SzilardEngine 
    pub fn new(temperature: f64) -> Self 
    pub fn work_per_bit(&self) -> f64 
    pub fn extract_one_bit(&mut self) -> f64 
    pub fn extract_bits(&mut self, n: f64) -> f64 
    pub fn net_work(&self) -> f64 
    pub fn efficiency(&self) -> f64 
    pub fn reset(&mut self) 
pub struct SzilardCycle 
    pub fn execute(temperature: f64) -> Self 
    pub fn execute_n(temperature: f64, n_bits: f64) -> Self 
    pub fn verify_second_law(&self) -> bool 
pub struct InformationFuel 
    pub fn new(temperature: f64) -> Self 
    pub fn fuel_value(&self, n_bits: f64) -> f64 
    pub fn bits_for_work(&self, work: f64) -> f64 
    pub fn temperature_for_work(work: f64, n_bits: f64) -> f64 
```

## How It Works

Read the source in `src/` for full implementation details. All algorithms are documented with inline comments explaining the mathematical foundations.

## The Math

This crate implements formal mathematical constructs. See the source documentation for theorem statements and proofs of correctness.

## Testing

**129 tests** covering construction, serialization, correctness properties, edge cases, and composability with other lau-* crates.

## License

MIT
