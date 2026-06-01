# lau-agent-thermodynamics

**The thermodynamics of computation** — Landauer's bound, Carnot engines for learning, Maxwell's demon, Szilard engines, fluctuation theorems, Jarzynski equality, thermodynamic length, and optimal learning schedules, all in pure Rust.

[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-129-green.svg)](#testing)

---

## Overview

Every computation costs energy. This crate makes that cost precise by applying the full machinery of thermodynamics to agent systems. From the fundamental Landauer bound (erasing one bit costs at least kT ln 2) through Carnot efficiency limits on learning, to modern fluctuation theorems that allow transient second-law violations in small systems.

**Core thesis:** an agent's belief state is a thermodynamic system. Belief updates are computations. Computations dissipate heat. The laws of thermodynamics set inviolable bounds on what any agent can achieve with finite energy.

| Law | Agent Interpretation | Module |
|---|---|---|
| Zeroth | Agents in thermal equilibrium share beliefs | `constants` |
| First | Energy in = computation + dissipation | `first_law` |
| Second | Entropy of beliefs never decreases without work | `second_law` |
| Third | As T → 0, agent converges to optimal policy | `third_law` |
| Landauer | Erasing 1 bit costs ≥ kT ln 2 | `landauer` |
| Carnot | Learning efficiency ≤ 1 − T_cold/T_hot | `carnot` |
| Maxwell's Demon | Measurement costs resolve the paradox | `maxwell_demon` |
| Szilard Engine | 1 bit of information = kT ln 2 of work | `szilard` |
| Fluctuation | P(σ)/P(−σ) = e^(στ) for small agents | `fluctuation` |
| Jarzynski | ⟨e^(−βW)⟩ = e^(−βΔF) — nonequilibrium equality | `jarzynski` |

---

## Quick Start

```toml
# Cargo.toml
[dependencies]
lau-agent-thermodynamics = "0.1"
```

```rust
use lau_agent_thermodynamics::*;

// --- Landauer's bound at room temperature ---
let cost = LandauerCost::new(300.0);
println!("Erasing 1 bit: {:.3e} J", cost.per_bit());
println!("Erasing 1 byte: {:.3e} J", cost.for_bits(8.0));

// --- Belief state entropy ---
let uniform = BeliefState::uniform(4);    // 4 equally likely states
let peaked = BeliefState::deterministic(4, 0); // certain about state 0
println!("Uniform entropy: {:.2} bits", uniform.shannon_entropy()); // 2.0
println!("Peaked entropy:  {:.2} bits", peaked.shannon_entropy());  // 0.0

// --- Work required to learn (reduce entropy) ---
let work = uniform.work_to_reduce_entropy(&peaked, 300.0);
println!("Work to learn: {:.3e} J", work);

// --- Carnot engine for learning ---
let engine = CarnotEngine::new(600.0, 300.0);
println!("Max learning efficiency: {:.1}%", engine.efficiency() * 100.0);

// --- Szilard engine: extract work from information ---
let mut sz = SzilardEngine::new(300.0);
let work = sz.extract_one_bit(); // kT ln 2 of extractable work
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│           Agent as Thermodynamic System                   │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ First Law    │  │ Second Law   │  │ Third Law    │     │
│  │ E_in = E_comp│  │ ΔS ≥ 0      │  │ S → 0 as     │     │
│  │   + E_diss   │  │  (no work)  │  │   T → 0      │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │             │
│  ┌──────▼─────────────────▼─────────────────▼──────┐     │
│  │           ThermodynamicState                      │     │
│  │  T, U, S, belief_bits, W, Q                      │     │
│  │  F = U − TS (free energy)                        │     │
│  └──────────────────┬───────────────────────────────┘     │
│                     │                                    │
│  ┌──────────────────▼──────────────────────────────┐     │
│  │  Fundamental Bounds                               │     │
│  │  ┌───────────┐ ┌───────────┐ ┌──────────────┐   │     │
│  │  │ Landauer   │ │ Szilard    │ │ Maxwell's     │   │     │
│  │  │ kT ln(2)   │ │ Engine     │ │ Demon         │   │     │
│  │  │ per bit    │ │ kT ln(2)   │ │ (measurement  │   │     │
│  │  └───────────┘ │ per bit    │ │  cost = erase) │   │     │
│  │                └───────────┘ └──────────────┘   │     │
│  └──────────────────┬──────────────────────────────┘     │
│                     │                                    │
│  ┌──────────────────▼──────────────────────────────┐     │
│  │  Optimization & Nonequilibrium                    │     │
│  │  ┌───────────┐ ┌───────────┐ ┌──────────────┐   │     │
│  │  │ Carnot     │ │ Optimal    │ │ Schedule      │   │     │
│  │  │ Engine     │ │ Protocol   │ │ (annealing)   │   │     │
│  │  │ (learning) │ │ (min diss) │ │               │   │     │
│  │  └───────────┘ └───────────┘ └──────────────┘   │     │
│  │  ┌───────────┐ ┌───────────┐ ┌──────────────┐   │     │
│  │  │ Thermo     │ │ Jarzynski  │ │ Fluctuation   │   │     │
│  │  │ Length     │ │ Equality   │ │ Theorem       │   │     │
│  │  │ (Fisher)   │ │ ⟨e^-βW⟩=...│ │ P(σ)/P(-σ)   │   │     │
│  │  └───────────┘ └───────────┘ └──────────────┘   │     │
│  └──────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

---

## Modules in Detail

### Constants (`constants`)

Physical constants and the core `ThermodynamicState` type.

```rust
// Physical constants
BOLTZMANN  // k_B = 1.38064852 × 10⁻²³ J/K
LN2        // ln(2) = 0.693147...

// Thermodynamic state of an agent
let state = ThermodynamicState::new(300.0); // at 300K
state.free_energy();     // F = U − TS
state.landauer_cost();   // kT ln(2) × belief_bits
state.efficiency();      // (W − landauer) / W

// Serializable with serde
let json = serde_json::to_string(&state).unwrap();
```

### First Law (`first_law`)

Energy conservation for agent computations: E_input = E_computation + E_dissipation.

```rust
// Single computation step
let budget = EnergyBudget::new(1e-20, 0.5e-20);
budget.verify_conservation();     // true
budget.computation_fraction();    // 0.5

// Minimum energy for n bits at temperature T
let e_min = EnergyBudget::minimum_energy(8.0, 300.0);

// Track cumulative energy over a computation sequence
let mut accounting = FirstLawAccounting::new();
accounting.record(EnergyBudget::new(1e-20, 0.5e-20));
accounting.record(EnergyBudget::new(2e-20, 1.0e-20));
accounting.total_dissipation_fraction(); // fraction wasted as heat
accounting.verify_conservation();        // all steps conserve energy

// Energy audit of a multi-step computation
let audit = EnergyAudit::from_steps(&[
    (1e-20, 0.5e-20),  // (input, computation)
    (2e-20, 1.5e-20),
]);
audit.total_waste_ratio(); // overall waste fraction
```

### Second Law (`second_law`)

Entropy of agent beliefs never decreases without external work.

```rust
// Belief states as probability distributions
let b = BeliefState::uniform(4);        // [0.25, 0.25, 0.25, 0.25]
b.shannon_entropy();                     // 2.0 bits
b.thermodynamic_entropy(300.0);          // S = k_B × H × ln(2)
b.kl_divergence(&other);                // D(b || other)

// Bayesian belief updates
let posterior = prior.bayesian_update(&likelihoods);

// Work required to reduce entropy
let w = uniform.work_to_reduce_entropy(&peaked, 300.0);

// Track entropy changes and verify second law
let mut tracker = EntropyTracker::new(BeliefState::uniform(4), 300.0);
tracker.update(peaked.clone(), sufficient_work);
tracker.check_second_law(1);    // true if work >= required
tracker.total_entropy_change();  // net ΔS
```

### Third Law (`third_law`)

As temperature approaches absolute zero, the agent converges to its ground state (optimal policy) with zero entropy.

```rust
// Ground state: the deterministic optimal policy
let ground = GroundState::new(optimal_state: 0, n_states: 4, energy_gap: 1e-20);
ground.residual_entropy(300.0);  // small but nonzero
ground.residual_entropy(0.001);  // essentially zero

// Cooled agent: temperature-dependent belief
let agent = CooledAgent::new(ground, 300.0);
agent.belief();              // Boltzmann distribution
agent.entropy();             // temperature-dependent
agent.is_ground_state(tol);  // effectively optimal?

// Cooling schedule for simulated annealing
let schedule = CoolingSchedule::new(1000.0, 0.1, 100, CoolingType::Exponential);
let temps: Vec<f64> = schedule.iter().collect(); // 100 temperature steps

// Verify third law: entropy → 0 as T → 0
let s_low = ground.residual_entropy(0.001);
let s_high = ground.residual_entropy(1000.0);
assert!(s_low < s_high);
```

### Landauer's Principle (`landauer`)

The fundamental lower bound on computation: erasing one bit of information requires at least kT ln 2 of energy dissipated as heat.

```rust
let cost = LandauerCost::new(300.0);
cost.per_bit();            // kT ln 2 ≈ 2.87 × 10⁻²¹ J
cost.for_bits(8.0);        // cost for 1 byte
cost.for_belief(2.0);      // cost for belief with 2 bits of entropy
cost.bits_from_energy(1e-20); // how many bits can we erase?

// Erasure record: track actual vs minimum
let record = ErasureRecord::new(1.0, cost.per_bit() * 2.0, 300.0);
record.satisfies_bound();  // true (used ≥ minimum)
record.efficiency();       // 0.5 (used 2× the minimum)
record.excess_energy();    // energy above the bound
record.heat_generated();   // all energy becomes heat

// Recover temperature from cost and bit count
let t = LandauerCost::temperature_from_cost(cost.per_bit(), 1.0); // ≈ 300
```

### Carnot Engine (`carnot`)

Maximum efficiency for converting energy into learning (analogous to converting heat into work).

```rust
let engine = CarnotEngine::new(t_hot: 600.0, t_cold: 300.0);
engine.efficiency();           // 1 − T_c/T_h = 0.5 (50%)
engine.max_work(100.0);        // 50 J from 100 J input
engine.heat_rejected(100.0);   // 50 J waste heat
engine.cop();                  // coefficient of performance (heat pump mode)

// Learning engine: maps Carnot cycle to agent learning
let learner = LearningEngine::new(600.0, 300.0, 4); // 4 belief states
learner.max_info_gain();           // bits learnable per cycle
learner.cost_per_bit();            // energy cost per bit learned
learner.optimal_learning_rate(budget, total_bits); // ∈ [0, 1]
```

### Maxwell's Demon (`maxwell_demon`)

Resolution of the Maxwell's demon paradox: the cost of measurement and memory erasure exactly balances the work extracted.

```rust
let demon = MaxwellDemon::new(300.0);
demon.work_extracted_per_bit();     // kT ln 2
demon.measurement_cost();           // kT ln 2 (same!)
demon.total_cost_per_cycle();       // 2 × kT ln 2
demon.net_gain();                   // 0 (no free lunch)

// Full demon cycle: measure → extract → erase
let cycle = DemonCycle::execute(300.0);
cycle.work_extracted;     // +kT ln 2
cycle.measurement_cost;   // kT ln 2
cycle.erasure_cost;       // kT ln 2
cycle.net_work;           // 0 (second law upheld)

// Multiple bits
let n_cycle = DemonCycle::execute_n(300.0, 8);
n_cycle.verify_second_law();  // always true
```

### Szilard Engine (`szilard`)

The Szilard engine extracts kT ln 2 of work from 1 bit of information. With memory reset, net work is zero.

```rust
let mut engine = SzilardEngine::new(300.0);
engine.work_per_bit();          // kT ln 2
let w = engine.extract_one_bit();
engine.bits_consumed;           // 1.0

engine.include_reset_cost = true;
engine.extract_bits(5.0);
engine.net_work();              // ≈ 0 (reset costs exactly what's extracted)

// Full cycle analysis
let cycle = SzilardCycle::execute(300.0);
cycle.work_extracted;  // kT ln 2
cycle.reset_cost;      // kT ln 2
cycle.net_work;        // 0
cycle.verify_second_law(); // true

// Information as fuel
let fuel = InformationFuel::new(300.0);
fuel.fuel_value(8.0);           // energy content of 8 bits
fuel.bits_for_work(1e-20);      // bits needed for given work
InformationFuel::temperature_for_work(w, 1.0); // inverse calculation
```

### Thermodynamic Length (`thermodynamic_length`)

The Fisher information metric defines a Riemannian manifold over belief distributions. Thermodynamic length measures the minimum-dissipation path between two beliefs.

```rust
// Fisher information matrix for a categorical distribution
let fisher = FisherInformation::categorical(&[0.25, 0.25, 0.25, 0.25]);
fisher.matrix;                    // 3×3 matrix (n-1 free parameters)
fisher.inverse();                 // Cramér-Rao bound
fisher.infinitesimal_distance(&dθ); // ds = √(dθ^T G dθ)

// Fisher information for Gaussian
let fisher_g = FisherInformation::gaussian(3, 1.0); // identity for unit variance

// Path between beliefs on the statistical manifold
let path = ThermodynamicPath::linear(&start, &end, 100);
path.thermodynamic_length(&fisher);  // total geodesic length
path.dissipation(&fisher, 1.0);      // ∫ (dθ/dt)^T G (dθ/dt) dt

// Geodesic (Fisher-Rao) distance
let dist = GeodesicDistance::categorical(&p, &q);
dist.distance;  // 2 × arccos(Σ √(pᵢqᵢ))

let dist_g = GeodesicDistance::gaussian(&μ_p, &μ_q, variance);
```

### Optimal Protocol (`optimal_protocol`)

Minimum-dissipation control protocols for agent learning, using the Fisher metric to distribute time optimally.

```rust
let builder = OptimalProtocolBuilder::new(fisher, total_time: 1.0, n_steps: 50);
let protocol = builder.build(&start, &end);

protocol.times;       // time points (nonuniform)
protocol.parameters;  // parameter values at each step
protocol.rates;       // learning rates dθ/dt
protocol.duration();
protocol.n_steps();

// Optimality: time distributed proportional to √(Fisher metric)
// Slower in high-curvature regions, faster in flat regions
```

### Jarzynski Equality (`jarzynski`)

The remarkable nonequilibrium equality: ⟨e^(−βW)⟩ = e^(−βΔF). Allows computing equilibrium free energy differences from nonequilibrium work measurements.

```rust
let je = JarzynskiEquality::new(300.0);
je.beta();                         // 1/(kT)
je.work_to_exponential(w);         // e^(−βW)
je.free_energy_from_work(&[w1, w2, w3]); // ΔF from work samples

// Nonequilibrium trajectory
let mut traj = NonequilibriumTrajectory::new(300.0, initial_free_energy);
traj.add_work_step(w1);
traj.add_work_step(w2);
traj.total_work();              // cumulative
traj.jarzynski_estimator();     // ⟨e^(−βW)⟩⁻¹ → ΔF estimate
traj.convergence_check(tol);    // has the estimate stabilized?

// Work distribution from repeated experiments
let dist = WorkDistribution::from_samples(&works);
dist.mean();
dist.variance();
dist.jarzynski_free_energy();
dist.crooks_intersection(&reverse_dist); // ΔF from forward/reverse crossing
```

### Fluctuation Theorem (`fluctuation`)

For small agents, entropy can temporarily decrease — but exponentially rarely. P(σ)/P(−σ) = e^(στ).

```rust
let ft = FluctuationTheorem::new(tau: 1.0, temperature: 300.0);
ft.probability_ratio(1.0);          // P(σ=1) / P(σ=−1) = e
ft.probability_of_decrease(-1e-22); // P(ΔS < 0) — small but nonzero
ft.resolution_limit();              // k_B / τ

// Detailed fluctuation (Crooks' theorem)
let df = DetailedFluctuation::new(forward_entropy: 1e-20, reverse_entropy: 0.0, temperature: 300.0);
df.crooks_ratio(delta_f: 0.0);     // P_F(W) / P_R(−W)
df.net_entropy_production();

// Agent-scale: collect statistics from many trajectories
let mut af = AgentFluctuation::new(1.0, 300.0);
af.record(1e-22);
af.record(-0.5e-22); // second law violation!
af.negative_fraction();          // fraction of violations
af.verify_integral_theorem();    // ⟨e^(−ΔS/k)⟩ ≈ 1
af.is_satisfied(tolerance: 0.1);
```

### Learning Schedule (`schedule`)

Optimal annealing schedules for agent learning, respecting energy budgets and thermodynamic constraints.

```rust
let config = ScheduleConfig {
    t_start: 600.0,
    t_end: 100.0,
    total_time: 1.0,
    n_steps: 100,
    n_states: 8,
    energy_budget: 1e-17,
};

let schedule = LearningSchedule::design(config);
schedule.temperatures;         // 101 temperature steps
schedule.entropy_estimates;    // entropy at each step
schedule.total_cost;           // cumulative energy cost
schedule.within_budget();      // stays under energy_budget
schedule.entropy_reduction();  // total bits learned
schedule.efficiency();         // cost-effectiveness

// Get recommendations
let rec = ScheduleRecommendation::recommend(config);
rec.carnot_efficiency;    // theoretical maximum
rec.estimated_cost;       // predicted energy use
rec.suggestions;          // optimization advice
```

---

## Testing

129 tests covering all modules:

```bash
cargo test
```

Test categories:
- **Constants** — Boltzmann constant, state construction, free energy, serialization
- **First Law** — conservation verification, budget tracking, cumulative accounting
- **Second Law** — entropy of beliefs, KL divergence, Bayesian updates, violation detection
- **Third Law** — ground state entropy, cooling, Boltzmann distributions, residual entropy
- **Landauer** — per-bit cost, erasure records, bound satisfaction, efficiency
- **Carnot** — efficiency, max work, COP, learning engine, cost per bit
- **Maxwell's Demon** — work extraction, measurement cost, cycle analysis, second law verification
- **Szilard Engine** — bit extraction, reset costs, information fuel, cycle net work
- **Thermodynamic Length** — Fisher matrix, geodesic distance, path dissipation, symmetry
- **Optimal Protocol** — protocol construction, duration, step count
- **Jarzynski** — free energy estimation, work distributions, Crooks intersection, convergence
- **Fluctuation Theorem** — probability ratios, negative entropy fraction, integral theorem, resolution limits
- **Schedule** — monotone cooling, entropy decrease, budget constraints, recommendations

---

## Mathematical Background

### Landauer's Principle

Erasing one bit of information requires dissipating at least:

> E_min = k_B T ln(2) ≈ 2.87 × 10⁻²¹ J at room temperature

This is not a technological limitation — it's a fundamental physical law.

### Carnot Efficiency for Learning

A learning agent is modeled as a heat engine between a "hot" reservoir (prior, high entropy) and a "cold" reservoir (posterior, low entropy):

> η_Carnot = 1 − T_cold / T_hot

No learning algorithm can exceed this efficiency.

### Jarzynski Equality

For a system driven from equilibrium through a protocol:

> ⟨e^(−βW)⟩ = e^(−βΔF)

This remarkable identity holds for **arbitrary** nonequilibrium processes. It allows computing equilibrium free energy differences from ensembles of nonequilibrium work measurements.

### Fluctuation Theorem

For small systems over short times:

> P(σ) / P(−σ) = e^(στ)

where σ is the entropy production rate and τ is the observation time. This allows transient second-law violations, but they become exponentially rare as τ increases.

### Thermodynamic Length

The Fisher information metric G defines a Riemannian structure on the space of probability distributions. The thermodynamic length between distributions p and q is:

> L(p, q) = min_path ∫ √(dθ^T G(θ) dθ)

The minimum-dissipation protocol distributes time proportionally to √(dθ^T G dθ).

---

## Dependencies

| Crate | Purpose |
|---|---|
| `nalgebra` | Linear algebra (vectors, matrices) with serde support |
| `serde` / `serde_json` | Serialization of all state types |
| `approx` (dev) | Floating-point assertions in tests |

---

## License

MIT
