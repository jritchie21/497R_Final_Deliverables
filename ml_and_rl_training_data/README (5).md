# Orbital Debris Collection Training Data Generator

## Overview

This repository contains tools for generating large-scale training datasets for machine learning and reinforcement learning models focused on orbital debris collection mission planning. The system generates random Low Earth Orbit (LEO) debris scenarios, solves them optimally using mixed-integer linear programming (Gurobi), and outputs structured training data suitable for supervised learning or imitation learning.

## Files Description

### Core Scripts

**`generate_training_data.py`**
- Main data generation script that creates randomized debris collection scenarios
- Generates 10,000+ scenarios with varying debris configurations, orbital parameters, and fuel budgets
- Solves each scenario optimally using Gurobi optimization
- Outputs structured JSON Lines format for easy loading and processing
- Configurable to vary or fix the interceptor starting orbit

**`max_hohmann.py`**
- Computes upper-bound transfer costs between circular orbits
- Calculates delta-v (fuel) and time costs for orbital maneuvers including:
  - Altitude changes (one-tangent transfers)
  - Inclination changes
  - RAAN (Right Ascension of Ascending Node) changes
  - Phase changes for rendezvous
- Used by the optimizer to build cost matrices for path planning

### Generated Data Files

**`debris_training_data_test.jsonl`**
- Small test dataset (10 scenarios) for validation
- Used to verify the data generation pipeline works correctly

**`debris_training_data_full.jsonl`**
- Full production dataset (10,000 scenarios)
- Each line contains one complete scenario with:
  - Interceptor starting orbit parameters
  - Debris field (10-20 objects per scenario)
  - Fuel budget constraint
  - Optimal visiting sequence (ground truth labels)
  - Total value collected and fuel consumed

## How It Works

### Data Generation Pipeline

1. **Scenario Generation**
   - Randomly generates LEO orbits (200-1000 km altitude)
   - Creates 10-20 debris objects per scenario with random:
     - Altitudes, inclinations, and RAANs
     - Values (representing mass, strategic importance, etc.)
   - Assigns random fuel budgets (20-80 km/s delta-v)

2. **Cost Matrix Computation**
   - Uses `max_hohmann.py` to compute transfer costs between all orbit pairs
   - Builds fuel cost matrix for optimization

3. **Optimal Path Finding**
   - Formulates as orienteering problem (maximize value within fuel budget)
   - Uses Gurobi MILP solver with two-stage optimization:
     - Stage 1: Maximize total debris value collected
     - Stage 2: Minimize fuel use while maintaining optimal value
   - Miller-Tucker-Zemlin subtour elimination constraints

4. **Data Export**
   - Saves each scenario and its optimal solution to JSONL file
   - One JSON object per line for easy streaming and processing

### Data Format

Each line in the `.jsonl` files contains:

```json
{
  "scenario_id": 0,
  "timestamp": "2024-12-18T10:30:45.123456",
  "num_debris": 15,
  "fuel_budget": 45.3,
  "interceptor_orbit": {
    "altitude": 420.5,
    "inclination": 52.1,
    "raan": 120.3
  },
  "debris_orbits": [
    {
      "id": 1,
      "altitude": 350.2,
      "inclination": 48.7,
      "raan": 95.4,
      "value": 150
    },
    ...
  ],
  "optimal_sequence": [0, 5, 12, 3, 8, 1],
  "total_value": 850.0,
  "fuel_used": 42.1
}
```

## Usage

### Generating Training Data

```bash
# Generate test dataset (10 scenarios)
python generate_training_data.py

# Edit the script to generate full dataset (10,000 scenarios)
# Uncomment the full dataset section in __main__
```

**Configuration Options:**
- `num_scenarios`: Number of scenarios to generate
- `vary_starting_orbit`: True to randomize interceptor position, False for fixed start
- `output_file`: Output filename for JSONL data
- `checkpoint_every`: Progress reporting frequency

### Loading Data for Training

```python
import json

def load_training_data(filename='debris_training_data_full.jsonl'):
    """Load all scenarios from JSONL file"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Usage
training_data = load_training_data()
print(f"Loaded {len(training_data)} scenarios")
```

## Machine Learning Recommendations

### 1. Supervised Learning (Sequence Prediction)

**Recommended Architectures:**

**Pointer Networks**
- Specifically designed for variable-length sequence outputs
- Learns attention mechanism to select next debris target
- Handles permutation-invariant input (debris can be in any order)

**Transformer-Based Models**
- Use encoder-decoder architecture
- Encode debris field as set of tokens
- Decode optimal visiting sequence
- Pre-trained models (BERT-style) could be adapted

**Graph Neural Networks (GNNs)**
- Represent debris field as graph (nodes = debris, edges = transfer costs)
- Learn node embeddings capturing orbital relationships
- Output node selection probabilities for visiting order

**Implementation Approach:**
```python
# Input features per scenario
X = {
    'interceptor': [altitude, inclination, raan],
    'debris_features': [[alt_i, inc_i, raan_i, value_i], ...],
    'fuel_budget': scalar,
    'cost_matrix': [[cost_ij], ...]  # optional
}

# Output labels
y = optimal_sequence  # [0, 5, 12, 3, 8]

# Training objective
# Cross-entropy loss on next-debris prediction at each step
```

**Training Strategy:**
- Split data: 80% train, 10% validation, 10% test
- Use teacher forcing during training (feed correct previous selections)
- Evaluate on complete sequence accuracy and value collected
- Consider beam search during inference

### 2. Reinforcement Learning (Policy Optimization)

**Imitation Learning → RL Fine-tuning (Recommended)**

**Phase 1: Behavioral Cloning**
- Train initial policy to imitate optimal solutions from dataset
- Faster convergence than pure RL
- Provides good initialization

```python
# Pseudo-code for imitation learning
state = encode_debris_scenario(scenario)
action_probs = policy_network(state)
loss = cross_entropy(action_probs, optimal_next_action)
```

**Phase 2: RL Fine-tuning**
- Use Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC)
- Allow policy to explore beyond imitation data
- May discover better strategies for edge cases

**State Space Design:**
```python
state = {
    'interceptor_orbit': [a, e, i, Omega, omega, nu],  # 6 elements
    'remaining_fuel': scalar,
    'visited_mask': [0/1 for each debris],
    'debris_features': [[orbit_elements, value], ...],
    'debris_distances': [delta_v to each unvisited debris]
}
```

**Action Space:**
- Discrete: Select debris ID (1 to N) or "end mission"
- Can include invalid action masking (already visited, insufficient fuel)

**Reward Function:**
```python
reward = debris_value_collected - alpha * delta_v_used - beta * time_penalty
# where alpha, beta balance fuel efficiency vs speed
```

**RL Environment Setup:**
- Create OpenAI Gym-compatible environment
- Use your `max_hohmann.py` for delta-v calculations
- Episode terminates when fuel exhausted or mission ended
- Generate new random scenarios for each episode during training

### 3. Hybrid Approaches

**Value Function Approximation**
- Train neural network to predict: "What's the maximum value I can collect from this state?"
- Use as heuristic in greedy or beam search algorithms
- Much faster than running Gurobi, nearly optimal results

**Graph Attention Networks + Heuristics**
- Use GNN to learn edge weights (transfer priorities)
- Run fast graph-based algorithms (Christofides, Lin-Kernighan)
- Orders of magnitude faster than MILP, good solution quality

### 4. Model Evaluation Metrics

**Performance Metrics:**
- **Solution Quality**: Value collected compared to optimal
- **Fuel Efficiency**: Fuel used vs. optimal fuel usage
- **Sequence Accuracy**: Exact match rate with optimal sequence
- **Inference Time**: Critical for real-time planning
- **Generalization**: Performance on out-of-distribution scenarios (different debris counts, fuel budgets)

**Benchmark Tests:**
- Vary debris count (5, 10, 15, 20, 25 objects)
- Vary fuel budgets (tight, moderate, generous)
- Different orbital regimes (low LEO, high LEO, mixed)
- Clustered vs. dispersed debris fields

### 5. Practical Considerations

**Data Augmentation:**
- Rotate debris IDs (relabel while preserving sequence logic)
- Vary fuel budgets for existing scenarios
- Apply small perturbations to orbital parameters

**Curriculum Learning:**
- Start training on easy scenarios (few debris, high fuel)
- Gradually increase difficulty (more debris, tighter constraints)
- Improves convergence and final performance

**Multi-Task Learning:**
- Train single model for multiple objectives (value, fuel, time)
- Learn transferable representations of orbital dynamics

**Deployment:**
- Use model as fast heuristic initialization for MILP solver
- Warm-start Gurobi with ML-predicted sequence
- Achieves near-optimal solutions with dramatic speedup

## Requirements

- Python 3.8+
- NumPy
- Gurobi (with valid license)
- gurobipy

**Optional for ML/RL:**
- PyTorch or TensorFlow
- OpenAI Gym (for RL environment)
- PyTorch Geometric (for GNNs)

## Future Extensions

- Incorporate more realistic Lambert transfer calculations
- Add time-window constraints for debris visibility
- Include probabilistic debris removal success rates
- Multi-agent scenarios (multiple interceptor satellites)
- Dynamic debris fields (objects added/removed during mission)

## References

This work builds on the orienteering problem formulation and applies it to orbital debris collection with realistic astrodynamics constraints. The two-stage optimization approach ensures both value maximization and fuel efficiency.

## License

[Your License Here]

## Contact

[Your Contact Information]
