# Terrain-Aware Path Recommendation

A physics-based reinforcement learning system for generating safe hiking paths that outperform traditional pathfinding algorithms like A*. Uses real terrain data and research-backed physics simulation to learn routes that naturally exhibit switchback patterns—just like real mountain trails.

---

## Key Features

- **PyBullet Physics Simulation**: Realistic terrain interaction with research-backed friction coefficients
- **Sparse Reward Design**: Goal-only rewards force agent to learn from physics, not reward hacking
- **Terrain-Aware Energy Model**: Naismith's Rule-based stamina system encourages gradual climbing
- **SAC Training**: Soft Actor-Critic for continuous action spaces with curriculum learning
- **A* Comparison**: Automated comparison between RL paths, A* paths, and actual hiking trails

---

## Research Motivation

Traditional pathfinding algorithms (A*, Dijkstra) optimize for distance or simple slope penalties but fail to account for:

1. **Cumulative fatigue**: Steep direct routes drain energy exponentially
2. **Terrain traversability**: Scree and loose rock are dangerous regardless of slope
3. **Safety margins**: Real trails use switchbacks for a reason

This project demonstrates that RL agents, trained with realistic physics constraints, naturally discover switchback patterns—the same solution humans evolved over centuries of mountain travel.

---

## Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd Terrain-Aware-Path-Recommendation
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data (if not already present)
python 1.download_data.py
python 2.preprocess_data.py

# 4. Train agent (expect 2-5 hours)
python train_pybullet_agent.py --timesteps 2000000

# 5. Compare with A* and actual trails
python compare_with_astar.py --visualize --metrics
```

---

## File Structure

```
├── pybullet_terrain_env.py     # NEW: PyBullet physics environment
├── train_pybullet_agent.py     # NEW: SAC training with curriculum
├── compare_with_astar.py       # NEW: A* comparison and visualization
│
├── 1.download_data.py          # Download DEM and landcover
├── 2.preprocess_data.py        # Process terrain for RL
├── 4.test_agent.py             # Test trained agent
├── requirements.txt            # Python dependencies
│
├── physics_hiking_env.py       # LEGACY: Original grid-based environment
├── 3.build_and_train.py        # LEGACY: Original PPO training
│
├── data/
│   ├── raw/                    # DEM, landcover, GPX files
│   └── processed/              # Slope, stability, terrain maps
└── outputs/
    ├── checkpoints/            # Trained model files
    └── tensorboard/            # Training logs
```

---

## Physics Model

### Terrain Friction Coefficients

Research-backed values from biomechanics and geotechnical literature:

| Surface | Friction (μ) | Source |
|---------|-------------|--------|
| Established trail | 0.75 | [1] |
| Dry rock | 0.70 | [1] |
| Grass | 0.55 | [2] |
| Forest floor | 0.50 | [2] |
| Scree/talus | 0.35 | [3] |

### Energy Model (Naismith-Based)

Following Naismith's Rule [4] and Scarf's validation [5]:

```
Energy Cost = Horizontal Distance + (Vertical Gain × 8)
```

This 8:1 equivalence ratio means 1 meter of climbing equals 8 meters of flat walking in effort—making steep direct routes extremely costly and naturally encouraging switchbacks.

---

## Comparison with A*

The `compare_with_astar.py` tool generates side-by-side comparisons:

| Metric | A* Path | RL Agent | Advantage |
|--------|---------|----------|-----------|
| **Energy Cost** | High | Lower | RL finds efficient switchbacks |
| **Max Slope** | Steep | Gradual | RL avoids dangerous terrain |
| **Risk Score** | Higher | Lower | RL prioritizes safety |
| **Path Length** | Shorter | Longer | Acceptable trade-off |

---

## Training Details

**Algorithm**: SAC (Soft Actor-Critic) [6]
- Better for continuous action spaces than PPO
- Off-policy for sample efficiency
- Entropy regularization for exploration

**Curriculum Learning**:
```python
Level 0: 25m  @ 500 steps   → Level 1: 50m  @ 1000 steps
Level 2: 100m @ 2000 steps  → Level 3: 250m @ 5000 steps
Level 4: 500m @ 10000 steps → Level 5: 1000m @ 25000 steps
```

---

## References

1. Ziaei, M. et al. (2017). "Coefficient of friction, walking speed and cadence on slippery and dry surfaces." *Int. J. Occupational Safety and Ergonomics*. DOI: [10.1080/10803548.2017.1398922](https://doi.org/10.1080/10803548.2017.1398922)

2. Beschorner, K.E. et al. (2016). "Required coefficient of friction during level walking is predictive of slipping." *Gait & Posture*. DOI: [10.1016/j.gaitpost.2016.05.021](https://doi.org/10.1016/j.gaitpost.2016.05.021)

3. Piazza, F. et al. (2022). "Active Scree Slope Stability Investigation Based on Geophysical and Geotechnical Approach." *MDPI Water*, 14(16), 2569. DOI: [10.3390/w14162569](https://doi.org/10.3390/w14162569)

4. Naismith, W.W. (1892). "Cruach Ardran, Stobinian, and Ben More." *Scottish Mountaineering Club Journal*, 2, 136.

5. Scarf, P.A. (2007). "Route choice in mountain navigation, Naismith's Rule, and the equivalence of distance and climb." *J. Operational Research Society*, 58(9), 1199-1205. DOI: [10.1057/palgrave.jors.2602249](https://doi.org/10.1057/palgrave.jors.2602249)

6. Haarnoja, T. et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." *ICML 2018*. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)

7. Wellhausen, L. et al. (2019). "Where Should I Walk? Predicting Terrain Properties from Images via Self-Supervised Learning." *IEEE RA-L*. DOI: [10.1109/LRA.2019.2930266](https://doi.org/10.1109/LRA.2019.2930266)

---

## License

MIT License - See LICENSE file for details.
