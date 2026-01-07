# Terrain-Aware Path Recommendation

A physics-based reinforcement learning system for generating safe hiking paths that outperform traditional pathfinding algorithms like A*. Uses real terrain data and research-backed physics simulation to learn routes that naturally exhibit switchback patterns—just like real mountain trails.

---

## Key Features

- **PyBullet Physics Simulation**: Realistic terrain interaction with research-backed friction coefficients
- **Dense Progress Rewards**: Gradient-based rewards guide agent toward goals efficiently
- **Research-Backed Safety Model**: Fall distance thresholds from mountaineering trauma research
- **PPO Training**: Proximal Policy Optimization with 17-level curriculum (10m → 14.6km)
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

### Slope-Based Traversal Techniques

The agent uses different techniques based on terrain steepness, with speed factors derived from mountaineering research [10]:

| Slope Range | Technique | Speed Factor | Reward Multiplier |
|-------------|-----------|--------------|-------------------|
| 0-30° | Walking | 100% | 100% |
| 30-45° | Steep Hiking | 60% | 80% |
| 45-60° | Rock Climbing / Careful Descent | 30% | 50% |
| 60-75° | Technical Climbing / Rappelling | 15% | 30% |
| 75-90° | Extreme Climbing | 5% | 10% |

**Research basis:**
- Normal hiking: ~5 km/h horizontal, ~300 m/h vertical (Naismith's Rule [4])
- Technical rock climbing: ~185 m/h vertical (23.4m in 7:36 min on 5c route) [10]
- Mountaineering estimate: 200-300 vertical m/h for moderate terrain [10]

Lower reward multipliers on steep terrain encourage finding easier routes when possible, but allow traversing any terrain when necessary.

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

**Algorithm**: PPO (Proximal Policy Optimization) [6]
- Stable on-policy learning
- Works well with curriculum learning
- Dense progress rewards for gradient guidance

**Full Trail Curriculum** (14.6km, 1,397m elevation gain):

| Levels | Distance Range | Max Steps |
|--------|---------------|----------|
| 0-6 | 10m → 50m | 500 → 1,500 |
| 7-11 | 75m → 500m | 2,000 → 12,000 |
| 12-14 | 1km → 5km | 25,000 → 120,000 |
| 15-16 | 10km → 14.6km | 250,000 → 400,000 |

Agent advances when achieving 60% success rate at each level.

---

## References

1. Ziaei, M. et al. (2017). "Coefficient of friction, walking speed and cadence on slippery and dry surfaces." *Int. J. Occupational Safety and Ergonomics*. DOI: [10.1080/10803548.2017.1398922](https://doi.org/10.1080/10803548.2017.1398922)

2. Beschorner, K.E. et al. (2016). "Required coefficient of friction during level walking is predictive of slipping." *Gait & Posture*. DOI: [10.1016/j.gaitpost.2016.05.021](https://doi.org/10.1016/j.gaitpost.2016.05.021)

3. Piazza, F. et al. (2022). "Active Scree Slope Stability Investigation Based on Geophysical and Geotechnical Approach." *MDPI Water*, 14(16), 2569. DOI: [10.3390/w14162569](https://doi.org/10.3390/w14162569)

4. Naismith, W.W. (1892). "Cruach Ardran, Stobinian, and Ben More." *Scottish Mountaineering Club Journal*, 2, 136.

5. Scarf, P.A. (2007). "Route choice in mountain navigation, Naismith's Rule, and the equivalence of distance and climb." *J. Operational Research Society*, 58(9), 1199-1205. DOI: [10.1057/palgrave.jors.2602249](https://doi.org/10.1057/palgrave.jors.2602249)

6. Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms." [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

7. Dickinson, E. et al. (2022). "Epidemiology of mountaineering fall accidents." *Wilderness & Environmental Medicine*, 33(2), 158-165. DOI: [10.1016/j.wem.2022.01.004](https://doi.org/10.1016/j.wem.2022.01.004)

8. Hohlrieder, M. et al. (2007). "Severity and predictors of injury in climbing falls." *High Altitude Medicine & Biology*, 8(1), 39-43. DOI: [10.1089/ham.2006.1048](https://doi.org/10.1089/ham.2006.1048)

9. Wellhausen, L. et al. (2019). "Where Should I Walk? Predicting Terrain Properties from Images via Self-Supervised Learning." *IEEE RA-L*. DOI: [10.1109/LRA.2019.2930266](https://doi.org/10.1109/LRA.2019.2930266)

10. Watts, P.B. et al. (2000). "Metabolic response during sport rock climbing and the effects of active versus passive recovery." *Int. J. Sports Medicine*, 21(3), 185-190. See also: Draper, N. et al. (2009). "Self-selected and imposed speed climbing." *J. Sports Sciences*, 27(4), 391-401. DOI: [10.1080/02640410802603827](https://doi.org/10.1080/02640410802603827)

---

## License

MIT License - See LICENSE file for details.
