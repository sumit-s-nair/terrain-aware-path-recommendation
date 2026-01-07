"""Test trained agent behavior at various distances."""
import numpy as np
from stable_baselines3 import PPO
from pybullet_terrain_env import PyBulletTerrainEnv
import os

print("="*60)
print("TESTING TRAINED AGENT BEHAVIOR")
print("="*60)

# Find the latest checkpoint
checkpoint_dir = "outputs/checkpoints"
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
if not checkpoints:
    print("No checkpoints found!")
    exit(1)

# Get most recent by steps
latest = sorted(checkpoints, key=lambda x: int(x.split('_')[-2]) if '_steps' in x else 0)[-1]
model_path = os.path.join(checkpoint_dir, latest)
print(f"Loading: {model_path}")

model = PPO.load(model_path)

# Test at different distances
for goal_dist in [150, 200, 250]:
    print(f"\n{'='*60}")
    print(f"Testing at {goal_dist}m goal distance")
    print("="*60)
    
    env = PyBulletTerrainEnv(goal_distance_meters=goal_dist, max_steps=6000)
    
    successes = 0
    for ep in range(3):
        obs, info = env.reset()
        initial_dist = info['initial_distance']
        min_dist = initial_dist
        
        for step in range(6000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            
            curr_dist = info['goal_distance']
            min_dist = min(min_dist, curr_dist)
            
            if term or trunc:
                break
        
        progress = initial_dist - min_dist
        success = "✅" if info.get('reached_goal', False) else "❌"
        if info.get('reached_goal', False):
            successes += 1
        
        print(f"  Ep {ep+1}: {success} Start={initial_dist:.0f}m → Min={min_dist:.0f}m (progress={progress:.0f}m) in {step} steps")
        
        # Show technique breakdown
        techs = info.get('technique_counts', {})
        total = sum(techs.values())
        if total > 0:
            tech_str = ", ".join([f"{k}:{v/total*100:.0f}%" for k,v in techs.items() if v > 0])
            print(f"       Techniques: {tech_str}")
    
    print(f"  Success rate: {successes}/3")
    env.close()

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
