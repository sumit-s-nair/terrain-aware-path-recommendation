"""
Training Script for PyBullet Terrain Agent using SAC.

Uses Soft Actor-Critic (SAC) algorithm for continuous action spaces
with curriculum learning for progressive difficulty increase.

References:
    [1] Haarnoja, T. et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy 
        Deep Reinforcement Learning with a Stochastic Actor." ICML 2018.
    [2] Scarf, P. A. (2007). "Route choice in mountain navigation, Naismith's Rule." 
        J. Operational Research Society, 58(9), 1199-1205.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

# Ensure we can import local modules
sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

from pybullet_terrain_env import PyBulletTerrainEnv, CURRICULUM_LEVELS


OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"

for d in [CHECKPOINT_DIR, LOG_DIR, TENSORBOARD_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class CurriculumCallback(BaseCallback):
    """
    Callback to manage curriculum learning progression.
    
    Advances to next difficulty level when success rate exceeds threshold.
    """
    
    def __init__(
        self,
        success_threshold: float = 0.8,
        check_freq: int = 1000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.check_freq = check_freq
        
        self.episode_successes = []
        self.episode_rewards = []
        self.current_level = 0
        self.levels_completed = []
    
    def _on_step(self) -> bool:
        # Check for episode completion
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            infos = self.locals.get("infos", [{}] * len(dones))
            
            for done, info in zip(dones, infos):
                if done:
                    success = info.get("reached_goal", False)
                    self.episode_successes.append(1 if success else 0)
                    
                    if self.verbose > 0 and success:
                        print(f"✅ Goal reached! Episode reward: {info.get('episode', {}).get('r', 'N/A')}")
        
        # Check curriculum progression periodically
        if self.n_calls % self.check_freq == 0 and len(self.episode_successes) >= 20:
            recent_successes = self.episode_successes[-50:]  # Last 50 episodes
            success_rate = sum(recent_successes) / len(recent_successes)
            
            level_config = CURRICULUM_LEVELS[min(self.current_level, len(CURRICULUM_LEVELS) - 1)]
            required = level_config.get("required_successes", 50)
            
            if self.verbose > 0:
                print(f"\n📊 Curriculum Level {self.current_level}: "
                      f"Success rate {success_rate:.1%} (need {self.success_threshold:.0%})")
            
            # Only check success rate - the required_successes is for tracking, not blocking
            if success_rate >= self.success_threshold:
                if self.current_level < len(CURRICULUM_LEVELS) - 1:
                    self.current_level += 1
                    self.levels_completed.append(self.num_timesteps)
                    self.episode_successes = []  # Reset for new level
                    
                    new_config = CURRICULUM_LEVELS[self.current_level]
                    print(f"\n🎓 CURRICULUM ADVANCED to Level {self.current_level}!")
                    print(f"   Goal distance: {new_config['goal_distance']}m")
                    print(f"   Max steps: {new_config['max_steps']}")
                    
                    # Update environment parameters
                    self._update_env_curriculum()
        
        return True
    
    def _update_env_curriculum(self):
        """Update environment configuration for new curriculum level."""
        try:
            config = CURRICULUM_LEVELS[self.current_level]
            
            for env in self.training_env.envs:
                if hasattr(env, 'env'):
                    actual_env = env.env  # Get unwrapped env from Monitor
                else:
                    actual_env = env
                
                actual_env.goal_distance_meters = config["goal_distance"]
                actual_env.max_steps = config["max_steps"]
                actual_env.curriculum_level = self.current_level
                
        except Exception as e:
            print(f"Warning: Could not update curriculum: {e}")


class ProgressCallback(BaseCallback):
    """Progress bar and logging callback."""
    
    def __init__(self, total_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.episode_count = 0
        self.recent_rewards = []
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="🏔️ Training Hiking Agent")
    
    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(1)
        
        # Update postfix with stats
        if self.locals.get("dones") is not None:
            for done, info in zip(self.locals["dones"], self.locals.get("infos", [])):
                if done and "episode" in info:
                    self.episode_count += 1
                    self.recent_rewards.append(info["episode"]["r"])
                    
                    if len(self.recent_rewards) > 100:
                        self.recent_rewards = self.recent_rewards[-100:]
                    
                    if self.pbar and self.episode_count % 10 == 0:
                        avg_reward = np.mean(self.recent_rewards)
                        self.pbar.set_postfix({
                            "ep": self.episode_count,
                            "avg_r": f"{avg_reward:.0f}",
                            "goal_rate": f"{sum(1 for r in self.recent_rewards if r > 0) / len(self.recent_rewards):.1%}"
                        })
        
        return True
    
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()


class HeatmapCallback(BaseCallback):
    """
    Callback to save trajectory heatmaps periodically.
    
    Saves heatmap images showing where the agent explored during episodes.
    """
    
    def __init__(
        self,
        save_freq: int = 50,  # Save every N episodes
        save_path: str = "outputs/heatmaps",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        self.episode_count = 0
        self.best_progress = 0.0
        self.recent_progress = []
    
    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            dones = self.locals["dones"]
            infos = self.locals.get("infos", [{}] * len(dones))
            
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    self.episode_count += 1
                    
                    # Track progress
                    progress = info.get("progress_made", 0.0)
                    self.recent_progress.append(progress)
                    if len(self.recent_progress) > 100:
                        self.recent_progress = self.recent_progress[-100:]
                    
                    # Update best progress
                    if progress > self.best_progress:
                        self.best_progress = progress
                        if self.verbose > 0:
                            print(f"\n🎯 New best progress: {progress:.1f}m")
                    
                    # Save heatmap periodically
                    if self.episode_count % self.save_freq == 0:
                        self._save_heatmap(info, i)
                    
                    # Log progress stats
                    if self.episode_count % 50 == 0:
                        avg_progress = np.mean(self.recent_progress) if self.recent_progress else 0
                        print(f"\n📈 Avg progress (last 100 eps): {avg_progress:.1f}m, Best: {self.best_progress:.1f}m")
        
        return True
    
    def _save_heatmap(self, info: dict, env_idx: int):
        """Save heatmap from episode info."""
        heatmap = info.get("heatmap")
        if heatmap is None:
            return
        
        # Get the actual environment to use save_heatmap method
        try:
            env = self.training_env.envs[env_idx]
            if hasattr(env, 'env'):
                actual_env = env.env
            else:
                actual_env = env
            
            # Set heatmap data and save
            actual_env.trajectory_heatmap = heatmap
            
            filename = f"heatmap_ep{self.episode_count:06d}.png"
            filepath = self.save_path / filename
            actual_env.save_heatmap(str(filepath))
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save heatmap: {e}")



def make_env(curriculum_level: int = 0, log_dir: str = None):
    """Factory function to create monitored environment."""
    def _init():
        config = CURRICULUM_LEVELS[min(curriculum_level, len(CURRICULUM_LEVELS) - 1)]
        
        env = PyBulletTerrainEnv(
            goal_distance_meters=config["goal_distance"],
            max_steps=config["max_steps"],
            curriculum_level=curriculum_level,
        )
        
        if log_dir:
            env = Monitor(env, log_dir)
        
        return env
    
    return _init


def train(
    total_timesteps: int = 2_000_000,
    learning_rate: float = 3e-4,
    buffer_size: int = 500_000,
    batch_size: int = 512,
    starting_level: int = 0,
    n_envs: int = 1,  # Single env for PyBullet stability
    checkpoint_freq: int = 50000,
    resume_from: str = None,
):
    """
    Train SAC agent on terrain environment.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: SAC learning rate
        buffer_size: Replay buffer size
        batch_size: Training batch size
        starting_level: Starting curriculum level (0-5)
        n_envs: Number of parallel environments
        checkpoint_freq: Steps between checkpoints
        resume_from: Path to checkpoint to resume from
    """
    print("=" * 60)
    print("🏔️  TERRAIN-AWARE PATH RECOMMENDATION TRAINING")
    print("=" * 60)
    print(f"Algorithm: SAC (Soft Actor-Critic)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Curriculum starting level: {starting_level}")
    print(f"Checkpoint frequency: {checkpoint_freq:,}")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device.upper()}")
    print()
    
    # Create environment
    log_path = str(LOG_DIR / datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_path, exist_ok=True)
    
    env = DummyVecEnv([make_env(starting_level, log_path)])
    
    # Create or load model
    if resume_from and Path(resume_from).exists():
        print(f"📂 Loading model from {resume_from}")
        model = PPO.load(resume_from, env=env)
    else:
        print("🆕 Creating new PPO model (best performer from comparison tests)")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,             # Higher entropy for exploration (BC+PPO setting)
            verbose=1,
            tensorboard_log=str(TENSORBOARD_DIR),
            device=device,
        )
    
    # Setup callbacks
    callbacks = [
        ProgressCallback(total_timesteps),
        CurriculumCallback(success_threshold=0.60, check_freq=5000),  # Lowered from 80%
        HeatmapCallback(save_freq=50, save_path=str(OUTPUT_DIR / "heatmaps")),
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(CHECKPOINT_DIR),
            name_prefix="terrain_ppo"
        ),
    ]
    
    # Train
    print("\n🚀 Starting training...")
    print("   Press Ctrl+C to stop early (model will be saved)")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=f"SAC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            reset_num_timesteps=resume_from is None,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = CHECKPOINT_DIR / f"terrain_sac_final_{timestamp}.zip"
    model.save(final_path)
    print(f"\n💾 Final model saved to: {final_path}")
    
    # Cleanup
    env.close()
    
    return model, str(final_path)


def evaluate(model_path: str, n_episodes: int = 10, render: bool = False):
    """Evaluate trained model."""
    print(f"\n📊 Evaluating model: {model_path}")
    
    model = SAC.load(model_path)
    
    # Test at different difficulty levels
    results = []
    
    for level in range(len(CURRICULUM_LEVELS)):
        config = CURRICULUM_LEVELS[level]
        env = PyBulletTerrainEnv(
            goal_distance_meters=config["goal_distance"],
            max_steps=config["max_steps"],
        )
        
        successes = 0
        total_reward = 0
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    if info.get("reached_goal"):
                        successes += 1
                    break
            
            total_reward += episode_reward
        
        success_rate = successes / n_episodes
        avg_reward = total_reward / n_episodes
        
        print(f"Level {level} ({config['goal_distance']}m): "
              f"Success {success_rate:.1%}, Avg reward {avg_reward:.0f}")
        
        results.append({
            "level": level,
            "distance": config["goal_distance"],
            "success_rate": success_rate,
            "avg_reward": avg_reward,
        })
        
        env.close()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train terrain hiking agent")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="Total training timesteps")
    parser.add_argument("--level", type=int, default=0,
                        help="Starting curriculum level (0-5)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="Path to model to evaluate (skip training)")
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate(args.evaluate)
    else:
        train(
            total_timesteps=args.timesteps,
            starting_level=args.level,
            resume_from=args.resume,
        )
