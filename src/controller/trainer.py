
import torch
import numpy as np
import time
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.environment import DefenseEnv
from src.view.renderer import Renderer
try:
    from src.view.renderer_3d import Renderer3D
except ImportError as e:
    print(f"DEBUG: Failed to import Renderer3D: {e}")
    Renderer3D = None # Fallback if ursina not installed

from src.agent.agent import Agent, Memory
from config.parameters import Params

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models')
os.makedirs(MODELS_DIR, exist_ok=True)

def list_models():
    """List all available trained models"""
    models = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith('.pth'):
                models.append(f)
    return models

def train(render=True, render_3d=False, max_episodes=1000, model_name=None, test_mode=False):
    env = DefenseEnv()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    renderer = None
    if render:
        print(f"DEBUG: render_3d arg is {render_3d}")
        print(f"DEBUG: Renderer3D class is {Renderer3D}")
        
        if render_3d and Renderer3D:
            print("DEBUG: Initializing 3D Renderer")
            renderer = Renderer3D(env)
        else:
            print("DEBUG: Initializing 2D Renderer (Fallback or Requested)")
            renderer = Renderer(env)
    else:
        print("🚀 FAST TRAINING MODE - No visualization")
    
    # PPO Hyperparameters
    update_timestep = 2000
    lr = Params.LEARNING_RATE
    gamma = Params.GAMMA
    K_epochs = 4
    eps_clip = 0.2
    
    memory = Memory()
    agent = Agent(state_dim, action_dim, lr, gamma, K_epochs, eps_clip)
    
    # Determine model path
    if model_name:
        model_path = os.path.join(MODELS_DIR, model_name)
        if not model_path.endswith('.pth'):
            model_path += '.pth'
    else:
        model_path = os.path.join(MODELS_DIR, 'latest.pth')
    
    # Load existing model if exists
    if os.path.exists(model_path):
        print(f"✅ Loading model from {model_path}")
        agent.load(model_path)
    else:
        if test_mode:
            print(f"❌ Model not found: {model_path}")
            print("Available models:")
            for m in list_models():
                print(f"  - {m}")
            return
        print("🆕 No existing model found, starting fresh.")
    
    time_step = 0
    total_rewards = []
    best_reward = float('-inf')
    start_time = time.time()
    
    mode_str = "🧪 TEST MODE" if test_mode else "🎯 TRAINING MODE"
    print(f"\n{'='*50}")
    print(f"{mode_str} - {max_episodes} episodes")
    print(f"Model: {model_path}")
    print(f"{'='*50}\n")
    
    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset()
        current_ep_reward = 0
        
        # Update episode display in renderer
        if render and renderer and hasattr(renderer, 'set_episode'):
            renderer.set_episode(i_episode)
        
        while True:
            time_step += 1
            
            # Select action (deterministic in test mode, stochastic in training)
            action, action_logprob = agent.select_action(state)
            state_tensor = torch.FloatTensor(state)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Only save to memory and train if not in test mode
            if not test_mode:
                memory.states.append(state_tensor)
                memory.actions.append(torch.FloatTensor(action))
                memory.logprobs.append(torch.FloatTensor(action_logprob))
                memory.rewards.append(reward)
                memory.is_terminals.append(done)
                
                if time_step % update_timestep == 0:
                    agent.update(memory)
                    memory.clear_memory()
                    time_step = 0
                
            current_ep_reward += reward
            
            # Visualization
            if render and renderer:
                if not renderer.render():
                    env.close()
                    renderer.close()
                    return

            if done:
                break
                
            state = next_state
        
        total_rewards.append(current_ep_reward)
        if current_ep_reward > best_reward:
            best_reward = current_ep_reward
        
        # Progress reporting
        if i_episode % 10 == 0 or i_episode == 1:
            elapsed = time.time() - start_time
            avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
            eps_per_sec = i_episode / elapsed if elapsed > 0 else 0
            print(f"Episode {i_episode:4d}/{max_episodes} | Reward: {current_ep_reward:8.2f} | Avg(100): {avg_reward:8.2f} | Best: {best_reward:8.2f} | Speed: {eps_per_sec:.1f} ep/s")
        
        # Save during training
        if not test_mode and i_episode % 50 == 0:
            agent.save(model_path)
            print(f"💾 Model saved to {model_path}")
    
    # Final save (training only)
    if not test_mode:
        agent.save(model_path)
        # Also save a timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(MODELS_DIR, f"agent_{timestamp}.pth")
        agent.save(backup_path)
        print(f"💾 Backup saved to {backup_path}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"✅ {'Test' if test_mode else 'Training'} Complete!")
    print(f"📊 Final Average Reward (last 100): {np.mean(total_rewards[-100:]):.2f}")
    print(f"🏆 Best Reward: {best_reward:.2f}")
    print(f"⏱️ Total Time: {elapsed/60:.1f} minutes")
    print(f"{'='*50}")
            
    if render and renderer:
        renderer.close()
    env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ASES Defense Agent Trainer')
    parser.add_argument('--3d', action='store_true', help='Enable 3D visualization')
    parser.add_argument('--fast', action='store_true', help='Fast training mode (no visualization)')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--model', type=str, default=None, help='Model name to load/save (without .pth)')
    parser.add_argument('--test', action='store_true', help='Test mode (no training, just run agent)')
    parser.add_argument('--list', action='store_true', help='List available models')
    args = parser.parse_args()
    
    # List models and exit
    if args.list:
        print("\n📁 Available Models:")
        print(f"Location: {MODELS_DIR}")
        print("-" * 40)
        models = list_models()
        if models:
            for m in models:
                filepath = os.path.join(MODELS_DIR, m)
                size = os.path.getsize(filepath) / 1024
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M")
                print(f"  📦 {m} ({size:.1f} KB) - {mtime}")
        else:
            print("  No models found. Train one with: python trainer.py --fast --episodes 1000")
        print()
        sys.exit(0)
    
    # Fast mode disables rendering
    render_enabled = not args.fast
    
    # Test mode always enables rendering
    if args.test:
        render_enabled = True
    
    train(
        render=render_enabled, 
        render_3d=args.__dict__.get('3d', False), 
        max_episodes=args.episodes,
        model_name=args.model,
        test_mode=args.test
    )
