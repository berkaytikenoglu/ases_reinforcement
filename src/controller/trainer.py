
import torch
import numpy as np
import time
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.stdout.reconfigure(line_buffering=True) # Force instant flushing for UI console

from src.model.environment import DefenseEnv
from src.view.renderer import Renderer
try:
    from src.view.renderer_3d import Renderer3D
except ImportError as e:
    print(f"DEBUG: Failed to import Renderer3D: {e}")
    Renderer3D = None # Fallback if ursina not installed

from src.agent.agent import Agent, Memory
from config.parameters import Params
from src.view.training_visualizer import get_visualizer, start_visualizer, close_visualizer

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
        print("FAST TRAINING MODE - No visualization")
    
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
        print(f"Loading model from {model_path}")
        agent.load(model_path)
    else:
        if test_mode:
            print(f"Model not found: {model_path}")
            print("Available models:")
            for m in list_models():
                print(f"  - {m}")
            return
        print("No existing model found, starting fresh.")
    
    time_step = 0
    total_rewards = []
    time_step = 0
    total_rewards = []
    best_reward = float('-inf')     # Best single episode reward (for stats)
    best_avg_reward = float('-inf') # Best average reward (for saving best model)
    start_time = time.time()
    
    mode_str = "TEST MODE" if test_mode else "TRAINING MODE"
    print(f"\n{'='*50}")
    print(f"{mode_str} - {max_episodes} episodes")
    print(f"Model: {model_path}")
    print(f"{'='*50}\n")
    
    # Start training visualizer (only for training mode, not test)
    visualizer = None
    if not test_mode:
        visualizer = start_visualizer()
    
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
            next_state, reward, terminated, truncated, info = env.step(action)
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
        # Progress reporting
        if i_episode % 1 == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
            eps_per_sec = i_episode / elapsed if elapsed > 0 else 0
            
            # Hit stats
            hits = info.get('hits', 0)
            shots = info.get('shots', 0)
            dmg = info.get('agent_hits', 0)
            spawned = info.get('spawned', 0)
            ammo_left = info.get('ammo', 0)
            ammo_used = Params.AMMO_CAPACITY - ammo_left
            hit_rate = (hits / spawned * 100) if spawned > 0 else 0.0
            
            # Result Status
            reason = info.get('reason', '')
            hp = info.get('health', 0)
            
            result = "SUCCESS"
            if reason == 'agent_died':
                result = "DIED"
            elif reason == 'threat_reached_ground_penalty' or hits < spawned:
                result = "FAILED"
            
            print(f"Episode {i_episode:4d}/{max_episodes} | Reward: {current_ep_reward:8.2f} | Avg(100): {avg_reward:8.2f} | Hits: {hits:2d}/{spawned:2d} ({hit_rate:5.1f}%) | Miss: {shots-hits:2d} | HP: {hp} | Result: {result}")
            
            # Update training visualizer graphs
            if visualizer:
                visualizer.add_episode(i_episode, current_ep_reward, avg_reward, hits, spawned, ammo_used, result)
                if i_episode % 5 == 0:  # Update plots every 5 episodes for performance
                    visualizer.update_plots()
            
            # Save BEST model (based on Avg Reward stability, not single crazy episode)
            if not test_mode and avg_reward > best_avg_reward and i_episode >= 50:
                best_avg_reward = avg_reward
                best_model_path = model_path.replace('.pth', '_best.pth')
                agent.save(best_model_path)
                print(f" >>> New Best Model! Avg: {best_avg_reward:.2f} Saved to {os.path.basename(best_model_path)}")
        
        # Save during training
        if not test_mode and i_episode % 100 == 0:
            agent.save(model_path)
            if visualizer:
                # Save with model name as prefix
                base_name = os.path.basename(model_path).replace('.pth', '')
                viz_path = os.path.join(os.path.dirname(model_path), f"{base_name}_training_metrics.png")
                visualizer.save_figure(viz_path)
                # print(f" > Metrics saved to {base_name}_training_metrics.png")
            print(f"Model saved to {model_path}")
    
    # Final save (training only)
    if not test_mode:
        agent.save(model_path)
        print(f"Final model saved to {model_path}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"DONE! {'Test' if test_mode else 'Training'} Complete!")
    print(f"Final Average Reward (last 100): {np.mean(total_rewards[-100:]):.2f}")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Total Time: {elapsed/60:.1f} minutes")
    print(f"{'='*50}")
            
    if render and renderer:
        renderer.close()
    
    # Save and close visualizer
    if visualizer:
        graph_path = model_path.replace('.pth', '_training_metrics.png')
        visualizer.save_figure(graph_path)
        visualizer.close()
        
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
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode (disable GPU)')
    args = parser.parse_args()
    
    # Set device globally before training
    if args.cpu:
        import torch
        torch.cuda.is_available = lambda: False
        print("[CPU] Forced CPU mode")
    
    # List models and exit
    if args.list:
        print("\nAvailable Models:")
        print(f"Location: {MODELS_DIR}")
        print("-" * 40)
        models = list_models()
        if models:
            for m in models:
                filepath = os.path.join(MODELS_DIR, m)
                size = os.path.getsize(filepath) / 1024
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M")
                print(f"   {m} ({size:.1f} KB) - {mtime}")
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
