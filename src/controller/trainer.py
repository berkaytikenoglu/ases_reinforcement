
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
from src.agent.normalization import RewardScaler
from config.parameters import Params
from src.view.training_visualizer import get_visualizer, start_visualizer, close_visualizer

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../../models')
os.makedirs(MODELS_DIR, exist_ok=True)

def list_models():
    """List all available trained models (folders and .pth files)"""
    models = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            path = os.path.join(MODELS_DIR, f)
            if os.path.isdir(path):
                # It's an agent folder
                models.append(f)
            elif f.endswith('.pth'):
                # Legacy model file
                models.append(f.replace('.pth', ''))
    return sorted(models)

def train(render=True, render_3d=False, max_episodes=1000, model_name=None, test_mode=False, starting_phase=None):
    # Determine initial phase: Override > Default(1)
    init_phase = starting_phase if starting_phase is not None else 1
    env = DefenseEnv(curriculum_phase=init_phase) 
    
    state_dim = 15 # SMART OBSERVATION SIZE
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
    # PPO Hyperparameters
    update_timestep = Params.ROLLOUT_STEPS
    lr = Params.LEARNING_RATE
    gamma = Params.GAMMA
    K_epochs = Params.PPO_EPOCHS
    eps_clip = getattr(Params, 'CLIP_RANGE', 0.2)
    
    memory = Memory()
    agent = Agent(state_dim, action_dim, lr, gamma, K_epochs, eps_clip)
    
    # Reward Normalization
    reward_scaler = RewardScaler(gamma=gamma)
    
    # Determine model path
    save_path = None
    best_model_path = None
    
    if model_name:
        # New Folder Structure: models/{name}/{name}_latest.pth
        model_dir = os.path.join(MODELS_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        save_path = os.path.join(model_dir, f"{model_name}_latest.pth")
        best_model_path = os.path.join(model_dir, f"{model_name}_best.pth")
        
        # Check if we have a legacy model in root to migrate/load
        legacy_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
        
        if os.path.exists(save_path):
            model_path = save_path
        elif os.path.exists(best_model_path):
            # Fallback to best model if latest doesn't exist (early stop)
            print(f"Latest model not found, using best model: {best_model_path}")
            model_path = best_model_path
        elif os.path.exists(legacy_path):
            print(f"Found legacy model {legacy_path}, loading it as base...")
            model_path = legacy_path
        else:
            model_path = save_path # Will start fresh
            
    else:
        # Default fallback
        model_path = os.path.join(MODELS_DIR, 'latest.pth')
        save_path = model_path
    
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
    success_history = []  # FAZ 2 -> 3 geçişi için gerekli
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
        # Update Learning Rate
        if not test_mode:
            agent.decay_lr(i_episode, max_episodes)
            
        state, _ = env.reset()
        reward_scaler.reset() # Reset running return calculation for new episode
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
                # Normalize reward for training stability
                # Raw reward is kept for display/logging (current_ep_reward)
                norm_reward = reward_scaler(reward)
                
                memory.states.append(state_tensor)
                memory.actions.append(torch.FloatTensor(action))
                memory.logprobs.append(torch.FloatTensor(action_logprob))
                memory.rewards.append(norm_reward)
                memory.is_terminals.append(done)
                
                if time_step % update_timestep == 0:
                    agent.update(memory)
                    memory.clear_memory()
                    time_step = 0
                
            current_ep_reward += reward
            
            # Visualization
            if render and renderer:
                # Optimized rendering: Draw every 3rd frame in training, every frame in test
                if test_mode or time_step % 3 == 0:
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
            # Strict logic: SUCCESS only if ANY spawned AND ALL spawned are destroyed
            if spawned == 0 or hits < spawned:
                 result = "FAILED"
                 
            if env.curriculum_phase == 1:
                result = "TRAINING" # In Phase 1, it's always training, not failing
                 
            if reason == 'agent_died':
                result = "DIED"
                
            # Track Success Result
            success_history.append(1 if result == "SUCCESS" else 0)
            # Calculate Success Rate (Rolling 100)
            success_rate = np.mean(success_history[-100:]) * 100 if len(success_history) > 0 else 0
            
            phase = env.curriculum_phase
            
            # Auto-Curriculum Promotion (EPISODE-BASED for reliable progression)
            if phase == 1 and i_episode >= 100:
                # Phase 1 complete after 100 episodes of aim training
                env.curriculum_phase = 2
                print(f"\n >>> PROMOTION! Entering Phase 2: ONE SHOT MODE (after 100 ep aim training) <<<\n")
                if renderer: renderer.set_phase_text("PHASE 2: ONE SHOT")
                if visualizer: visualizer.set_phase(2, i_episode)
                
            elif phase == 2 and i_episode >= 600 and success_rate > 30.0:
                # Phase 2 complete after 600 total episodes + 30% success
                env.curriculum_phase = 3
                print(f"\n >>> PROMOTION! Entering Phase 3: FULL WARFARE <<<\n")
                if renderer: renderer.set_phase_text("PHASE 3: FULL WARFARE")
                if visualizer: visualizer.set_phase(3, i_episode)

            # Ammo check - Only for Phase > 1
            if phase > 1 and result != "SUCCESS" and ammo_left <= 0:
                result += " (Ammo Issue)"
            
            ammo_used = Params.AMMO_CAPACITY - ammo_left
            ammo_used_pct = (ammo_used / Params.AMMO_CAPACITY) * 100
            print(f"Ep {i_episode:4d} | P{phase} | R: {current_ep_reward:6.1f} | Avg: {avg_reward:6.1f} | Hits: {hits}/{spawned} | Used: {ammo_used}/{Params.AMMO_CAPACITY} ({ammo_used_pct:.0f}%) | {result}")
            
            # Update training visualizer graphs
            if visualizer:
                visualizer.add_episode(i_episode, current_ep_reward, avg_reward, hits, spawned, ammo_used, result)
                if i_episode % 1 == 0:  # Update every episode (User Request: Real-time, Optimized)
                    visualizer.update_plots()
            
            # Save BEST model
            if not test_mode and avg_reward > best_avg_reward and i_episode >= 50:
                best_avg_reward = avg_reward
                # Use pre-calculated best path or fallback
                target_best_path = best_model_path if best_model_path else model_path.replace('.pth', '_best.pth')
                agent.save(target_best_path)
                print(f" >>> New Best Model! Avg: {best_avg_reward:.2f} Saved to {os.path.basename(target_best_path)}")
        
        # Save during training
        if not test_mode and i_episode % 100 == 0:
            if save_path:
                agent.save(save_path)
                print(f"Model saved to {save_path}")
                
            if visualizer and model_name:
                # Save metrics to the agent folder
                model_dir = os.path.join(MODELS_DIR, model_name)
                viz_path = os.path.join(model_dir, f"{model_name}_training_metrics.png")
                visualizer.save_figure(viz_path)
            elif visualizer:
                 # Fallback
                 visualizer.save_figure("training_metrics.png")
    
    # Final save (training only)
    if not test_mode:
        target = save_path if save_path else model_path
        agent.save(target)
        print(f"Final model saved to {target}")
    
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
    parser.add_argument('--viz3d', action='store_true', help='Enable 3D visualization')
    parser.add_argument('--fast', action='store_true', help='Fast training mode (no visualization)')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--model', type=str, default=None, help='Model name to load/save (without .pth)')
    parser.add_argument('--test', action='store_true', help='Test mode (no training, just run agent)')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode (disable GPU)')
    parser.add_argument('--phase', type=int, default=None, help='Override starting phase (1-3) for testing')
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
        render_3d=args.viz3d, 
        max_episodes=args.episodes,
        model_name=args.model,
        test_mode=args.test,
        starting_phase=args.phase
    )
