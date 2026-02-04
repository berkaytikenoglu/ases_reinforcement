"""
Real-time Training Metrics Visualization
Generates live graphs during training showing:
- Episode Rewards
- Success Rate (rolling average)
- Hits per Episode
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for live updates
import numpy as np
from collections import deque
import threading
import threading
import time
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.parameters import Params

class TrainingVisualizer:
    def __init__(self, max_history=500):
        self.max_history = max_history
        
        # Data storage
        self.episodes = []
        self.rewards = []
        self.avg_rewards = []
        self.hits = []
        self.spawned = []
        self.hit_rates = []
        self.ammo_used = []
        self.results = []  # 'SUCCESS', 'FAILED', 'DIED'
        
        # Rolling stats
        self.success_rate_window = deque(maxlen=50)
        self.success_rates = []
        
        # Plot setup
        self.fig = None
        self.axes = None
        self.is_running = False
        self.lock = threading.Lock()
        
    def start(self):
        """Initialize the plot window"""
        self.is_running = True
        
        # Create figure with subplots
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('ASES Training Metrics', fontsize=14, fontweight='bold')
        
        # Style
        plt.style.use('dark_background')
        self.fig.patch.set_facecolor('#1a1a2e')
        for ax in self.axes.flat:
            ax.set_facecolor('#16213e')
            ax.grid(True, alpha=0.3, color='#4a4a6a')
        
        # Labels (TÜRKÇE)
        self.axes[0, 0].set_title('Bölüm Ödülleri', color='#00d4ff')
        self.axes[0, 0].set_xlabel('Bölüm (Episode)')
        self.axes[0, 0].set_ylabel('Ödül')
        
        self.axes[0, 1].set_title('Başarı Oranı (50 Bölüm Ort.)', color='#00ff88')
        self.axes[0, 1].set_xlabel('Bölüm (Episode)')
        self.axes[0, 1].set_ylabel('Başarı %')
        self.axes[0, 1].set_ylim(0, 100)
        
        self.axes[1, 0].set_title('İsabet Sayısı', color='#ff6b6b')
        self.axes[1, 0].set_xlabel('Bölüm (Episode)')
        self.axes[1, 0].set_ylabel('İsabet / Tehdit')
        
        self.axes[1, 1].set_title('Mühimmat Kullanımı', color='#ffd93d')
        self.axes[1, 1].set_xlabel('Bölüm (Episode)')
        self.axes[1, 1].set_ylabel('Kullanılan Mermi')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
    def add_episode(self, episode, reward, avg_reward, hits, spawned, ammo_used, result):
        """Add data for a completed episode"""
        with self.lock:
            self.episodes.append(episode)
            self.rewards.append(reward)
            self.avg_rewards.append(avg_reward)
            self.hits.append(hits)
            self.spawned.append(spawned)
            self.ammo_used.append(ammo_used)
            self.results.append(result)
            
            # Calculate hit rate
            hit_rate = (hits / spawned * 100) if spawned > 0 else 0
            self.hit_rates.append(hit_rate)
            
            # Update success rate
            is_success = 1 if result == 'SUCCESS' else 0
            self.success_rate_window.append(is_success)
            success_rate = np.mean(self.success_rate_window) * 100
            self.success_rates.append(success_rate)
            
            # Trim if too long
            if len(self.episodes) > self.max_history:
                self.episodes = self.episodes[-self.max_history:]
                self.rewards = self.rewards[-self.max_history:]
                self.avg_rewards = self.avg_rewards[-self.max_history:]
                self.hits = self.hits[-self.max_history:]
                self.spawned = self.spawned[-self.max_history:]
                self.hit_rates = self.hit_rates[-self.max_history:]
                self.ammo_used = self.ammo_used[-self.max_history:]
                self.results = self.results[-self.max_history:]
                self.success_rates = self.success_rates[-self.max_history:]
    
    def update_plots(self):
        """Refresh the plots with current data"""
        if not self.is_running or self.fig is None:
            return
            
        with self.lock:
            if len(self.episodes) < 2:
                return
                
            eps = self.episodes
            
            # Clear and redraw
            for ax in self.axes.flat:
                ax.clear()
                ax.set_facecolor('#16213e')
                ax.grid(True, alpha=0.3, color='#4a4a6a')
            
            # Reward plot
            self.axes[0, 0].plot(eps, self.rewards, 'c-', alpha=0.4, linewidth=0.8, label='Ödül')
            self.axes[0, 0].plot(eps, self.avg_rewards, '#00d4ff', linewidth=2, label='Ort.(100)')
            self.axes[0, 0].set_title('Bölüm Ödülleri', color='#00d4ff')
            self.axes[0, 0].set_xlabel('Bölüm')
            self.axes[0, 0].set_ylabel('Ödül')
            self.axes[0, 0].legend(loc='upper left', fontsize=8)
            
            # Success rate plot
            self.axes[0, 1].plot(eps, self.success_rates, '#00ff88', linewidth=2)
            self.axes[0, 1].fill_between(eps, self.success_rates, alpha=0.3, color='#00ff88')
            self.axes[0, 1].set_title('Başarı Oranı (50 Bölüm)', color='#00ff88')
            self.axes[0, 1].set_xlabel('Bölüm')
            self.axes[0, 1].set_ylabel('Başarı %')
            self.axes[0, 1].set_ylim(0, 100)
            
            # Hits plot (bar for hits, line for spawned)
            self.axes[1, 0].bar(eps, self.hits, color='#ff6b6b', alpha=0.7, width=1.0, label='İsabet')
            self.axes[1, 0].plot(eps, self.spawned, 'w--', linewidth=1.5, alpha=0.8, label='Toplam Tehdit')
            self.axes[1, 0].set_title('İsabet Sayısı', color='#ff6b6b')
            self.axes[1, 0].set_xlabel('Bölüm')
            self.axes[1, 0].set_ylabel('Adet')
            self.axes[1, 0].legend(loc='upper left', fontsize=8)
            
            # Ammo efficiency
            self.axes[1, 1].plot(eps, self.ammo_used, '#ffd93d', linewidth=1.5)
            self.axes[1, 1].fill_between(eps, self.ammo_used, alpha=0.3, color='#ffd93d')
            self.axes[1, 1].set_title('Mühimmat Kullanımı', color='#ffd93d')
            self.axes[1, 1].set_xlabel('Bölüm')
            self.axes[1, 1].set_ylabel('Kullanılan Mermi')
            
            # Stats text (TÜRKÇE ve BEST eklendi)
            if len(self.rewards) > 0:
                latest_reward = self.rewards[-1]
                latest_avg = self.avg_rewards[-1]
                latest_success = self.success_rates[-1]
                latest_hits = self.hits[-1]
                best_reward_so_far = max(self.rewards)
                
                stats_text = f"Tehdit: {Params.THREATS_PER_EPISODE} | Mermi: {Params.AMMO_CAPACITY}\nSon: Ödül={latest_reward:.0f} | Ort={latest_avg:.0f} | Başarı={latest_success:.0f}% | En İyi={best_reward_so_far:.0f}"
                self.fig.suptitle(f'ASES Eğitim Metrikleri\n{stats_text}', fontsize=10, fontweight='bold', color='white')
            
            plt.tight_layout()
            
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
        except Exception:
            pass
    
    def close(self):
        """Close the visualization"""
        self.is_running = False
        if self.fig:
            plt.close(self.fig)
            
    def save_figure(self, path):
        """Save current plots to file"""
        if self.fig:
            self.fig.savefig(path, dpi=150, facecolor='#1a1a2e', edgecolor='none')
            print(f"Training metrics saved to {path}")


# Singleton instance for easy access
_visualizer = None

def get_visualizer():
    global _visualizer
    if _visualizer is None:
        _visualizer = TrainingVisualizer()
    return _visualizer

def start_visualizer():
    viz = get_visualizer()
    viz.start()
    return viz

def close_visualizer():
    global _visualizer
    if _visualizer:
        _visualizer.close()
        _visualizer = None
