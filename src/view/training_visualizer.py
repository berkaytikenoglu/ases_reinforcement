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
from config.rewards import Rewards

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
        self.death_rates = []   # Rolling death rate
        self.fail_rates = []    # Rolling fail rate
        self.death_rate_window = deque(maxlen=50)
        self.fail_rate_window = deque(maxlen=50)
        
        # Rolling stats
        self.success_rate_window = deque(maxlen=50)
        self.success_rates = []
        
        # Plot setup
        self.fig = None
        self.axes = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Phase Tracking (USER REQUEST: Show in title)
        self.current_phase = 1
        self.phase_transitions = {}  # {episode: phase}
        
    def _build_timeline_text(self):
        """Build a sleek horizontal phase progress bar"""
        def get_fmt(p, name):
            if self.current_phase == p:
                return f"⬢ {name} (AKTİF)"
            if self.current_phase > p:
                return f"✔ {name}"
            return f"⬡ {name}"
            
        parts = [
            get_fmt(1, "NİŞAN (1-100)"),
            get_fmt(2, "TEK ATIM (100-200)"),
            get_fmt(3, "SAVAŞ (200+)")
        ]
        return "  ──  ".join(parts)
        
    def start(self):
        """Initialize the plot window"""
        self.is_running = True
        
        # Create figure with subplots (2x3 Grid)
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 3, figsize=(16, 10))
        self.fig.suptitle('ASES Training Metrics', fontsize=14, fontweight='bold')
        
        # Style
        plt.style.use('dark_background')
        self.fig.patch.set_facecolor('#1a1a2e')
        for ax in self.axes.flat:
            ax.set_facecolor('#16213e')
            ax.grid(True, alpha=0.3, color='#4a4a6a')
        
        # Row 1
        self.axes[0, 0].set_title('Bölüm Ödülleri', color='#00d4ff')
        
        self.axes[0, 1].set_title('Başarı Oranı (50 Bölüm)', color='#00ff88')
        self.axes[0, 1].set_ylim(0, 100)
        
        # Deaths (New)
        self.axes[0, 2].set_title('Toplam Ölüm (Cumulative)', color='#ff0000')
        self.axes[0, 2].set_xlabel('Bölüm')
        self.axes[0, 2].set_ylabel('Ölüm Sayısı')
        
        # Row 2
        self.axes[1, 0].set_title('İsabet / Tehdit', color='#ff6b6b')
        
        self.axes[1, 1].set_title('Mühimmat Kullanımı', color='#ffd93d')
        
        self.axes[1, 2].set_title('Hayatta Kalma Oranı (50 Bölüm)', color='#00ff00')
        self.axes[1, 2].set_ylim(0, 100)
        self.axes[1, 2].set_xlabel('Bölüm')
        self.axes[1, 2].set_ylabel('%')
        
        # Phase Timeline (Sleek Top Bar)
        self.timeline_text = self.fig.text(
            0.5, 0.94,  # Top Center
            self._build_timeline_text(),
            transform=self.fig.transFigure,
            fontsize=9,
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='center',
            color='#00ff88',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1117', edgecolor='#00ff88', alpha=0.9, linewidth=1.5)
        )

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
            
            # Track Deaths
            is_died = 1 if "DIED" in result else 0
            if len(self.deaths) > 0:
                self.deaths.append(self.deaths[-1] + is_died)
            else:
                self.deaths.append(is_died)
                
            # Track Fails
            is_failed = 1 if "FAILED" in result else 0
            if len(self.fails) > 0:
                self.fails.append(self.fails[-1] + is_failed)
            else:
                self.fails.append(is_failed)
            
            # ... existing stats code ...
        
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
            
            # Track Deaths (Rolling Window)
            is_died = 1 if "DIED" in result else 0
            self.death_rate_window.append(is_died)
            current_death_rate = np.mean(self.death_rate_window) * 100
            self.death_rates.append(current_death_rate)
                
            # Track Fails (Rolling Window)
            is_failed = 1 if "FAILED" in result else 0
            self.fail_rate_window.append(is_failed)
            current_fail_rate = np.mean(self.fail_rate_window) * 100
            self.fail_rates.append(current_fail_rate)
            
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
                # Trim new lists too
                self.death_rates = self.death_rates[-self.max_history:]
                self.fail_rates = self.fail_rates[-self.max_history:]
    
    def update_plots(self):
        """Refresh the plots with current data (OPTIMIZED: set_data)"""
        with self.lock:
            if not self.episodes:
                return

            eps = list(self.episodes)
            
            # --- INITIALIZE LINES ONCE ---
            if not hasattr(self, 'lines') or len(self.lines) == 0:
                self.lines = {}
                
                # 1. Rewards
                self.lines['reward'], = self.axes[0, 0].plot([], [], 'c-', alpha=0.4, linewidth=0.8, label='Ödül')
                self.lines['avg_reward'], = self.axes[0, 0].plot([], [], '#00d4ff', linewidth=2, label='Ort.(100)')
                self.axes[0, 0].set_title('Bölüm Ödülleri', color='#00d4ff')
                self.axes[0, 0].set_xlabel('Bölüm')
                self.axes[0, 0].set_ylabel('Ödül')
                self.axes[0, 0].legend(loc='upper left', fontsize=8)
                
                # 2. Success Rate
                self.lines['success'], = self.axes[0, 1].plot([], [], '#00ff88', linewidth=2)
                self.axes[0, 1].set_title('Başarı Oranı (50 Bölüm)', color='#00ff88')
                self.axes[0, 1].set_ylim(0, 105)
                
                # 3. Losses
                self.lines['died'], = self.axes[0, 2].plot([], [], '#ff0000', linewidth=2, label='Ölüm %')
                self.lines['failed'], = self.axes[0, 2].plot([], [], '#ff9900', linewidth=2, linestyle='--', label='Başarısız %')
                self.axes[0, 2].set_title('Kayıp Oranı (Son 50)', color='#ff4444')
                self.axes[0, 2].set_ylim(0, 105)
                self.axes[0, 2].legend(loc='upper left', fontsize=8)

                # 4. Hits vs Spawned
                self.lines['hits'], = self.axes[1, 0].plot([], [], 'g-', label='Vurulan')
                self.lines['spawned'], = self.axes[1, 0].plot([], [], 'w--', label='Üretilen')
                self.axes[1, 0].set_title('İsabet Sayısı', color='#ff6b6b')
                self.axes[1, 0].legend(loc='upper left', fontsize=8)
                
                # 5. Ammo
                self.lines['ammo'], = self.axes[1, 1].plot([], [], '#ffd93d', linewidth=1.5)
                self.axes[1, 1].set_title('Mühimmat Kullanımı', color='#ffd93d')
                self.axes[1, 1].set_xlabel('Bölüm')
                self.axes[1, 1].set_ylabel('Kullanılan Mermi')
                
                # 6. Survival Rate (USER REQUEST: Hayatta Kalma Oranı)
                self.lines['survival'], = self.axes[1, 2].plot([], [], '#00ff00', linewidth=2)
                self.axes[1, 2].set_title('Hayatta Kalma Oranı (50 Bölüm)', color='#00ff00')
                self.axes[1, 2].set_xlabel('Bölüm')
                self.axes[1, 2].set_ylabel('%')
                
            # --- UPDATE DATA ---
            self.lines['reward'].set_data(eps, self.rewards)
            self.lines['avg_reward'].set_data(eps, self.avg_rewards)
            self.lines['success'].set_data(eps, self.success_rates)
            
            if len(self.death_rates) == len(eps):
                self.lines['died'].set_data(eps, self.death_rates)
                self.lines['failed'].set_data(eps, self.fail_rates)
            
            self.lines['hits'].set_data(eps, self.hits)
            self.lines['spawned'].set_data(eps, self.spawned)
            self.lines['ammo'].set_data(eps, self.ammo_used)
            
            # Survival Rate = 100 - Death Rate
            if len(self.death_rates) == len(eps):
                survival_rates = [100 - dr for dr in self.death_rates]
                self.lines['survival'].set_data(eps, survival_rates)
            
            # --- RESCALE AXES ---
            for ax in self.axes.flat:
                # Only rescale if we have data to avoid warnings
                if len(eps) > 0:
                    ax.relim()
                    ax.autoscale_view()
            
            # --- TEXT UPDATES ---
            # Remove old text objects
            if hasattr(self, 'stats_text') and self.stats_text:
                pass # Text update in suptitle is cleaner or remove old text object logic
                
            # Global Title Update
            if len(self.rewards) > 0:
                latest_reward = self.rewards[-1]
                latest_avg = self.avg_rewards[-1]
                latest_success = self.success_rates[-1]
                best_reward_so_far = max(self.rewards)
                
                # Phase Info with Status Indicators (USER REQUEST)
                if self.current_phase == 1:
                    phase_info = "FAZ 1: NİŞAN EĞİTİMİ | 🛡️ Ölümsüz=AÇIK | 🔫 Tetik=KİLİTLİ"
                    condition = "Koşul: Avg > -1.5 & Ep > 50"
                elif self.current_phase == 2:
                    phase_info = "FAZ 2: TEK MERMİ | 🛡️ Ölümsüz=KAPALI | 🔫 Tetik=TEK ATIM"
                    condition = "Koşul: %80 Başarı & Ep > 100"
                else:
                    phase_info = "FAZ 3: TAM SAVAŞ | 🛡️ Ölümsüz=KAPALI | 🔫 Tetik=SERBEST"
                    condition = "Final Modu"
                
                # Update Timeline Text
                if hasattr(self, 'timeline_text'):
                    self.timeline_text.set_text(self._build_timeline_text())
                stats_text = (f"Bölüm: {eps[-1]} | {phase_info}\n"
                              f"Son: Ödül={latest_reward:.0f} | Ort={latest_avg:.0f} | Başarı={latest_success:.0f}% | {condition}")
                self.fig.suptitle(f'ASES OTONOM SAVUNMA AJANI\n{stats_text}', fontsize=10, fontweight='bold', color='white')

            try:
                # Use draw_idle for optimized redraw
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except Exception:
                pass
    
    def set_phase(self, phase, episode=None):
        """Update current curriculum phase (called by trainer on promotion)"""
        with self.lock:
            old_phase = self.current_phase
            self.current_phase = phase
            if episode is not None and phase != old_phase:
                self.phase_transitions[episode] = phase
    
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
