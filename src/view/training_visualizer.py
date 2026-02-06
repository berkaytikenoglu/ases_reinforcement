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
from matplotlib.ticker import MaxNLocator
import threading
import time
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.parameters import Params
from config.rewards import Rewards

class TrainingVisualizer:
    def __init__(self, max_history=500, max_episodes=1000):
        self.max_history = max_history
        self.max_episodes = max_episodes
        
        # Data storage
        self.episodes = []
        self.rewards = []
        self.avg_rewards = []
        self.hits = []
        self.misses = []  # NEW: Missed threats
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
        
        # IFF Hit Rates
        self.enemy_hit_rates = []
        self.friendly_hit_rates = []
        
        # New: IFF Stats
        self.friendly_fires = []
        
        # Plot setup
        self.fig = None
        self.axes = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Phase Tracking (USER REQUEST: Show in title)
        self.current_phase = 1
        self.phase_transitions = {}  # {episode: phase}
        
    def _build_timeline_text(self):
        """Build a sleek horizontal phase progress bar (Simplified - No misleading ranges)"""
        def get_fmt(p, name):
            if self.current_phase == p:
                return f"⬢ {name} (AKTİF)"
            if self.current_phase > p:
                return f"✔ {name} (GEÇİLDİ)"
            return f"⬡ {name}"
            
        parts = [
            get_fmt(1, "NİŞAN EĞİTİMİ"),
            get_fmt(2, "SNIPER MODU"),
            get_fmt(3, "IFF & UAV"),
            get_fmt(4, "SAVAŞ MODU")
        ]
        return "  ──  ".join(parts)
        
    def start(self):
        """Initialize the plot window"""
        self.is_running = True
        
        # Create figure with subplots (2x3 Grid)
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 3, figsize=(16, 10))
        self.fig.suptitle('ASES Training Metrics', fontsize=14, fontweight='bold')
        
        # Reserve space for Header & Timeline
        self.fig.subplots_adjust(top=0.85) # Push plots down to make room
        
        # Style
        plt.style.use('dark_background')
        self.fig.patch.set_facecolor('#1a1a2e')
        for ax in self.axes.flat:
            ax.set_facecolor('#16213e')
            ax.grid(True, alpha=0.3, color='#4a4a6a')
        
        # ... (titles row 1) ...

        # Phase Timeline (Sleek Top Bar)
        self.timeline_text = self.fig.text(
            0.5, 0.91,  # Centered in the GAP (0.85 - 0.95)
            self._build_timeline_text(),
            transform=self.fig.transFigure,
            fontsize=9,
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='center',
            color='#00ff88',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1117', edgecolor='#00ff88', alpha=0.9, linewidth=1.5)
        )

        # DUPLICATE REMOVED
        
        
    def add_episode(self, episode, reward, avg_reward, hits, spawned, ammo_used, result, friendly_fire=0, enemies_spawned=None):
        """Add data for a completed episode"""
        if enemies_spawned is None: enemies_spawned = spawned # Fallback
        with self.lock:
            self.episodes.append(episode)
            self.rewards.append(reward)
            self.avg_rewards.append(avg_reward)
            self.hits.append(hits)
            self.spawned.append(spawned)
            self.misses.append(spawned - hits) # Calculate misses
            self.friendly_fires.append(friendly_fire)
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
            
            # Calculate hit rate (Total - Legacy)
            hit_rate = (hits / spawned * 100) if spawned > 0 else 0
            self.hit_rates.append(hit_rate)
            
            # --- IFF ACCURACY RATES ---
            # 1. Enemy Hit Rate (Good)
            if enemies_spawned > 0:
                enemy_rate = (hits / enemies_spawned) * 100
                if enemy_rate > 100: enemy_rate = 100 # Cap
            else:
                enemy_rate = 0
            self.enemy_hit_rates.append(enemy_rate)
            
            # 2. Friendly Hit Rate (Bad)
            friendlies_spawned = spawned - enemies_spawned
            if friendlies_spawned > 0:
                friendly_rate = (friendly_fire / friendlies_spawned) * 100
                if friendly_rate > 100: friendly_rate = 100 # Cap
            else:
                friendly_rate = 0
            self.friendly_hit_rates.append(friendly_rate)
            
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
                self.misses = self.misses[-self.max_history:]
                self.spawned = self.spawned[-self.max_history:]
                self.hit_rates = self.hit_rates[-self.max_history:]
                self.ammo_used = self.ammo_used[-self.max_history:]
                self.results = self.results[-self.max_history:]
                self.success_rates = self.success_rates[-self.max_history:]
                # Trim new lists too
                self.death_rates = self.death_rates[-self.max_history:]
                self.fail_rates = self.fail_rates[-self.max_history:]
                self.fail_rates = self.fail_rates[-self.max_history:]
                self.friendly_fires = self.friendly_fires[-self.max_history:]
                self.enemy_hit_rates = self.enemy_hit_rates[-self.max_history:]
                self.friendly_hit_rates = self.friendly_hit_rates[-self.max_history:]
    
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
                
                # 2. IFF Accuracy (Replacing Success Rate)
                self.lines['enemy_rate'], = self.axes[0, 1].plot([], [], '#00ff88', linewidth=2, label='Düşman Vurma %')
                self.lines['friendly_rate'], = self.axes[0, 1].plot([], [], '#ff0000', linewidth=2, label='Dost Vurma %')
                self.axes[0, 1].set_title('IFF Performansı (Hedef Vurma)', color='#ffffff')
                self.axes[0, 1].set_ylim(0, 105)
                self.axes[0, 1].legend(loc='upper left', fontsize=8)
                
                # 3. Outcome Rates (Consolidated: Success / Fail / Death)
                self.lines['outcome_success'], = self.axes[0, 2].plot([], [], '#00ff88', linewidth=2, label='Başarı %')
                self.lines['outcome_failed'], = self.axes[0, 2].plot([], [], '#ff9900', linewidth=2, linestyle='--', label='Başarısız %')
                self.lines['outcome_died'], = self.axes[0, 2].plot([], [], '#ff0000', linewidth=2, linestyle=':', label='Ölüm %')
                self.axes[0, 2].set_title('Sonuç Dağılımı (Son 50)', color='#ffffff')
                self.axes[0, 2].set_ylim(0, 105)
                self.axes[0, 2].legend(loc='upper left', fontsize=8)

                # 4. Hits vs Spawned
                self.lines['hits'], = self.axes[1, 0].plot([], [], 'g-', label='Vurulan')
                self.lines['missed'], = self.axes[1, 0].plot([], [], 'r--', alpha=0.6, label='Kaçan')
                self.lines['spawned'], = self.axes[1, 0].plot([], [], 'w:', alpha=0.3, label='Toplam')
                self.axes[1, 0].set_title('İsabet Sayısı', color='#ff6b6b')
                self.axes[1, 0].legend(loc='upper left', fontsize=8)
                
                # 5. Ammo
                self.lines['ammo'], = self.axes[1, 1].plot([], [], '#ffd93d', linewidth=1.5)
                self.axes[1, 1].set_title('Mühimmat Kullanımı', color='#ffd93d')
                self.axes[1, 1].set_xlabel('Bölüm')
                self.axes[1, 1].set_ylabel('Kullanılan Mermi')
                self.axes[1, 1].set_ylim(bottom=0) # USER REQUEST: Fix negative axis
                
                # 6. Friendly Fire (Replacing Survival Rate)
                self.lines['friendly_fire'], = self.axes[1, 2].plot([], [], '#ff00ff', linewidth=2, label='Dost Ateşi')
                self.axes[1, 2].set_title('Dost Ateşi (Friendly Fire)', color='#ff00ff')
                self.axes[1, 2].set_xlabel('Bölüm')
                self.axes[1, 2].set_ylabel('Vuruş Sayısı')
                self.axes[1, 2].set_ylim(0, 20) # USER REQUEST: Fix "0.00-0.05" scale -> Range 0-20
                self.axes[1, 2].yaxis.set_major_locator(MaxNLocator(integer=True))
                
            # --- UPDATE DATA ---
            self.lines['reward'].set_data(eps, self.rewards)
            self.lines['avg_reward'].set_data(eps, self.avg_rewards)
            
            self.lines['enemy_rate'].set_data(eps, self.enemy_hit_rates)
            self.lines['friendly_rate'].set_data(eps, self.friendly_hit_rates)
            if len(self.death_rates) == len(eps):
                self.lines['outcome_success'].set_data(eps, self.success_rates)
                self.lines['outcome_failed'].set_data(eps, self.fail_rates)
                self.lines['outcome_died'].set_data(eps, self.death_rates)
                
            # Friendly Fire
            self.lines['friendly_fire'].set_data(eps, self.friendly_fires)
            
            self.lines['hits'].set_data(eps, self.hits)
            self.lines['missed'].set_data(eps, self.misses)
            self.lines['spawned'].set_data(eps, self.spawned)
            self.lines['ammo'].set_data(eps, self.ammo_used)
            
            
            # --- RESCALE AXES ---
            for ax in self.axes.flat:
                # Only rescale if we have data to avoid warnings
                if len(eps) > 0:
                    ax.relim()
                    ax.autoscale_view()
            
            # --- CUSTOM AXIS FOR AMMO (Fixed scale based on Phase) ---
            if self.current_phase == 1:
                self.axes[1, 1].set_ylim(0, Params.AMMO_CAPACITY + 5)
            elif self.current_phase == 2:
                self.axes[1, 1].set_ylim(0, Params.PHASE2_AMMO + 5)
            else:
                self.axes[1, 1].set_ylim(0, Params.PHASE3_AMMO + 5)
            
            # --- TEXT UPDATES ---
            # Remove old text objects
            if hasattr(self, 'stats_text') and self.stats_text:
                pass # Text update in suptitle is cleaner or remove old text object logic
                
            # Global Title Update
            if len(self.rewards) > 0:
                latest_reward = self.rewards[-1]
                latest_avg = self.avg_rewards[-1]
                latest_success = self.success_rates[-1]
                
                # Phase Info with Status Indicators (USER REQUEST)
                if self.current_phase == 1:
                    phase_info = "FAZ 1: NİŞAN EĞİTİMİ"
                elif self.current_phase == 2:
                    phase_info = "FAZ 2: TEK MERMİ"
                elif self.current_phase == 3:
                     phase_info = "FAZ 3: IFF & UAV"
                else:
                    phase_info = "FAZ 4: TAM SAVAŞ"
                
                # Update Timeline Text
                if hasattr(self, 'timeline_text'):
                    self.timeline_text.set_text(self._build_timeline_text())
                
                # CLEAN MINIMALIST HEADER (User Request)
                # Calculate Best Reward
                max_reward = max(self.rewards)
                best_ep_idx = self.rewards.index(max_reward)
                best_ep = eps[best_ep_idx]
                
                # Status Flags (Inferred)
                god_mode_status = "AÇIK" if Params.AGENT_HEALTH > 10 else "KAPALI"
                fire_status = "AÇIK" # Always open in training
                
                header_text = (f"ASES OTONOM SAVUNMA AJANI | {phase_info}\n"
                               f"Hedef: {self.max_episodes} Bölüm | İlerleme: {eps[-1]} ({eps[-1]/self.max_episodes*100:.1f}%) | Ort: {latest_avg:.0f} | Best: {max_reward:.0f} (Ep {best_ep})\n"
                               f"Ölümsüzlük: {god_mode_status} | Ateş: {fire_status}")
                               
                self.fig.suptitle(header_text, fontsize=10, fontweight='bold', color='white')

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

def get_visualizer(max_episodes=1000):
    global _visualizer
    if _visualizer is None:
        _visualizer = TrainingVisualizer(max_episodes=max_episodes)
    return _visualizer

def start_visualizer(max_episodes=1000):
    viz = get_visualizer(max_episodes)
    viz.start()
    return viz

def close_visualizer():
    global _visualizer
    if _visualizer:
        _visualizer.close()
        _visualizer = None
