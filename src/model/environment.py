
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.parameters import Params
from config.rewards import Rewards
from src.model.entities import Threat, DefenseSystem

class DefenseEnv(gym.Env):
    def __init__(self):
        super(DefenseEnv, self).__init__()
        
        # Action Space: [Angle (-1 to 1), Fire Trigger (0 to 1)]
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        # Observation Space: [CoolDown, Wind, Ammo_Ratio, Threat1_X, Threat1_Y, Threat1_DX, Threat1_DY, ...]
        # Tracking up to 3 closest threats
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        self.width = Params.SCREEN_WIDTH
        self.height = Params.SCREEN_HEIGHT
        
        self.defense_system = None
        self.threats = []
        self.wind_force = 0
        self.steps = 0
        
        # Episode stats
        self.threats_spawned = 0
        self.threats_destroyed = 0
        self.threats_missed = 0
        self.spawn_timer = 0
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.defense_system = DefenseSystem(self.width / 2, self.height - 50, Params.RELOAD_TIME, Params.AMMO_CAPACITY)
        self.threats = []
        self.wind_force = random.uniform(-Params.MAX_WIND_FORCE, Params.MAX_WIND_FORCE)
        self.steps = 0
        
        # Reset episode stats
        self.threats_spawned = 0
        self.threats_destroyed = 0
        self.threats_missed = 0
        self.spawn_timer = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Get observation with up to 3 closest threats"""
        # Sort threats by distance to defense system
        threats_by_dist = []
        for t in self.threats:
            dist = math.hypot(t.x - self.defense_system.x, t.y - self.defense_system.y)
            threats_by_dist.append((dist, t))
        threats_by_dist.sort(key=lambda x: x[0])
        
        # Base observation
        obs = [
            self.defense_system.cooldown_timer / Params.RELOAD_TIME,
            self.wind_force / Params.MAX_WIND_FORCE,
            self.defense_system.ammo / Params.AMMO_CAPACITY,
        ]
        
        # Add up to 3 closest threats (4 values each: x, y, dx, dy)
        for i in range(3):
            if i < len(threats_by_dist):
                t = threats_by_dist[i][1]
                obs.extend([
                    t.x / self.width,
                    t.y / self.height,
                    t.dx / 100.0,
                    t.dy / 100.0
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        angle_input = float(action[0])  # -1 to 1
        fire_trigger = float(action[1])  # 0 to 1
        
        # Map angle: 90 degrees is UP, range is 30-150 degrees
        angle_rad = math.radians(90 - angle_input * 60)
        
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        # Find closest threat
        closest_threat = None
        min_dist = float('inf')
        for t in self.threats:
            dist = math.hypot(t.x - self.defense_system.x, t.y - self.defense_system.y)
            if dist < min_dist:
                min_dist = dist
                closest_threat = t

        # Survival bonus (small reward for staying alive)
        reward += Rewards.SURVIVAL_BONUS
        
        # Time step penalty (encourages efficiency)
        reward += Rewards.TIME_STEP_PENALTY
        
        # Action: Fire
        if fire_trigger > 0.5:
            if len(self.threats) == 0:
                # WASTED AMMO - firing when no threats exist
                reward += Rewards.WASTED_AMMO_PENALTY
            elif self.defense_system.fire(angle_rad, Params.PROJECTILE_SPEED):
                # Successful fire - apply base miss penalty (offset by hit reward if hits)
                reward += Rewards.MISS_PENALTY
                
                # Aim bonus if pointed at closest threat
                if closest_threat:
                    target_angle = math.atan2(
                        -(closest_threat.y - self.defense_system.y),
                        closest_threat.x - self.defense_system.x
                    )
                    angle_diff = abs(angle_rad - target_angle)
                    if angle_diff < Rewards.AIM_ANGLE_THRESHOLD:
                        reward += Rewards.AIM_BONUS
            else:
                # Cooldown violation
                reward += Rewards.COOLDOWN_VIOLATION
        
        # Risky angle penalty
        if abs(angle_input) > 0.9:
            reward += Rewards.RISKY_ANGLE_PENALTY
        
        # Update World
        self._update_physics()
        self._spawn_threats()
        self._update_wind()
        
        # Check Collisions
        hits = self._check_collisions()
        self.threats_destroyed += hits
        reward += hits * Rewards.HIT_REWARD
        
        # Check Threats Reaching Ground (GAME OVER condition)
        misses = self._check_threats_reached_base()
        self.threats_missed += misses
        
        if misses > 0:
            # Threat reached ground - episode ends!
            reward += Rewards.GROUND_HIT_PENALTY
            terminated = True
            info['reason'] = 'threat_reached_ground'
        
        # Check Win Condition - All threats spawned and destroyed
        if self.threats_spawned >= Params.THREATS_PER_EPISODE and len(self.threats) == 0:
            if self.threats_destroyed == self.threats_spawned:
                # PERFECT WIN - all threats destroyed!
                reward += Rewards.EPISODE_WIN_BONUS
                terminated = True
                info['reason'] = 'perfect_win'
            else:
                # Episode complete (threats finished spawning and all cleared)
                terminated = True
                info['reason'] = 'episode_complete'
        
        # Max steps failsafe
        if self.steps >= 3000:
            truncated = True
            
        info['hits'] = self.threats_destroyed
        info['misses'] = self.threats_missed
        info['spawned'] = self.threats_spawned
            
        return self._get_obs(), reward, terminated, truncated, info

    def _update_physics(self):
        dt = Params.DT
        
        self.defense_system.update(dt)
        for proj in self.defense_system.projectiles:
            proj.update(dt, self.wind_force, Params.GRAVITY)
            
        # Cleanup projectiles out of bounds
        self.defense_system.projectiles = [
            p for p in self.defense_system.projectiles 
            if 0 <= p.x <= self.width and 0 <= p.y <= self.height
        ]
                                           
        for t in self.threats:
            t.update(dt, self.wind_force, Params.GRAVITY)
            
    def _spawn_threats(self):
        """Spawn threats with controlled rate and max concurrent limit"""
        # Don't spawn if we've reached the episode limit
        if self.threats_spawned >= Params.THREATS_PER_EPISODE:
            return
            
        # Don't spawn if we're at max concurrent
        if len(self.threats) >= Params.MAX_CONCURRENT_THREATS:
            return
            
        # Spawn timer
        self.spawn_timer += 1
        if self.spawn_timer >= Params.SPAWN_INTERVAL:
            self.spawn_timer = 0
            
            x = random.uniform(50, self.width - 50)
            y = -400  # Spawn outside dome
            target_x = self.width / 2
            speed = random.uniform(Params.THREAT_SPEED_MIN, Params.THREAT_SPEED_MAX)
            
            self.threats.append(Threat(x, y, speed, target_x, gravity=Params.GRAVITY))
            self.threats_spawned += 1
            
    def _update_wind(self):
        if self.steps % Params.WIND_CHANGE_INTERVAL == 0:
            self.wind_force += random.uniform(-0.5, 0.5)
            self.wind_force = max(-Params.MAX_WIND_FORCE, min(Params.MAX_WIND_FORCE, self.wind_force))

    def _check_collisions(self):
        """Check projectile-threat collisions, return number of hits"""
        hits = 0
        for proj in self.defense_system.projectiles[:]:
            for t in self.threats[:]:
                dist = math.hypot(proj.x - t.x, proj.y - t.y)
                if dist < (proj.radius + t.radius):
                    # HIT!
                    if proj in self.defense_system.projectiles:
                        self.defense_system.projectiles.remove(proj)
                    if t in self.threats:
                        self.threats.remove(t)
                        hits += 1
                        break
        return hits

    def _check_threats_reached_base(self):
        """Check if any threats reached the ground"""
        misses = 0
        for t in self.threats[:]:
            # Threat reached ground (with some buffer for visual)
            if t.y >= self.height + 50:
                self.threats.remove(t)
                misses += 1
        return misses
    
    def get_episode_stats(self):
        """Return current episode statistics"""
        return {
            'spawned': self.threats_spawned,
            'destroyed': self.threats_destroyed,
            'missed': self.threats_missed,
            'remaining': len(self.threats),
            'ammo': self.defense_system.ammo
        }
