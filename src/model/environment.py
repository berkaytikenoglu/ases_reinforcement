
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
    def __init__(self, curriculum_phase=1):
        super(DefenseEnv, self).__init__()
        
        self.curriculum_phase = curriculum_phase
        
        # Action Space: [Angle (-1 to 1), Fire Trigger (0 to 1)]
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        
        # RADAR-ENABLED OBSERVATION SPACE (Fixed Size: 15)
        # 0: HAS_TARGET (Binary: 1.0 = target exists, 0.0 = no target) <<< RADAR SENSOR
        # 1: CAN_FIRE (Binary: 1.0 = weapon ready, 0.0 = cooldown/no ammo) <<< RADAR SENSOR
        # 2: Aim Error (Rad, amplified x5)
        # 3: Relative Angle to Target (Sin)
        # 4: Relative Angle to Target (Cos)
        # 5: Target Distance (Normalized)
        # 6: Target Closing Speed (Normalized)
        # 7: Time to Impact (Normalized)
        # 8: Wind Deflection Effect
        # 9: Ammo Ratio
        # 10: Threat Density (Count / Max)
        # 11-14: Reserved / Zeros
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        self.width = Params.SCREEN_WIDTH
        self.height = Params.SCREEN_HEIGHT
        
        self.defense_system = None
        self.threats = []
        self.wind_force = 0
        self.steps = 0
        
        # Stats
        self.threats_spawned = 0
        self.threats_destroyed = 0
        self.threats_missed = 0
        self.threats_hit_agent = 0
        self.agent_health = Params.AGENT_HEALTH
        self.shots_fired = 0
        self.spawn_timer = 0
        self.hit_streak = 0
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.defense_system = DefenseSystem(self.width / 2, self.height - 50, Params.RELOAD_TIME, Params.AMMO_CAPACITY)
        self.threats = []
        self.wind_force = random.uniform(-Params.MAX_WIND_FORCE, Params.MAX_WIND_FORCE)
        self.steps = 0
        
        self.threats_spawned = 0
        self.threats_destroyed = 0
        self.threats_missed = 0
        self.threats_hit_agent = 0
        self.agent_health = Params.AGENT_HEALTH
        self.shots_fired = 0
        self.spawn_timer = 100 # Instant spawn logic
        self.hit_streak = 0
        
        return self._get_obs(), {}
    
    def _calculate_ideal_angle(self, target):
        """Calculate the ballistic angle required to hit the target"""
        # Simplified for Bootcamp: Pure Geometry
        # Since Gravity is 5.0 and Speed is 300.0, the bullet is basically a laser.
        # No need for complex ballistic compensation which was causing errors.
        
        dx = target.x - self.defense_system.x
        dy = -(target.y - self.defense_system.y) # Positive Up
        
        # Base angle
        angle_rad = math.atan2(dy, dx)
        
        return angle_rad

    def _get_obs(self):
        """Construct SMART Observation Vector with RADAR SENSORS"""
        # Find critical target (closest / most dangerous)
        target = None
        min_tti = float('inf')
        
        for t in self.threats:
            dist = math.hypot(t.x - self.defense_system.x, t.y - self.defense_system.y)
            speed = math.hypot(t.dx, t.dy)
            tti = dist / (speed + 1e-5)
            if tti < min_tti:
                min_tti = tti
                target = t
        
        obs = np.zeros(15, dtype=np.float32)
        
        # ===== RADAR SENSORS (BINARY FLAGS) =====
        # 0: HAS_TARGET - Critical! Agent must know if there's something to shoot at
        obs[0] = 1.0 if target is not None else 0.0
        
        # 1: CAN_FIRE - Is the weapon ready? (Ammo > 0 AND no cooldown)
        can_fire = (self.defense_system.ammo > 0 and self.defense_system.cooldown_timer <= 0)
        obs[1] = 1.0 if can_fire else 0.0
        
        # ===== GLOBAL AGENT STATE =====
        # 9: Ammo Ratio
        obs[9] = self.defense_system.ammo / Params.AMMO_CAPACITY
        # 10: Threat Density
        obs[10] = len(self.threats) / max(1, Params.MAX_CONCURRENT_THREATS)
        
        # ===== TARGET-SPECIFIC DATA (only if target exists) =====
        if target:
            # Physics calculations
            dx = target.x - self.defense_system.x
            dy = -(target.y - self.defense_system.y)
            dist = math.hypot(dx, dy)
            
            # Ideal Angle
            ideal_angle = self._calculate_ideal_angle(target)
            current_angle = self.defense_system.angle
            
            # 2: Aim Error (Rad) -> THE MOST IMPORTANT FEATURE
            # AMPLIFIED SIGNAL: Multiply by 5.0 so 0.2 rad (11 deg) becomes 1.0 (Full signal)
            aim_error = (ideal_angle - current_angle)
            obs[2] = aim_error * 5.0 
            
            # 3-4: Relative Angle
            obs[3] = math.sin(ideal_angle)
            obs[4] = math.cos(ideal_angle)
            
            # 5: Distance
            obs[5] = dist / 1000.0
            
            # 6: Closing Speed (Target getting closer?)
            obs[6] = target.dy / 10.0 
            
            # 7: Time to Impact
            obs[7] = min_tti / 10.0
            
            # 8: Wind Deflection
            obs[8] = self.wind_force
            
        return obs

    def step(self, action):
        self.steps += 1
        angle_input = float(action[0])  # -1 to 1
        fire_trigger = float(action[1])  # 0 to 1
        
        # Map angle: 90 degrees is UP, range is 30-150 degrees
        # Fixed mapping: 90 + input*60
        target_angle_deg = 90 - angle_input * 60
        target_angle_rad = math.radians(target_angle_deg)
        
        # Smoothly rotate turret towards target angle (Simulate servo speed)
        # Or instant for RL? Let's keep it instant for strong correlation.
        self.defense_system.angle = target_angle_rad 
        
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        # --- CURRICULUM LOGIC ---
        
        # Phase 1: AIM ONLY (No firing allowed, reward based on alignment)
        # Phase 1: AIM ONLY (No firing allowed, reward based on alignment)
        if self.curriculum_phase == 1:
            obs = self._get_obs()
            has_target = obs[0]
            raw_aim_error = abs(obs[2] / 5.0)  # Raw radians error
            
            # --- STRICT DISCIPLINE REWARD (User Request) ---
            
            if has_target > 0.5:
                # 1. TRACKING DISCIPLINE
                # Punishment for NOT locking on. Ideally aim_error should be 0.
                reward -= raw_aim_error * 2.0 
                
                # Bonus for being LOCKED ON (The only way to get validation)
                if raw_aim_error < 0.1: # approx 6 degrees
                    reward += 1.0
            else:
                # 2. STABILITY PENALTY
                # If no target, changing angle randomly is bad. Punishment for high action magnitude.
                # action[0] controls target angle. We want to discourage rapid changes.
                # Assuming 'angle_input' (action[0]) is effectively velocity/change in this context
                action_mag = abs(float(action[0]))
                reward -= action_mag * 0.5
                
            # Prevent firing in Phase 1
            fire_trigger = 0.0
            
        # Phase 2: ONE SHOT (Episode ends after 1 shot)
        elif self.curriculum_phase == 2:
            pass # We will terminate after the shot result is known or ammo is out
                
        # Phase 3: FULL GAME (Normal rewards)
        elif self.curriculum_phase == 3:
            pass # Standard logic
            
        # ------------------------
        
        # Physics Update
        self._update_physics()
        self._spawn_threats()
        self._update_wind()
        
        # --- GUIDANCE REWARD (Alignment) ---
        # User Request: "Encourage aiming at threat"
        # Constant small reward for keeping the crosshair on target
        # NEW: Check HAS_TARGET first, then use obs[2] for Aim Error
        obs_current = self._get_obs()
        has_target = obs_current[0]  # 1.0 if target exists
        raw_aim_error = obs_current[2] / 5.0  # Aim Error is now at index 2
        
        if has_target > 0.5 and abs(raw_aim_error) < 0.2:  # Only if target exists!
            alignment_bonus = (0.2 - abs(raw_aim_error)) * 5.0 # Max +1.0 per step
            reward += alignment_bonus
        # -----------------------------------
        
        # Fire Logic (Standard RL: Agent learns by mistake)
        # Removed Trigger Safety per user request (Reinforcement Learning philosophy)
        if fire_trigger > 0.5 and self.curriculum_phase != 1 and not (self.curriculum_phase == 2 and self.shots_fired >= 1):
             if self.defense_system.fire(self.defense_system.angle, Params.PROJECTILE_SPEED):
                 self.shots_fired += 1
                 # Real Cost: Use the global configuration (-100.0)
                 reward += Rewards.FIRE_PENALTY
                 
                 # ===== BLIND FIRE PENALTY =====
                 # If firing when NO TARGET exists, massive penalty!
                 # This teaches: "Don't shoot at empty sky"
                 if has_target < 0.5:  # No target visible
                     reward += Rewards.WASTED_AMMO_PENALTY  # -500 for blind fire
                 # ==============================
        
        # Collisions
        hits = self._check_collisions()
        self.threats_destroyed += hits
        
        if hits > 0:
            if self.curriculum_phase == 1:
                pass # No shooting, no hits possible
            elif self.curriculum_phase == 2:
                reward += 2000.0 # Phase 2 Hit Reward (Bigger incentive)
                terminated = True
                info['result'] = 'SUCCESS'
            else:
                reward += Rewards.HIT_REWARD * hits
        
        # Misses (Ground Hits)
        misses = self._check_threats_reached_base()
        self.threats_missed += misses
        
        if misses > 0:
            if self.curriculum_phase == 1:
                pass # FAZ 1: Kaçırma cezası yok (Sadece odaklan)
            elif self.curriculum_phase == 2:
                reward -= 500.0 # Phase 2 Miss Penalty (Stronger punishment)
                terminated = True
                info['result'] = 'FAILED'
            elif self.curriculum_phase == 3:
                reward += Rewards.GROUND_HIT_PENALTY * misses
        
        # Agent Death
        agent_hits = self._check_agent_collisions()
        if agent_hits > 0:
            if self.curriculum_phase == 1:
                pass # FAZ 1: Ölümsüzlük (Eğitime devam)
            else:
                self.agent_health -= agent_hits
                if self.agent_health <= 0:
                    reward += Rewards.DEATH_PENALTY
                    terminated = True
                    info['reason'] = 'agent_died'

        # Win Condition (Phase 1 & 3)
        target_threats = Params.PHASE1_THREATS_PER_EPISODE if self.curriculum_phase == 1 else Params.THREATS_PER_EPISODE
        
        if self.curriculum_phase != 2 and self.threats_spawned >= target_threats and len(self.threats) == 0:
             terminated = True
             info['reason'] = 'episode_complete'
             if self.curriculum_phase == 3 and self.threats_destroyed == self.threats_spawned:
                 reward += Rewards.EPISODE_WIN_BONUS
                 
        # Phase 2: Terminate only after the shot is resolved
        if self.curriculum_phase == 2 and self.shots_fired >= 1 and len(self.defense_system.projectiles) == 0:
            terminated = True
            if 'result' not in info: info['result'] = 'FAILED'

        # Max steps
        if self.steps >= 2000:
            truncated = True

        info['hits'] = self.threats_destroyed
        info['spawned'] = self.threats_spawned
        info['ammo'] = self.defense_system.ammo  # Remaining ammo
        info['shots'] = self.shots_fired         # Shots taken this episode
    
        return self._get_obs(), reward, terminated, truncated, info

    def _update_physics(self):
        dt = Params.DT
        
        self.defense_system.update(dt)
        for proj in self.defense_system.projectiles:
            proj.update(dt, self.wind_force, Params.GRAVITY)
            
        # Cleanup projectiles out of bounds (extended for upward shots)
        self.defense_system.projectiles = [
            p for p in self.defense_system.projectiles 
            if -500 <= p.x <= self.width + 500 and -500 <= p.y <= self.height + 100
        ]
                                           
        for t in self.threats:
            t.update(dt, self.wind_force, Params.GRAVITY)
            
    def _spawn_threats(self):
        """Spawn threats with controlled rate and max concurrent limit"""
        # Phase 1: Faster episodes (less threats, faster spawn)
        if self.curriculum_phase == 1:
            max_threats = Params.PHASE1_THREATS_PER_EPISODE
            spawn_interval = Params.PHASE1_SPAWN_INTERVAL
        else:
            max_threats = Params.THREATS_PER_EPISODE
            spawn_interval = Params.SPAWN_INTERVAL
            
        # Don't spawn if we've reached the episode limit
        if self.threats_spawned >= max_threats:
            return
            
        # Don't spawn if we're at max concurrent
        if len(self.threats) >= Params.MAX_CONCURRENT_THREATS:
            return
            
        # Spawn timer
        self.spawn_timer += 1
        if self.spawn_timer >= spawn_interval:
            self.spawn_timer = 0
            
            x = random.uniform(50, self.width - 50)
            y = -400  # Spawn outside dome
            target_x = self.width / 2
            
            # Progressive Difficulty: Slower threats in Phase 1
            if self.curriculum_phase == 1:
                speed = random.uniform(Params.PHASE1_THREAT_SPEED_MIN, Params.PHASE1_THREAT_SPEED_MAX)
            else:
                speed = random.uniform(Params.THREAT_SPEED_MIN, Params.THREAT_SPEED_MAX)
            
            self.threats.append(Threat(x, y, speed, target_x, gravity=Params.GRAVITY))
            self.threats_spawned += 1
            
    def _update_wind(self):
        if self.steps % Params.WIND_CHANGE_INTERVAL == 0:
            self.wind_force += random.uniform(-0.5, 0.5)
            self.wind_force = max(-Params.MAX_WIND_FORCE, min(Params.MAX_WIND_FORCE, self.wind_force))

    def _check_collisions(self):
        """Check projectile-threat collisions, return number of hits and engagement rewards"""
        hits = 0
        engagement_reward = 0
        
        for proj in self.defense_system.projectiles[:]:
            for t in self.threats[:]:
                dist = math.hypot(proj.x - t.x, proj.y - t.y)
                if dist < (proj.radius + t.radius):
                    # HIT!
                    if proj in self.defense_system.projectiles:
                        self.defense_system.projectiles.remove(proj)
                    if t in self.threats:
                        # Calculate distance to agent for strategy mode
                        dist_to_agent = math.hypot(t.x - self.defense_system.x, t.y - self.defense_system.y)
                        
                        # ===== HETEROJEN STRATEJİ MODU =====
                        # Distance Scaling Factor (0.0 - 1.0+)
                        # Max range approx 800 pixels
                        dist_factor = min(1.0, dist_to_agent / 800.0)
                        dist_bonus = Rewards.HIT_REWARD * Rewards.DISTANCE_MULTIPLIER * dist_factor
                        engagement_reward += dist_bonus
                        
                        if dist_to_agent > Rewards.EARLY_ENGAGEMENT_DISTANCE:
                            # UZAK MESAFE (SCOPE): Çok erken önleme
                            engagement_reward += Rewards.EARLY_HIT_BONUS
                            engagement_reward += Rewards.AREA_DEFENSE_HIT_BONUS # Extra teşvik
                        elif dist_to_agent > Rewards.AREA_DEFENSE_DISTANCE:
                            # AREA DEFENSE MODE: Yüksek irtifa - parçalayıcı alan savunması
                            engagement_reward += Rewards.AREA_DEFENSE_HIT_BONUS
                            engagement_reward += Rewards.STRATEGY_MATCH_BONUS  # Doğru mod!
                        elif dist_to_agent < Rewards.RAPID_FIRE_DISTANCE:
                            # RAPID FIRE MODE: Kritik yakınlık - odaklı seri atış
                            engagement_reward += Rewards.RAPID_FIRE_PRECISION_BONUS 
                            engagement_reward += Rewards.STRATEGY_MATCH_BONUS  # Doğru mod!
                        else:
                            # ORTA MESAFE: Normal engagement bonus
                            engagement_reward += 100.0 # Standart bonus
                        
                        self.threats.remove(t)
                        hits += 1
                        break
        
        self.last_engagement_reward = engagement_reward
        return hits

    def _check_agent_collisions(self):
        """Check if any threat hit the defense system directly"""
        hits = 0
        # Defense System Hitbox (Approximate)
        agent_radius = 20.0
        
        for t in self.threats[:]:
            dist = math.hypot(t.x - self.defense_system.x, t.y - self.defense_system.y)
            if dist < (t.radius + agent_radius):
                # AGENT HIT!
                self.threats.remove(t)
                hits += 1
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
