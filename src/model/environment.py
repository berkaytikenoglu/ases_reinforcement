
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
import time
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
        
        # Configure Phase-Specific Limits (USER REQUEST)
        if self.curriculum_phase == 1:
            self.max_threats = Params.PHASE1_THREATS_PER_EPISODE
            self.defense_system.ammo = Params.AMMO_CAPACITY # Standard 60
        elif self.curriculum_phase == 2:
            self.max_threats = Params.PHASE2_THREATS         # 3 Threats
            self.defense_system.ammo = Params.PHASE2_AMMO    # 3 Ammo (Strict)
            self.defense_system.cooldown_max = Params.PHASE2_RELOAD_TIME # Configurable Sniper Cooldown
        else:
            self.max_threats = Params.PHASE3_THREATS         # 10 Threats
            self.defense_system.ammo = Params.PHASE3_AMMO    # 30 Ammo
            self.defense_system.cooldown_max = Params.RELOAD_TIME # Normal (5)
            
        self.threats = []
        self.wind_force = random.uniform(-Params.MAX_WIND_FORCE, Params.MAX_WIND_FORCE)
        self.steps = 0
        
        self.threats_spawned = 0
        self.threats_destroyed = 0
        self.threats_missed = 0
        self.threats_hit_agent = 0
        
        # Configure Health based on Phase
        if self.curriculum_phase == 1:
            self.agent_health = Params.AGENT_HEALTH # Immortal logic handled in step()
        elif self.curriculum_phase == 2:
            self.agent_health = Params.PHASE2_HEALTH
        else:
            self.agent_health = Params.AGENT_HEALTH
            
        self.shots_fired = 0
        self.spawn_timer = 100 # Instant spawn logic
        self.hit_streak = 0
        
        # Anti-Jitter: Reset previous angle to current (fresh start)
        self.prev_angle = self.defense_system.angle
        
        return self._get_obs(), {}
    
    def _calculate_ideal_angle(self, target):
        """Calculate the angle required to hit the target - SIMPLIFIED"""
        # SIMPLE: Just point directly at target's current position
        # Since projectile is very fast (300) and has no gravity, 
        # direct aim works reasonably well
        
        dx = target.x - self.defense_system.x
        dy = -(target.y - self.defense_system.y) # Positive Up (screen coords inverted)
        
        # Simple angle to target
        angle_rad = math.atan2(dy, dx)
        
        return angle_rad

    def _get_obs(self):
        # Scan for threats
        current_time = time.time()
        target = None
        min_tti = float('inf')
        
        # Simple radar: find the highest threat (closest to base/agent)
        if self.threats:
            # Sort by Y (higher Y means closer to base at bottom)
            target = max(self.threats, key=lambda t: t.y)
            dist = math.hypot(target.x - self.defense_system.x, target.y - self.defense_system.y)
            min_tti = dist / Params.PROJECTILE_SPEED
            
        can_fire = self.defense_system.cooldown_timer <= 0 and self.defense_system.ammo > 0
        
        # Fixed 15-dim observation vector
        obs = np.zeros(15, dtype=np.float32)
        
        # 0: HAS_TARGET
        obs[0] = 1.0 if target else 0.0
        # 1: CAN_FIRE
        obs[1] = 1.0 if can_fire else 0.0
        
        # ===== GLOBAL AGENT STATE =====
        # 9: Ammo Ratio (Phase Specific)
        current_max_ammo = Params.AMMO_CAPACITY
        if self.curriculum_phase == 2: current_max_ammo = Params.PHASE2_AMMO
        elif self.curriculum_phase == 3: current_max_ammo = Params.PHASE3_AMMO
        obs[9] = self.defense_system.ammo / current_max_ammo
        
        # 10: Threat Density
        obs[10] = len(self.threats) / max(1, Params.MAX_CONCURRENT_THREATS)

        # ... (rest of feature extraction remains same in logic)

        
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
            
            # --- STRICT ANTI-JITTER & LOCK-ON REWARDS ---
            
            if has_target > 0.5:
                # 1. TRACKING MODE
                # Guidanace Gradient: Reward getting closer instead of just punishing error
                # Old: reward -= raw_aim_error * 5.0 (Valley of Death)
                # New: Positive reinforcement for better aim (INCREASED 2.0 -> 5.0)
                reward += (1.0 - min(1.0, raw_aim_error)) * 5.0 
                
                # Big Carrot: Massive Bonus for locking on
                # Big Carrot: Massive Bonus for locking on (ADJUSTED: 0.05 -> 0.08)
                if raw_aim_error < 0.08: # approx 4.5 degrees (More achievable)
                    reward += 20.0       # Massive incentive
                elif raw_aim_error < 0.15: # approx 8.5 degrees (Good enough)
                    reward += 5.0        # Small encouragement
            else:
                pass # Anti-Jitter logic moved to global section below
            # Prevent firing in Phase 1
            fire_trigger = 0.0
            
        # Phase 2: ONE SHOT (Strict Ammo - Episode continues until ammo out or threats done)
        elif self.curriculum_phase == 2:
            pass # Standard Logic (Termination handled by completion check)
                
        # Phase 3: FULL GAME (Normal rewards)
        elif self.curriculum_phase == 3:
            pass # Standard logic
            
        # ------------------------
        
        # Physics Update
        self._update_physics()
        self._spawn_threats()
        self._update_wind()
        
        # --- GLOBAL ANTI-JITTER PENALTY ---
        # User Request: "Agent is vibrating in Phase 2" -> Because this was only in Phase 1!
        # Now it applies ALWAYS.
        
        # Get current observation to check for target presence
        obs_current = self._get_obs()
        has_target = obs_current[0]  # 1.0 if target exists
        
        current_angle_deg = math.degrees(self.defense_system.angle)
        prev_angle_deg = math.degrees(self.prev_angle) if hasattr(self, 'prev_angle') else current_angle_deg
        delta_angle = abs(current_angle_deg - prev_angle_deg)
        
        if has_target < 0.5: # Only punish jitter when IDLE (No target)
            if delta_angle > 1.0: # Ignore tiny micro-movements
                 reward -= delta_angle * 0.05
                 
        # Store current angle for next step jitter check
        self.prev_angle = self.defense_system.angle
        # ----------------------------------
        
        # --- GUIDANCE REWARD (Alignment) ---
        # User Request: "Encourage aiming at threat"
        # Constant small reward for keeping the crosshair on target
        # NEW: Check HAS_TARGET first, then use obs[2] for Aim Error
        raw_aim_error = obs_current[2] / 5.0  # Aim Error is now at index 2
        
        if has_target > 0.5 and abs(raw_aim_error) < 0.2:  # Only if target exists!
            alignment_bonus = (0.2 - abs(raw_aim_error)) * 5.0 # Max +1.0 per step
            reward += alignment_bonus
        # -----------------------------------
        
        # Fire Logic (PURE RL: Agent learns by mistake, not by constraint)
        # Agent CAN fire whenever it wants, but learns through REWARDS/PENALTIES
        # No aim_is_good constraint - let the reward signal teach!
        if fire_trigger > 0.0 and self.curriculum_phase != 1 and has_target > 0.5:
             if self.defense_system.fire(self.defense_system.angle, Params.PROJECTILE_SPEED):
                 self.shots_fired += 1
                 # Real Cost: Use the global configuration (-100.0)
                 reward += Rewards.FIRE_PENALTY
                 
                 # ===== PHASE 2 OPTIMIZATION: DARE TO SHOOT =====
                 # Check aim quality AT THE MOMENT of firing
                 if has_target > 0.5:
                     # Calculate error again to be precise
                     shot_aim_error = abs(obs_current[2] / 5.0) 
                     
                     if shot_aim_error < 0.1:
                         # GOOD SHOT BONUS: Reward for taking the risk!
                         # Net: +300 (Bonus) - 100 (Cost) = +200 profit just for trying
                         reward += Rewards.GOOD_SHOT_BONUS
                     elif shot_aim_error > Rewards.BAD_AIM_THRESHOLD: # > 0.3
                         # BAD SHOT PENALTY: Punish blind firing
                         reward += Rewards.BAD_AIM_PENALTY
                 # ===============================================

                 # ===== BLIND FIRE PENALTY =====
                 # If firing when NO TARGET exists, massive penalty!
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
                # DO NOT TERMINATE immediately (We have 5 threats to clear)
                # terminated = True 
                info['result'] = 'Hit' # Status update
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
                # DO NOT TERMINATE immediately (Keep going for remaining threats)
                # terminated = True
                info['result'] = 'Miss'
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

        # Win Condition (Phase 1 & 3) - MOVED UP FOR AMMO CHECK
        if self.curriculum_phase == 1:
            target_threats = Params.PHASE1_THREATS_PER_EPISODE 
        elif self.curriculum_phase == 2:
            target_threats = Params.PHASE2_THREATS # SNIPER CHALLENGE: Deal with 5 threats
        else:
            target_threats = Params.PHASE3_THREATS

        # Ammo Depletion Check (FAIL Condition)
        # If ammo is 0, no projectiles in air, but threats still exist/spawning -> FAILED
        if self.curriculum_phase > 1: # Phase 1 has infinite ammo
             # Check if we still have work to do
             threats_remain = len(self.threats) > 0 or self.threats_spawned < target_threats
             # Check if we are helpless
             is_helpless = self.defense_system.ammo <= 0 and len(self.defense_system.projectiles) == 0
             
             if is_helpless and threats_remain:
                 reward += Rewards.AMMO_DEPLETED_PENALTY # -10000
                 terminated = True
                 info['result'] = 'AmmoFail'
                 info['reason'] = 'ammo_depleted'
        
        # Terminate Condition: All threats spawned and cleared
        # We wait for: 
        # 1. Total threats spawned >= target
        # 2. No active threats on screen
        # 3. No active projectiles (important for last-moment hits)
        if self.threats_spawned >= target_threats and len(self.threats) == 0 and len(self.defense_system.projectiles) == 0:
            terminated = True
            info['reason'] = 'episode_complete'
            
            # Mission SUCCESS (All targets down)
            if self.threats_destroyed == self.threats_spawned:
                # Win Bonus for Phase 3 (Full War)
                if self.curriculum_phase == 3:
                    reward += Rewards.EPISODE_WIN_BONUS
                
                # Ammo Efficiency Bonus (Phase 2 & 3)
                # User Request: "Daha fazla puan alıyor mu?" -> YES NOW.
                if self.curriculum_phase >= 2:
                     unused_ammo = self.defense_system.ammo
                     ammo_bonus = unused_ammo * Rewards.AMMO_EFFICIENCY_BONUS
                     reward += ammo_bonus

        # Max steps
        if self.steps >= 2000:
            truncated = True

        info['hits'] = self.threats_destroyed
        info['spawned'] = self.threats_spawned
        info['ammo'] = self.defense_system.ammo  # Remaining ammo
        info['shots'] = self.shots_fired         # Shots taken this episode
        info['health'] = self.agent_health       # Current health status
    
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
        elif self.curriculum_phase == 2:
            max_threats = Params.PHASE2_THREATS
            spawn_interval = Params.SPAWN_INTERVAL
        else:
            max_threats = Params.PHASE3_THREATS
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
