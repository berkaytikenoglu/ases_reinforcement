import math
import random
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.parameters import Params

class Entity:
    def __init__(self, x, y, dx=0, dy=0):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.active = True

    def update(self, dt):
        self.x += self.dx * dt
        self.y += self.dy * dt

class Projectile(Entity):
    def __init__(self, x, y, speed, angle_rad):
        # Calculate velocity components based on angle
        dx = speed * math.cos(angle_rad)
        dy = -speed * math.sin(angle_rad) # Negative dy because y=0 is top
        super().__init__(x, y, dx, dy)
        self.radius = Params.PROJECTILE_RADIUS

    def update(self, dt, wind_force, gravity=9.8):
        # Apply wind force to x velocity (minimal effect on fast projectiles)
        self.dx += wind_force * dt * 0.1
        # NO gravity for defense projectiles - they fly straight to target
        super().update(dt)


class Threat(Entity):
    def __init__(self, x, y, speed, target_x, gravity=50.0, is_friendly=False, is_horizontal=False):
        # IFF (Identification Friend or Foe) System
        self.is_friendly = is_friendly  # True = Friend (Green), False = Enemy (Red)
        self.is_horizontal = is_horizontal  # True = Horizontal flight (UAV style)
        
        if is_horizontal:
            # Horizontal UAV: Flies at constant altitude, left-to-right or right-to-left
            dy = 0  # No vertical movement
            # Direction based on spawn position
            if x < 400:
                dx = speed  # Spawn left, fly right
            else:
                dx = -speed  # Spawn right, fly left
            super().__init__(x, y, dx, dy)
            self.radius = Params.THREAT_RADIUS
            self.gravity = 0  # No gravity for horizontal flight
        else:
            # Original ballistic/falling logic (top-down threats)
            TARGET_Y = 600
            dist_x = target_x - x
            dist_y = TARGET_Y - y
            
            a = 0.5 * gravity
            b = speed
            c = -dist_y
            
            delta = b**2 - 4*a*c
            if delta < 0: delta = 0
            
            time_to_impact = (-b + math.sqrt(delta)) / (2*a)
            
            if time_to_impact <= 0: time_to_impact = 1.0
            
            dx = dist_x / time_to_impact
            dy = speed 
            
            super().__init__(x, y, dx, dy)
            self.radius = Params.THREAT_RADIUS
            self.gravity = gravity

    def update(self, dt, wind_force, gravity=9.8):
        if self.is_horizontal:
            # Horizontal flight: No gravity, minimal wind effect
            self.dx += (wind_force * 0.1) * dt  # Very slight wind influence
            # No dy change (constant altitude)
        else:
            # Original ballistic logic
            self.dx += (wind_force * 0.5) * dt
            self.dy += self.gravity * dt
        super().update(dt)

class DefenseSystem:
    def __init__(self, x, y, cooldown_max, ammo_capacity=10):
        self.x = x
        self.y = y
        self.cooldown_max = cooldown_max
        self.cooldown_timer = 0
        self.ammo_capacity = ammo_capacity
        self.ammo = ammo_capacity 
        self.projectiles = []
        self.angle = math.pi / 2 # Default: Pointing Up (90 degrees)

    def update(self, dt):
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1

    def fire(self, angle_rad, speed):
        if self.cooldown_timer <= 0 and self.ammo > 0:
            proj = Projectile(self.x, self.y, speed, angle_rad)
            self.projectiles.append(proj)
            self.cooldown_timer = self.cooldown_max
            self.ammo -= 1
            return True
        return False
    
    def reload(self):
        self.ammo = self.ammo_capacity
