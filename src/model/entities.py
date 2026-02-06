import math
import random
import numpy as np

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
        self.radius = 10 # Increased from 3 to help learning

    def update(self, dt, wind_force, gravity=9.8):
        # Apply wind force to x velocity (minimal effect on fast projectiles)
        self.dx += wind_force * dt * 0.1
        # NO gravity for defense projectiles - they fly straight to target
        super().update(dt)


class Threat(Entity):
    def __init__(self, x, y, speed, target_x, gravity=50.0):
        # Improved Ballistic/Falling Logic
        # Goal: Hit (target_x, 600) starting from (x, y).
        # Constraint: Start moving DOWN immediately (dy > 0).
        # We set initial vertical velocity = speed (Downwards).
        
        TARGET_Y = 600
        dist_x = target_x - x
        dist_y = TARGET_Y - y # Positive (e.g., 750)
        
        # Quadratic Equation for Time t:
        # dist_y = dy_0 * t + 0.5 * g * t^2
        # 0.5*g * t^2 + speed * t - dist_y = 0
        
        a = 0.5 * gravity
        b = speed
        c = -dist_y
        
        # Discriminant
        # t = (-b + sqrt(b^2 - 4ac)) / 2a  (Only positive root matters)
        delta = b**2 - 4*a*c
        if delta < 0: delta = 0 # Should not happen if c is negative (dist_y positive)
        
        time_to_impact = (-b + math.sqrt(delta)) / (2*a)
        
        if time_to_impact <= 0: time_to_impact = 1.0 # Safety fallback
        
        # Horizontal velocity to close the X gap in exactly t seconds
        dx = dist_x / time_to_impact
        # Vertical velocity (Initial)
        dy = speed 
        
        super().__init__(x, y, dx, dy)
        self.radius = 15 # Increased from 5 to help learning

    def update(self, dt, wind_force, gravity=9.8):
        # Wind affects threats too, depending on their aerodynamics/weight
        # For simulation, let's say it affects them slightly less than projectiles if they are heavy
        self.dx += (wind_force * 0.5) * dt
        self.dy += gravity * dt # Apply gravity acceleration
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
