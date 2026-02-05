
import pygame
import math
from config.parameters import Params

class Renderer:
    def __init__(self, env):
        pygame.init()
        self.width = Params.SCREEN_WIDTH
        self.height = Params.SCREEN_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("ASES Reinforcement Defense Sytem")
        self.font = pygame.font.SysFont("Arial", 18)
        self.env = env
        self.clock = pygame.time.Clock()

    def render(self):
        # Handle events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill((30, 30, 50)) # Dark Blue background

        # Draw Defense System
        ds = self.env.defense_system
        
        # --- Draw FOV Cone (Analysis Tool) ---
        fov_angle_deg = Params.VISUAL_FOV_ANGLE
        current_angle = ds.angle # Radians
        
        # Calculate cone points
        cone_len = 400 # Visual length increased
        p1 = (int(ds.x), int(ds.y))
        
        # Left and Right edges of FOV
        left_angle = current_angle - math.radians(fov_angle_deg/2)
        right_angle = current_angle + math.radians(fov_angle_deg/2)
        
        p2 = (int(ds.x + math.cos(left_angle) * cone_len), 
              int(ds.y - math.sin(left_angle) * cone_len))
        p3 = (int(ds.x + math.cos(right_angle) * cone_len), 
              int(ds.y - math.sin(right_angle) * cone_len))
        
        # Draw transparent FOV surface
        fov_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # --- Target Detection for FOV Color ---
        is_locked = False
        for t in self.env.threats:
            # Calculate angle to threat
            dx = t.x - ds.x
            dy = -(t.y - ds.y)
            target_angle = math.atan2(dy, dx)
            # Normalize diff
            diff = (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < math.radians(fov_angle_deg/2):
                is_locked = True
                break
        
        if is_locked:
            # Target inside FOV: High-alert Green
            fov_color = (0, 255, 100, 120) 
            edge_color = (0, 255, 50, 200)
        else:
            # Idle: Scanning Pink (User Request)
            fov_color = (255, 100, 200, 60) 
            edge_color = (255, 50, 150, 150)
            
        pygame.draw.polygon(fov_surface, fov_color, [p1, p2, p3])
        pygame.draw.line(fov_surface, edge_color, p1, p2, 1)
        pygame.draw.line(fov_surface, edge_color, p1, p3, 1)
        self.screen.blit(fov_surface, (0,0))
        
        # Draw turret direction line
        end_x = int(ds.x + math.cos(current_angle) * 30)
        end_y = int(ds.y - math.sin(current_angle) * 30)
        pygame.draw.line(self.screen, (255, 255, 255), (ds.x, ds.y), (end_x, end_y), 3)
        
        pygame.draw.circle(self.screen, (0, 200, 0), (int(ds.x), int(ds.y)), 15)
        
        # Visualize Cooldown (Ring around defense system)
        if ds.cooldown_timer > 0:
            ratio = ds.cooldown_timer / Params.RELOAD_TIME
            pygame.draw.arc(self.screen, (255, 0, 0), 
                            (ds.x - 20, ds.y - 20, 40, 40), 0, ratio * 6.28, 2)

        # Draw Projectiles
        for p in ds.projectiles:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(p.x), int(p.y)), p.radius)

        # Draw Threats
        for t in self.env.threats:
            pygame.draw.circle(self.screen, (255, 50, 50), (int(t.x), int(t.y)), t.radius)

        # Draw Wind Indicator
        wind_start_x = self.width - 100
        wind_start_y = 50
        pygame.draw.line(self.screen, (200, 200, 255), (wind_start_x, wind_start_y), 
                         (wind_start_x + self.env.wind_force * 20, wind_start_y), 3)
        pygame.draw.circle(self.screen, (200, 200, 255), (wind_start_x, wind_start_y), 2)
        
        # Dynamic target threats based on phase
        target_threats = Params.PHASE1_THREATS_PER_EPISODE if self.env.curriculum_phase == 1 else \
                         (1 if self.env.curriculum_phase == 2 else Params.THREATS_PER_EPISODE)
                         
        # Draw HUD
        stats = [
            f"Ammo: {ds.ammo}/{ds.ammo_capacity}",
            f"Wind: {self.env.wind_force:.2f}",
            f"Saves: {self.env.threats_destroyed}",
            f"Misses: {self.env.threats_missed}",
            f"Threats: {self.env.threats_spawned}/{target_threats}"
        ]
        
        for i, line in enumerate(stats):
            text_surf = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surf, (10, 10 + i * 20))

        pygame.display.flip()
        self.clock.tick(Params.FPS)
        return True

    def close(self):
        pygame.quit()
