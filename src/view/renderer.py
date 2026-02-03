
import pygame
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
        
        # Draw HUD
        stats = [
            f"Ammo: {ds.ammo}/{ds.ammo_capacity}",
            f"Wind: {self.env.wind_force:.2f}",
            f"Saves: {self.env.threats_destroyed}",
            f"Misses: {self.env.threats_missed}",
            f"Threats: {self.env.threats_spawned}/{Params.THREATS_PER_EPISODE}"
        ]
        
        for i, line in enumerate(stats):
            text_surf = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surf, (10, 10 + i * 20))

        pygame.display.flip()
        self.clock.tick(Params.FPS)
        return True

    def close(self):
        pygame.quit()
