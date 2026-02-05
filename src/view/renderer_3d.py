
from ursina import *
from config.parameters import Params
import threading
import math

class Hemisphere(Mesh):
    def __init__(self, radius=1, segments=32, **kwargs):
        vertices = []
        triangles = []
        uvs = []
        
        # Generate vertices
        for i in range(segments // 2 + 1): # Standard sphere has 'segments' rings. We only need half (0 to 90 degrees)
            lat = math.pi * i / segments # 0 to pi/2
            y = math.cos(lat) * radius
            r = math.sin(lat) * radius
            
            for j in range(segments + 1):
                lon = 2 * math.pi * j / segments
                x = math.cos(lon) * r
                z = math.sin(lon) * r
                vertices.append((x, y, z))
                uvs.append((j/segments, i/(segments/2)))

        # Generate triangles
        for i in range(segments // 2):
            for j in range(segments):
                i1 = i * (segments + 1) + j
                i2 = i1 + 1
                i3 = (i + 1) * (segments + 1) + j
                i4 = i3 + 1
                triangles.append(i1)
                triangles.append(i2)
                triangles.append(i3)
                triangles.append(i3)
                triangles.append(i2)
                triangles.append(i4)
                
        super().__init__(vertices=vertices, triangles=triangles, uvs=uvs, **kwargs)

class Renderer3D:
    def __init__(self, env):
        # Ursina needs to run on the main thread usually, but since we are calling render() from a loop...
        # Ursina 'app.run()' blocks. So we need to set it up differently or use it in 'step' mode if possible.
        # Actually, for RL visualization, we usually update the Ursina entities in the render() call.
        # But Ursina is an engine that owns the loop.
        
        # Strategy: Initialize Ursina, create entities.
        # In render(), we call app.step() manually if possible, or we just update entity positions.
        
        # Note: Ursina instance is a singleton.
        try:
            # Check if base exists (it's injected into builtins by Ursina)
            _ = base
            self.app = base.app
            print("DEBUG: Using existing Ursina app")
        except NameError:
            print("DEBUG: Creating new Ursina app")
            self.app = Ursina(title="ASES 3D Defense Simulation", development_mode=False)
            
            window.color = color.black
            window.borderless = False
            window.exit_button.visible = False
            window.fps_counter.enabled = True
        
        # Visual Setup - ALWAYS runs
        # Sky and Atmosphere - Higher quality
        Sky(texture='sky_sunset')
        
        # Ground - Grass texture
        self.ground = Entity(model='cube', scale=(500, 5, 500), position=(0, -2.5, 0), texture='grass', color=color.green)
        
        # Dome - Light pink with transparency
        print("DEBUG: Loading Light Pink Hemisphere Dome...")
        self.dome = Entity(model=Hemisphere(radius=0.5), scale=180, position=(0, 0, 0), double_sided=True)
        self.dome.color = color.rgb(1.0, 0.8, 0.85) # Light pink
        self.dome.alpha = 0.25
        
        # Platform (Concrete Base under Cannon)
        self.platform = Entity(model='cylinder', scale=(6, 0.5, 6), position=(0, 0.25, 0), color=color.gray)
        
        # Defense System (Cannon)
        self.cannon_base = Entity(model='cube', color=color.dark_gray, scale=(2, 2, 2), position=(0, 1.5, 0))
        self.cannon_barrel = Entity(parent=self.cannon_base, model='cube', color=color.gray, scale=(0.5, 4, 0.5), position=(0, 0, 0), origin_y=-0.5)
        
        # FOV Cone (Analysis Tool - 3D)
        self.fov_cone = Entity(
            parent=self.cannon_barrel,
            model='cone',
            color=color.rgba(0, 255, 100, 40),
            scale=(math.tan(math.radians(Params.VISUAL_FOV_ANGLE/2)) * 10, 10, math.tan(math.radians(Params.VISUAL_FOV_ANGLE/2)) * 10),
            position=(0, 1, 0), # At the end of barrel
            origin_y=-0.5,
            rotation_x=0
        )
        
        # Entities Cache
        self.threat_entities = {} # Map object logic ID -> Ursina Entity
        self.projectile_entities = {}
        
        # Lighting
        PointLight(parent=camera, position=(0, 10, -10))
        AmbientLight(color=color.rgba(180, 180, 200, 100)) # Blue-ish ambient light
        
        # Instructions
        Text(text='Controls:\n[W,A,S,D] Move (Hold Shift for Speed)\n[Q,E] Up/Down\n[Right Click+Drag] Look\n[Scroll] Zoom', position=(-0.85, 0.45), origin=(-0.5, 0.5), color=color.white)
        
        # HUD Elements (Top Right)
        self.hud_ammo = Text(text='Ammo: --', position=(0.65, 0.45), origin=(0, 0.5), color=color.yellow, scale=1.5)
        self.hud_hits = Text(text='Saves: 0', position=(0.65, 0.40), origin=(0, 0.5), color=color.green, scale=1.5)
        self.hud_misses = Text(text='Misses: 0', position=(0.65, 0.35), origin=(0, 0.5), color=color.red, scale=1.5)
        self.hud_episode = Text(text='Episode: 1', position=(0.65, 0.30), origin=(0, 0.5), color=color.white, scale=1.2)
        self.hud_spawned = Text(text='Spawned: 0/50', position=(0.65, 0.25), origin=(0, 0.5), color=color.cyan, scale=1.2)
        self.hud_threats = Text(text='Active: 0', position=(0.65, 0.20), origin=(0, 0.5), color=color.orange, scale=1.2)
        self.hud_hp = Text(text='HEALTH: 3', position=(0.65, 0.15), origin=(0, 0.5), color=color.green, scale=2.0)
        
        # Stats tracking
        self.total_hits = 0
        self.total_misses = 0
        self.current_episode = 1
        
        self.env = env
        self.width_scale = 100.0 / Params.SCREEN_WIDTH 
        self.height_scale = 100.0 / Params.SCREEN_HEIGHT
        
    def _map_coords(self, x, y):
        new_x = (x - Params.SCREEN_WIDTH/2) * 0.1
        new_y = (Params.SCREEN_HEIGHT - y) * 0.1 
        new_z = 0 
        return (new_x, new_y, new_z)
    
    def _create_explosion(self, position):
        """Create a visual explosion effect at the given position"""
        # Create expanding orange sphere
        explosion = Entity(
            model='sphere',
            color=color.orange,
            scale=1,
            position=position
        )
        # Animate expansion and fade out
        explosion.animate_scale(8, duration=0.3)
        explosion.animate_color(color.rgba(255, 100, 0, 0), duration=0.3)
        # Destroy after animation
        destroy(explosion, delay=0.35)

    def render(self):
        import time
        time.sleep(0.016) 
        
        # --- Manual Camera Control ---
        base_speed = 20 * time.dt
        if held_keys['shift']:
            base_speed *= 3 # Sprint multiplier
            
        speed = base_speed
        if held_keys['w']: camera.position += camera.forward * speed
        if held_keys['s']: camera.position -= camera.forward * speed
        if held_keys['a']: camera.position -= camera.right * speed
        if held_keys['d']: camera.position += camera.right * speed
        if held_keys['q']: camera.position += camera.up * speed # Up
        if held_keys['e']: camera.position -= camera.up * speed # Down
        
        # Mouse Look (Right Click)
        if mouse.right:
            camera.rotation_y += mouse.velocity[0] * 100
            camera.rotation_x -= mouse.velocity[1] * 100
        
        # Clamp camera to stay above ground
        if camera.y < 1:
            camera.y = 1
            
        # Zoom (Scroll) uses built-in camera keys usually, but let's implement basic
        # Actually mouse.wheel is not in held_keys usually.
        # We can check held_keys['scroll up'] if Ursina maps it? Ursina uses input(key).
        # Since we are not in standard loop with input() callback, checking wheel is harder.
        # Let's trust WASD + Mouse Look is sufficient for now.
        # -----------------------------
        
        # UPDATE HUD
        ds = self.env.defense_system
        self.hud_ammo.text = f'Ammo: {ds.ammo}/{ds.ammo_capacity}'
        self.hud_threats.text = f'Active: {len(self.env.threats)}'
        
        # Dynamic target threats based on phase
        target_threats = Params.PHASE1_THREATS_PER_EPISODE if self.env.curriculum_phase == 1 else \
                         (1 if self.env.curriculum_phase == 2 else Params.THREATS_PER_EPISODE)
                         
        self.hud_spawned.text = f'Spawned: {self.env.threats_spawned}/{target_threats}'
        
        # Update HP
        hp = max(0, self.env.agent_health)
        self.hud_hp.text = f'HEALTH: {hp}'
        if hp >= 3:
            self.hud_hp.color = color.green
        elif hp == 2:
            self.hud_hp.color = color.yellow
        else:
            self.hud_hp.color = color.red
        
        # 1. Update Defense System
        # Map Cannon position (it's static usually but logic has x,y)
        cx, cy, cz = self._map_coords(ds.x, ds.y)
        self.cannon_base.position = (cx, 0.5, cz)
        
        # Rotate Cannon Barrel based on Agent's Angle
        # Agent angle: 0 (right), pi/2 (up). 
        # Ursina Z-rotation: 0 is Up, -90 is Right.
        # Conversion: Ursina_Z = - (ds.angle_degrees - 90)
        angle_deg = math.degrees(ds.angle)
        self.cannon_barrel.rotation_z = - (angle_deg - 90)
            
        if ds.cooldown_timer > 0:
            self.cannon_barrel.color = color.red
        else:
            self.cannon_barrel.color = color.green
            
        # 2. Update Threats
        # Sync logic threats with visual entities
        current_threats = self.env.threats
        
        # Remove dead ones - just clear the visuals, don't count here
        # The actual hit/miss counting is done by environment
        to_remove = []
        current_threat_ids = [id(t) for t in current_threats]
        
        for t_logic_id, t_entity in self.threat_entities.items():
            if t_logic_id not in current_threat_ids:
                explosion_pos = t_entity.position
                
                # Create explosion for visual feedback
                # We show explosion for all removed threats (hits look like explosions)
                if explosion_pos.y > 5:  # If still in air, likely a hit
                    self._create_explosion(explosion_pos)
                
                destroy(t_entity)
                to_remove.append(t_logic_id)
                
        for k in to_remove:
            del self.threat_entities[k]
        
        # Update HUD from environment stats (accurate source)
        self.hud_hits.text = f'Saves: {self.env.threats_destroyed}'
        self.hud_misses.text = f'Misses: {self.env.threats_missed}'
            
        # Add/Update living ones
        for t in current_threats:
            tid = id(t)
            tx, ty, tz = self._map_coords(t.x, t.y)
            
            if tid not in self.threat_entities:
                # Create new
                self.threat_entities[tid] = Entity(model='sphere', color=color.red, scale=2.0, position=(tx, ty, tz))
            else:
                # Update pos - Direct assignment, no animation lag
                self.threat_entities[tid].position = (tx, ty, tz)
                self.threat_entities[tid].rotation_y += 1 # Spin effect

        # 3. Update Projectiles
        current_projs = ds.projectiles
         # Remove dead ones
        to_remove_p = []
        for p_logic_id, p_entity in self.projectile_entities.items():
            if p_logic_id not in [id(p) for p in current_projs]:
                destroy(p_entity)
                to_remove_p.append(p_logic_id)
        for k in to_remove_p:
            del self.projectile_entities[k]

        for p in current_projs:
            pid = id(p)
            px, py, pz = self._map_coords(p.x, p.y)
            
            if pid not in self.projectile_entities:
                self.projectile_entities[pid] = Entity(model='sphere', color=color.yellow, scale=0.5, position=(px, py, pz))
            else:
                 self.projectile_entities[pid].position = (px, py, pz)

        # Step the Ursina engine
        self.app.step()
        
        return True # Continue loop

    def close(self):
        # self.app.quit() # Can't fully quit if we want to restart? Ursina is hard to restart in same process.
        # Ideally just clear entities.
        for e in self.threat_entities.values(): destroy(e)
        for e in self.projectile_entities.values(): destroy(e)
        pass 
    
    def set_episode(self, episode_num):
        """Update episode counter display"""
        self.current_episode = episode_num
        self.hud_episode.text = f'Episode: {episode_num}'
