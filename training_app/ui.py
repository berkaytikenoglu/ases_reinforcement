
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import sys
import subprocess

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.parameters import Params

class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASES Training Control Panel")
        self.root.geometry("500x600")
        
        self.params = {}
        self.load_params_from_class()
        
        self.create_widgets()
        
    def load_params_from_class(self):
        # Extract editable parameters
        self.params = {
            "GRAVITY": Params.GRAVITY,
            "DT": Params.DT,
            "AMMO_CAPACITY": Params.AMMO_CAPACITY,
            "RELOAD_TIME": Params.RELOAD_TIME,
            "PROJECTILE_SPEED": Params.PROJECTILE_SPEED,
            "THREAT_SPEED_MIN": Params.THREAT_SPEED_MIN,
            "THREAT_SPEED_MAX": Params.THREAT_SPEED_MAX,
            "SPAWN_RATE": Params.SPAWN_RATE,
            "MAX_WIND_FORCE": Params.MAX_WIND_FORCE,
            "LEARNING_RATE": Params.LEARNING_RATE
        }
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Simulation Parameters", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.entries = {}
        
        # Create input fields
        param_frame = ttk.Frame(main_frame)
        param_frame.pack(fill=tk.BOTH, expand=True)
        
        row = 0
        for key, value in self.params.items():
            ttk.Label(param_frame, text=key).grid(row=row, column=0, sticky=tk.W, pady=5, padx=5)
            entry = ttk.Entry(param_frame)
            entry.insert(0, str(value))
            entry.grid(row=row, column=1, sticky=tk.EW, pady=5, padx=5)
            self.entries[key] = entry
            row += 1
            
        # Buttons & Checkbox
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        
        self.use_3d_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_frame, text="Use 3D Visualization", variable=self.use_3d_var).pack(side=tk.TOP, pady=5)
        
        ttk.Button(btn_frame, text="Save Parameters", command=self.save_parameters).pack(side=tk.LEFT, padx=5, expand=True)
        ttk.Button(btn_frame, text="Start Training", command=self.start_training).pack(side=tk.LEFT, padx=5, expand=True)
        
    def save_parameters(self):
        # Read values from entries and update config/parameters.py
        # This is a bit "hacky" text replacement or we could use AST, 
        # but for simplicity let's regenerate the file content based on the inputs + default structure.
        # Or simpler: Just write the file completely since we know the structure.
        
        try:
            new_params = {}
            for key, entry in self.entries.items():
                val = entry.get()
                # Try to cast to float or int
                try:
                    if "." in val:
                        new_params[key] = float(val)
                    else:
                        new_params[key] = int(val)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid value for {key}")
                    return

            # Re-write parameters.py
            content = f"""
class Params:
    # Ekran ve Simülasyon Ayarları
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 60
    
    # Fizik
    GRAVITY = {new_params.get("GRAVITY", 9.8)}
    DT = {new_params.get("DT", 0.1)}
    
    # Savunma Sistemi (Ajan)
    AMMO_CAPACITY = {new_params.get("AMMO_CAPACITY", 10)}
    RELOAD_TIME = {new_params.get("RELOAD_TIME", 50)}
    PROJECTILE_SPEED = {new_params.get("PROJECTILE_SPEED", 20.0)}
    
    # Tehdit (Meteor/Hedef)
    THREAT_SPEED_MIN = {new_params.get("THREAT_SPEED_MIN", 5.0)}
    THREAT_SPEED_MAX = {new_params.get("THREAT_SPEED_MAX", 10.0)}
    SPAWN_RATE = {new_params.get("SPAWN_RATE", 100)}
    
    # Rüzgar (Stokastik)
    WIND_CHANGE_INTERVAL = 200
    MAX_WIND_FORCE = {new_params.get("MAX_WIND_FORCE", 2.0)}
    
    # Eğitim / RL
    LEARNING_RATE = {new_params.get("LEARNING_RATE", 0.0003)}
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
"""
            with open(os.path.join(os.path.dirname(__file__), '../config/parameters.py'), 'w', encoding='utf-8') as f:
                f.write(content)
                
            messagebox.showinfo("Success", "Parameters saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_training(self):
        self.save_parameters() # Save before starting
        
        # Launch trainer.py
        try:
            trainer_path = os.path.join(os.path.dirname(__file__), '../src/controller/trainer.py')
            
            cmd = [sys.executable, trainer_path]
            if self.use_3d_var.get():
                cmd.append("--3d")
            
            # Use Popen to run in background
            subprocess.Popen(cmd, cwd=os.path.dirname(trainer_path))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
