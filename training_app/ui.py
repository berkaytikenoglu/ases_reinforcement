
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import json
import os
import sys
import subprocess
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.parameters import Params

class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASES Training Control Panel")
        self.root.geometry("750x850")
        
        self.is_training = False
        self.current_process = None # Track subprocess
        self.PARAMS_DESC = {
            "GRAVITY": "Yerçekimi (m/s²)",
            "DT": "Zaman Adımı (s)",
            "AMMO_CAPACITY": "Mermi Kapasitesi",
            "RELOAD_TIME": "Yenileme (frame)",
            "PROJECTILE_SPEED": "Mermi Hızı",
            "THREAT_SPEED_MIN": "Min Tehdit Hızı",
            "THREAT_SPEED_MAX": "Max Tehdit Hızı",
            "SPAWN_INTERVAL": "Tehdit Sıklığı (düşük=sık)",
            "MAX_WIND_FORCE": "Max Rüzgar Gücü",
            "LEARNING_RATE": "Öğrenme Hızı (AI)",
            "THREATS_PER_EPISODE": "Görev Hedefi (Max Tehdit)",
            "MAX_CONCURRENT_THREATS": "Max Anlık Tehdit",
            "MAX_EPISODES": "Eğitim Süresi (Ep)",
            "AGENT_HEALTH": "Ajan Can Hakkı"
        }
        
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
            "SPAWN_INTERVAL": Params.SPAWN_INTERVAL,
            "MAX_WIND_FORCE": Params.MAX_WIND_FORCE,
            "LEARNING_RATE": Params.LEARNING_RATE,
            "LEARNING_RATE": Params.LEARNING_RATE,
            "MAX_EPISODES": getattr(Params, 'MAX_EPISODES', 1000),
            "AGENT_HEALTH": getattr(Params, 'AGENT_HEALTH', 3),
            
            # PPO Batch
            "ROLLOUT_STEPS": getattr(Params, 'ROLLOUT_STEPS', 2048),
            "MINIBATCH_SIZE": getattr(Params, 'MINIBATCH_SIZE', 64),
            "PPO_EPOCHS": getattr(Params, 'PPO_EPOCHS', 10)
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
        row = 0
        for key, value in self.params.items():
            ttk.Label(param_frame, text=key).grid(row=row, column=0, sticky=tk.W, pady=5, padx=5)
            
            if key == "MAX_EPISODES":
                # Use Combobox for Episodes
                entry = ttk.Combobox(param_frame, values=["10", "50", "100", "500", "1000", "2000", "5000", "10000"])
                entry.set(str(value))
            else:
                # Standard Entry
                entry = ttk.Entry(param_frame)
                entry.insert(0, str(value))
                
            entry.grid(row=row, column=1, sticky=tk.EW, pady=5, padx=5)
            
            # Description Label
            desc = self.PARAMS_DESC.get(key, "")
            ttk.Label(param_frame, text=desc, foreground="gray", font=("Arial", 8, "italic")).grid(row=row, column=2, sticky=tk.W, pady=5, padx=5)
            
            self.entries[key] = entry
            row += 1
            
        # Buttons & Checkbox
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        
        # Options row
        options_frame = ttk.Frame(btn_frame)
        options_frame.pack(side=tk.TOP, pady=5)
        
        self.use_3d_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Use 3D Visualization", variable=self.use_3d_var).pack(side=tk.LEFT, padx=10)
        
        # CPU/GPU Selection
        ttk.Label(options_frame, text="Device:").pack(side=tk.LEFT, padx=5)
        self.device_var = tk.StringVar(value="CPU")
        ttk.Radiobutton(options_frame, text="CPU", variable=self.device_var, value="CPU").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(options_frame, text="GPU", variable=self.device_var, value="GPU").pack(side=tk.LEFT, padx=2)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5, expand=True)

        # --- Model Management ---
        model_frame = ttk.LabelFrame(main_frame, text="Model Management", padding="5")
        model_frame.pack(fill=tk.X, pady=10)
        
        # Model Selection
        select_frame = ttk.Frame(model_frame)
        select_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(select_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(select_frame, textvariable=self.model_var, state="readonly", width=40)
        self.model_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(select_frame, text="↻", width=3, command=self.refresh_models).pack(side=tk.LEFT, padx=2)
        self.refresh_models() # Initial load
        
        # Action Buttons
        action_frame = ttk.Frame(model_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        # Test Visualization Mode
        self.test_viz_mode = tk.StringVar(value="3D")
        
        viz_frame = ttk.Frame(action_frame)
        viz_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(viz_frame, text="View:").pack(side=tk.LEFT)
        ttk.Radiobutton(viz_frame, text="2D", variable=self.test_viz_mode, value="2D").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(viz_frame, text="3D", variable=self.test_viz_mode, value="3D").pack(side=tk.LEFT, padx=2)
        
        # Phase Selection (User Request)
        phase_frame = ttk.Frame(action_frame)
        phase_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(phase_frame, text="Phase:").pack(side=tk.LEFT)
        self.phase_var = tk.StringVar(value="Auto")
        phase_combo = ttk.Combobox(phase_frame, textvariable=self.phase_var, values=["Auto", "Phase 1", "Phase 2", "Phase 3"], width=10, state="readonly")
        phase_combo.pack(side=tk.LEFT, padx=2)
        
        self.test_btn = ttk.Button(action_frame, text="Test Agent", command=self.test_agent)
        self.test_btn.pack(side=tk.LEFT, padx=5, expand=True)
        # ----------------------
        

        
        # Console Output Area
        log_frame = ttk.LabelFrame(main_frame, text="Training Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_area = ScrolledText(log_frame, state='disabled', height=15)
        self.log_area.pack(fill=tk.BOTH, expand=True)
        self.log_area.configure(font='TkFixedFont')

    def log(self, message):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

        
    def save_parameters(self, show_message=True):
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
    AGENT_HEALTH = {new_params.get("AGENT_HEALTH", 3)}
    RELOAD_TIME = {new_params.get("RELOAD_TIME", 50)}
    PROJECTILE_SPEED = {new_params.get("PROJECTILE_SPEED", 20.0)}
    
    # Tehdit (Meteor/Hedef)
    THREAT_SPEED_MIN = {new_params.get("THREAT_SPEED_MIN", 5.0)}
    THREAT_SPEED_MAX = {new_params.get("THREAT_SPEED_MAX", 10.0)}
    
    # Episode Kuralları
    THREATS_PER_EPISODE = {new_params.get("THREATS_PER_EPISODE", 30)}      # Her episode'da toplam tehdit sayısı
    MAX_CONCURRENT_THREATS = 3    # Aynı anda maksimum tehdit sayısı
    SPAWN_INTERVAL = {new_params.get("SPAWN_INTERVAL", 30)}
    
    # Rüzgar (Stokastik)
    WIND_CHANGE_INTERVAL = 200
    MAX_WIND_FORCE = {new_params.get("MAX_WIND_FORCE", 2.0)}
    
    # Eğitim / RL (PPO Hyperparameters)
    LEARNING_RATE = {new_params.get("LEARNING_RATE", 0.0003)}
    MAX_EPISODES = {new_params.get("MAX_EPISODES", 1000)}
    GAMMA = 0.99
    
    # Advanced PPO
    LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    
    ADVANTAGE_NORMALIZE = True
    LR_SCHEDULE = "linear"
    
    # BATCH & ROLLOUT
    ROLLOUT_STEPS = {new_params.get("ROLLOUT_STEPS", 2048)}
    MINIBATCH_SIZE = {new_params.get("MINIBATCH_SIZE", 64)}
    PPO_EPOCHS = {new_params.get("PPO_EPOCHS", 10)}
"""
            with open(os.path.join(os.path.dirname(__file__), '../config/parameters.py'), 'w', encoding='utf-8') as f:
                f.write(content)
                
            if show_message:
                messagebox.showinfo("Success", "Parameters saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))


    def refresh_models(self):
        models_dir = os.path.join(os.path.dirname(__file__), '../models')
        items = []
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                path = os.path.join(models_dir, f)
                if os.path.isdir(path):
                    # Folder based model
                    items.append({'name': f, 'mtime': os.path.getmtime(path)})
                elif f.endswith('.pth'):
                    # Legacy file based model
                    items.append({'name': f.replace('.pth', ''), 'mtime': os.path.getmtime(path)})
            
            # Sort by mtime descending
            items.sort(key=lambda x: x['mtime'], reverse=True)
            models = [item['name'] for item in items]
            
        self.model_combo['values'] = models
        if models:
            self.model_combo.current(0)
            
    def test_agent(self):
        model = self.model_var.get()
        if not model:
            messagebox.showwarning("Warning", "Please select a model first.")
            return
        self.start_training(mode="test", model_name=model)

    def stop_training(self):
        if self.current_process:
            self.log("Stopping training...")
            self.current_process.terminate()
            # self.current_process = None # Will be handled in finally block of run_process
        return

    def start_training(self, mode="train", model_name=None):
        if self.is_training:
            # If already training, this button acts as STOP
            self.stop_training()
            return
            
        if mode == "train":
            self.save_parameters(show_message=False) # Save before training
            if not model_name:
                model_name = self.model_var.get().strip()
        
        # UI Updates
        self.is_training = True
        
        # Change Start Button to Stop Button
        self.start_btn.config(text="Stop Training", command=self.stop_training)
        
        # Disable other buttons
        self.test_btn.config(state='disabled')
        
        # Launch trainer.py
        try:
            trainer_path = os.path.join(os.path.dirname(__file__), '../src/controller/trainer.py')
            
            cmd = [sys.executable, trainer_path]
            
            if mode == "test":
                cmd.append("--test")
                # Use local option from Model Management
                if self.test_viz_mode.get() == "3D":
                     cmd.append("--viz3d")
                
                # Phase Override
                phase_sel = self.phase_var.get()
                if "Phase 1" in phase_sel: cmd.extend(["--phase", "1"])
                elif "Phase 2" in phase_sel: cmd.extend(["--phase", "2"])
                elif "Phase 3" in phase_sel: cmd.extend(["--phase", "3"])
            else:
                # Training mode - use global option
                if self.use_3d_var.get():
                    cmd.append("--viz3d")
                else:
                    cmd.append("--fast")
            
            # CPU/GPU Selection
            if self.device_var.get() == "CPU":
                cmd.append("--cpu")
                    
            # Add Episode Count
            try:
                episodes = int(self.entries["MAX_EPISODES"].get())
                cmd.extend(["--episodes", str(episodes)])
            except:
                pass # Use default if invalid
            
            if not model_name:
            # Auto-generate COOL name with ordering
                cool_names = ["Titan", "Viper", "Shadow", "Hunter", "Eagle", "Raptor", "Ghost", "Spectre", "Predator", "Cobra", "Falcon", "Raven", "Onyx", "Pulse", "Storm", "Blitz", "Nova", "Apex"]
                
                # Find next order ID
                next_id = 1
                models_dir = os.path.join(os.path.dirname(__file__), '../models')
                if os.path.exists(models_dir):
                    for f in os.listdir(models_dir):
                        if "_" in f and f.split("_")[0].isdigit():
                            try:
                                fid = int(f.split("_")[0])
                                if fid >= next_id:
                                    next_id = fid + 1
                            except:
                                pass
                
                chosen_name = random.choice(cool_names)
                model_name = f"{next_id:03d}_{chosen_name}"
                self.log(f"Auto-generated Agent Name: {model_name}")

            if model_name:
                cmd.extend(["--model", model_name])

            
            try:
                threading.Thread(target=self.run_process, args=(cmd,), daemon=True).start()
            except Exception as e:
                self.is_training = False
                self.start_btn.config(state='normal', text="Start Training", command=self.start_training) # Reset if launch fails
                raise e
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")

    def run_process(self, cmd):
        self.log("-" * 50)
        self.log(f"Starting process: {' '.join(cmd)}")
        self.log("-" * 50)
        
        try:
            # Determine Project Root properly
            # ui.py is in training_app/, so root is one level up
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            
            # Popen with pipe redirection
            self.current_process = subprocess.Popen(
                cmd, 
                cwd=project_root, # Use project root as CWD (Fixes asset loading issues)
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            ) 
            
            # Read output line by line
            for line in self.current_process.stdout:
                # Schedule log update in main thread
                self.root.after(0, self.log, line.strip())
                
            self.current_process.wait()
            self.root.after(0, self.log, f"\nProcess finished with exit code {self.current_process.returncode}")
            
        except Exception as e:
            self.root.after(0, self.log, f"Error: {str(e)}")
            
        finally:
             self.current_process = None
             self.root.after(0, self.reset_buttons)
             self.is_training = False

    def reset_buttons(self):
        self.start_btn.config(state='normal', text="Start Training", command=self.start_training)
        self.test_btn.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
