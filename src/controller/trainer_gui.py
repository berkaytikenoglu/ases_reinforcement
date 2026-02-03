
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import subprocess
import sys
import os
import io
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)

class ConsoleRedirector:
    """Redirects stdout/stderr to a tkinter Text widget"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = io.StringIO()
        
    def write(self, text):
        self.buffer.write(text)
        self.text_widget.after(0, self._update_widget, text)
        
    def _update_widget(self, text):
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.config(state='disabled')
        
    def flush(self):
        pass

class TrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🛡️ ASES Defense - Eğitim Kontrol Paneli")
        self.root.geometry("800x700")
        self.root.configure(bg='#1e1e1e')
        
        self.training_thread = None
        self.is_training = False
        
        # Models directory
        self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
        os.makedirs(self.models_dir, exist_ok=True)
        
        self._create_styles()
        self._create_widgets()
        self._redirect_console()
        self._refresh_models()
        
        print("🛡️ ASES Defense Trainer GUI Başlatıldı!")
        print(f"📁 Model Konumu: {self.models_dir}")
        print("-" * 50)
        
    def _create_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Dark theme colors
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='#ffffff', font=('Segoe UI', 10))
        style.configure('Header.TLabel', background='#1e1e1e', foreground='#4fc3f7', font=('Segoe UI', 14, 'bold'))
        style.configure('TButton', font=('Segoe UI', 10), padding=10)
        style.configure('Train.TButton', background='#4caf50', foreground='white')
        style.configure('Test.TButton', background='#2196f3', foreground='white')
        style.configure('Stop.TButton', background='#f44336', foreground='white')
        
    def _create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Label(main_frame, text="🛡️ ASES Defence - Eğitim Kontrol Paneli", style='Header.TLabel')
        header.pack(pady=(0, 20))
        
        # Control Panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Left controls
        left_frame = ttk.Frame(control_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Model Selection
        model_frame = ttk.Frame(left_frame)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="📦 Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="latest")
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, width=25)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="🔄", width=3, command=self._refresh_models).pack(side=tk.LEFT)
        
        # Episode Count
        episode_frame = ttk.Frame(left_frame)
        episode_frame.pack(fill=tk.X, pady=5)
        ttk.Label(episode_frame, text="🔢 Episode Sayısı:").pack(side=tk.LEFT, padx=5)
        self.episode_var = tk.StringVar(value="500")
        self.episode_entry = ttk.Entry(episode_frame, textvariable=self.episode_var, width=10)
        self.episode_entry.pack(side=tk.LEFT, padx=5)
        
        # New Model Name
        name_frame = ttk.Frame(left_frame)
        name_frame.pack(fill=tk.X, pady=5)
        ttk.Label(name_frame, text="📝 Yeni Model Adı:").pack(side=tk.LEFT, padx=5)
        self.new_model_var = tk.StringVar(value="")
        self.new_model_entry = ttk.Entry(name_frame, textvariable=self.new_model_var, width=25)
        self.new_model_entry.pack(side=tk.LEFT, padx=5)
        
        # Checkboxes
        check_frame = ttk.Frame(left_frame)
        check_frame.pack(fill=tk.X, pady=5)
        self.viz_3d_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(check_frame, text="🎮 3D Görselleştirme", variable=self.viz_3d_var).pack(side=tk.LEFT, padx=10)
        self.fast_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(check_frame, text="⚡ Hızlı Eğitim (Görselleştirmesiz)", variable=self.fast_var).pack(side=tk.LEFT, padx=10)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        self.train_btn = tk.Button(button_frame, text="🚀 EĞİTİME BAŞLA", font=('Segoe UI', 12, 'bold'),
                                   bg='#4caf50', fg='white', width=20, height=2, command=self._start_training)
        self.train_btn.pack(side=tk.LEFT, padx=10)
        
        self.test_btn = tk.Button(button_frame, text="🧪 TEST ET", font=('Segoe UI', 12, 'bold'),
                                  bg='#2196f3', fg='white', width=15, height=2, command=self._start_testing)
        self.test_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = tk.Button(button_frame, text="⛔ DURDUR", font=('Segoe UI', 12, 'bold'),
                                  bg='#f44336', fg='white', width=15, height=2, command=self._stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        self.viz_btn = tk.Button(button_frame, text="🎬 3D GÖRÜNTÜLE", font=('Segoe UI', 12, 'bold'),
                                 bg='#9c27b0', fg='white', width=15, height=2, command=self._start_visualization)
        self.viz_btn.pack(side=tk.LEFT, padx=10)
        
        # Console Output
        console_frame = ttk.Frame(main_frame)
        console_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(console_frame, text="📋 Konsol Çıktısı:", style='Header.TLabel').pack(anchor=tk.W)
        
        self.console = scrolledtext.ScrolledText(console_frame, height=20, bg='#0d0d0d', fg='#00ff00',
                                                  font=('Consolas', 10), state='disabled')
        self.console.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Clear console button
        ttk.Button(console_frame, text="🗑️ Konsolu Temizle", command=self._clear_console).pack(anchor=tk.E)
        
        # Status bar
        self.status_var = tk.StringVar(value="Hazır")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, foreground='#888888')
        status_bar.pack(fill=tk.X, pady=5)
        
    def _redirect_console(self):
        self.console_redirector = ConsoleRedirector(self.console)
        sys.stdout = self.console_redirector
        sys.stderr = self.console_redirector
        
    def _refresh_models(self):
        models = ['latest']
        if os.path.exists(self.models_dir):
            for f in os.listdir(self.models_dir):
                if f.endswith('.pth'):
                    models.append(f.replace('.pth', ''))
        self.model_combo['values'] = models
        print(f"🔄 Modeller yenilendi. {len(models)-1} model bulundu.")
        
    def _clear_console(self):
        self.console.config(state='normal')
        self.console.delete(1.0, tk.END)
        self.console.config(state='disabled')
        
    def _start_training(self):
        if self.is_training:
            messagebox.showwarning("Uyarı", "Zaten bir eğitim devam ediyor!")
            return
            
        episodes = int(self.episode_var.get())
        model_name = self.new_model_var.get().strip() or self.model_var.get()
        render_3d = self.viz_3d_var.get()
        fast_mode = self.fast_var.get()
        
        self._set_training_state(True)
        
        def train_thread():
            try:
                from src.controller.trainer import train
                train(
                    render=not fast_mode,
                    render_3d=render_3d,
                    max_episodes=episodes,
                    model_name=model_name,
                    test_mode=False
                )
            except Exception as e:
                print(f"❌ Hata: {e}")
            finally:
                self.root.after(0, lambda: self._set_training_state(False))
                self.root.after(0, self._refresh_models)
                
        self.training_thread = threading.Thread(target=train_thread, daemon=True)
        self.training_thread.start()
        
    def _start_testing(self):
        if self.is_training:
            messagebox.showwarning("Uyarı", "Zaten bir işlem devam ediyor!")
            return
            
        model_name = self.model_var.get()
        episodes = int(self.episode_var.get())
        render_3d = self.viz_3d_var.get()
        
        self._set_training_state(True)
        
        def test_thread():
            try:
                from src.controller.trainer import train
                train(
                    render=True,
                    render_3d=render_3d,
                    max_episodes=episodes,
                    model_name=model_name,
                    test_mode=True
                )
            except Exception as e:
                print(f"❌ Hata: {e}")
            finally:
                self.root.after(0, lambda: self._set_training_state(False))
                
        self.training_thread = threading.Thread(target=test_thread, daemon=True)
        self.training_thread.start()
        
    def _start_visualization(self):
        """Start 3D visualization in a separate process"""
        model_name = self.model_var.get()
        episodes = self.episode_var.get()
        
        print(f"🎬 3D Görselleştirme başlatılıyor (ayrı pencerede)...")
        print(f"   Model: {model_name}")
        print(f"   Episodes: {episodes}")
        
        # Build command
        trainer_path = os.path.join(PROJECT_ROOT, 'src', 'controller', 'trainer.py')
        cmd = [
            sys.executable,
            trainer_path,
            '--3d',
            '--test',
            '--episodes', str(episodes),
            '--model', model_name
        ]
        
        # Run in subprocess (non-blocking)
        try:
            subprocess.Popen(cmd, cwd=PROJECT_ROOT)
            print("✅ 3D penceresi ayrı bir işlemde başlatıldı.")
            print("   (Bu pencereyi kapatmak 3D görünümü etkilemez)")
        except Exception as e:
            print(f"❌ Hata: {e}")
        
    def _stop_training(self):
        # Note: This is a simple implementation. True interruption requires more complex handling.
        print("⚠️ Durdurma isteği gönderildi. Mevcut episode tamamlanınca duracak.")
        # In a real implementation, you'd set a flag that the training loop checks
        
    def _set_training_state(self, is_training):
        self.is_training = is_training
        state = tk.DISABLED if is_training else tk.NORMAL
        self.train_btn.config(state=state)
        self.test_btn.config(state=state)
        self.viz_btn.config(state=state)
        self.stop_btn.config(state=tk.NORMAL if is_training else tk.DISABLED)
        self.status_var.set("Eğitim devam ediyor..." if is_training else "Hazır")

def main():
    root = tk.Tk()
    app = TrainerGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
