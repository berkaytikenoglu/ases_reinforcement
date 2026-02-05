
class Params:
    # Ekran ve Simülasyon Ayarları
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 60
    VISUAL_FOV_ANGLE = 30        # Görsel analiz için görüş açısı (Derece)
    
    # Fizik
    GRAVITY = 5.0                # BOOTCAMP: Yerçekimi düşürüldü (Daha düz atış)
    DT = 0.1
    
    # Savunma Sistemi (Ajan)
    AMMO_CAPACITY = 60
    AGENT_HEALTH = 3             # USER REQUEST: Reduced from 5 to 3 for higher stakes
    RELOAD_TIME = 5
    PROJECTILE_SPEED = 300.0     # BOOTCAMP: Daha hızlı mermi (Kolay vuruş)
    
    # Tehdit (Meteor/Hedef)
    THREAT_SPEED_MIN = 20.0      # Normal: 20
    THREAT_SPEED_MAX = 30.0      # Normal: 30
    
    # Phase 1 (Curriculum): Slower threats for easier tracking
    PHASE1_THREAT_SPEED_MIN = 10.0   # SLOW: Easy to track
    PHASE1_THREAT_SPEED_MAX = 15.0   # SLOW: Easy to track
    PHASE1_THREATS_PER_EPISODE = 3   # ULTRA FAST PHASE 1: 3 threats = even faster episodes
    PHASE1_SPAWN_INTERVAL = 10       # FAST PHASE 1: 2x spawn rate
    
    # Episode Kuralları
    THREATS_PER_EPISODE = 10     # Normal: 10 tehdit/episode
    MAX_CONCURRENT_THREATS = 1   # Teke tek
    SPAWN_INTERVAL = 20          # Normal: 20 frame aralık
    
    # Rüzgar (Stokastik)
    WIND_CHANGE_INTERVAL = 200
    MAX_WIND_FORCE = 0.5
    
    # Eğitim / RL (PPO Hyperparameters)
    LEARNING_RATE = 0.0003
    MAX_EPISODES = 1000           # FAST TRAINING: 1500 -> 1000 (Zaman kaybı yok)
    GAMMA = 0.99
    
    # Advanced PPO
    LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    
    ADVANTAGE_NORMALIZE = True
    LR_SCHEDULE = "linear"
    
    # BATCH & ROLLOUT
    ROLLOUT_STEPS = 2048
    MINIBATCH_SIZE = 64
    PPO_EPOCHS = 10
