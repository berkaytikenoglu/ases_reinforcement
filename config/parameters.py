
class Params:
    # Ekran ve Simülasyon Ayarları
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 800
    FPS = 60
    VISUAL_FOV_ANGLE = 120       # GENİŞ AÇI: 30 -> 120 (Uzaktaki hedefleri erkenden fark etmesi için)
    
    # Fizik
    GRAVITY = 5.0                # BOOTCAMP: Yerçekimi düşürüldü (Daha düz atış)
    DT = 0.1
    
    # Savunma Sistemi (Ajan)
    AMMO_CAPACITY = 60
    AGENT_HEALTH = 3             # USER REQUEST: Reduced from 5 to 3 for higher stakes
    RELOAD_TIME = 5
    PROJECTILE_SPEED = 500.0     # HIZLI VE ÖFKELİ: 300 -> 500 (Neredeyse laser gun!)
    PROJECTILE_RADIUS = 40       
    THREAT_RADIUS = 50           
    
    # Tehdit (Meteor/Hedef)
    THREAT_SPEED_MIN = 20.0    
    THREAT_SPEED_MAX = 30.0      
    
    # Phase 1 (Curriculum): Same speed as Phase 2 for consistent physics
    PHASE1_THREAT_SPEED_MIN = 20.0   # Faz 2 ile aynı
    PHASE1_THREAT_SPEED_MAX = 30.0   # Faz 2 ile aynı
    PHASE1_THREATS_PER_EPISODE = 2   # USER REQUEST: 2 threats for Phase 1
    PHASE1_SPAWN_INTERVAL = 60       # SLOW SPAWN (1 sec): More time to stabilize (Anti-Jitter Training)
    # Phase 1 (Curriculum): Aim Mastery
    PHASE1_EPISODES = 2500           # Standart: 2500 Episode
    
    # Phase 2 (Curriculum): One Shot Mode
    PHASE2_EPISODES = 2500           # Standart: 2500 Episode (Total: 5000)
    PHASE2_THREATS = 5               # MULTI-TARGET: 1 -> 5 (Daha fazla pratik şansı)
    PHASE2_AMMO = 20                 # BUFFER: 5 -> 20 (Target başı 4 mermi - Bol bol dene)
    PHASE2_HEALTH = 1                # 1 tehdit = 1 can (kaçırırsan ölürsün)
    PHASE2_RELOAD_TIME = 15          # HIZLI RELOAD: 30 -> 15 (İkinci şans için zaman)
    PHASE2_THREAT_SPEED_MIN = 20.0   
    PHASE2_THREAT_SPEED_MAX = 35.0   

    # Phase 3 (Full Game)
    PHASE3_THREATS = 10              # USER REQUEST: 10 Threads
    PHASE3_AMMO = 30                 # USER REQUEST: 30 Ammo (3 per threat - More forgiving)
    
    # Episode Kuralları
    THREATS_PER_EPISODE = 10     # Normal: 10 tehdit/episode
    MAX_CONCURRENT_THREATS = 3   # Teke tek
    SPAWN_INTERVAL = 60          # Normal: 60 frame aralık (User Request: More waiting)
    
    # Rüzgar (Stokastik)
    WIND_CHANGE_INTERVAL = 200
    MAX_WIND_FORCE = 0.5
    
    # Eğitim / RL (PPO Hyperparameters)
    LEARNING_RATE = 0.0005           # AGGRESIF: 0.0003 -> 0.0005
    MAX_EPISODES = 1000           # FAST TRAINING: 1500 -> 1000 (Zaman kaybı yok)
    GAMMA = 0.99
    
    # Advanced PPO
    LAMBDA = 0.95
    CLIP_RANGE = 0.3              # AGGRESSIVE UPDATES: 0.2 -> 0.3
    ENTROPY_COEF = 0.005          # LESS RANDOMNESS: 0.01 -> 0.005 (Focus on what works)
    VALUE_COEF = 0.5
    
    ADVANTAGE_NORMALIZE = True
    LR_SCHEDULE = "linear"
    
    # BATCH & ROLLOUT
    ROLLOUT_STEPS = 2048
    MINIBATCH_SIZE = 64
    PPO_EPOCHS = 10
