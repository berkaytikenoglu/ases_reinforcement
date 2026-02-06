
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
    PROJECTILE_SPEED = 600.0     # HIZLI VE ÖFKELİ: 500 -> 600 (User Request)
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

    # Phase 3 (Curriculum): IFF Recognition (Friend or Foe)
    PHASE3_EPISODES = 2500           # Standart: 2500 Episode
    PHASE3_THREATS = 8               # Toplam hedef (dost + düşman karışık)
    PHASE3_FRIENDLY_RATIO = 0.4      # %40 dost, %60 düşman
    PHASE3_HORIZONTAL_RATIO = 0.5    # %50 yatay uçuş
    PHASE3_AMMO = 20                 # Sadece düşmanlar için yeterli
    PHASE3_HEALTH = 3                # 3 can (biraz tolerans)
    PHASE3_RELOAD_TIME = 30          # SLOWER RELOAD: (User Request)
    
    # Phase 3 UAV Speed (Horizontal) - User Request: "Too Slow" -> Boosted
    PHASE3_UAV_SPEED_MIN = 40.0
    PHASE3_UAV_SPEED_MAX = 60.0      # Double the normal speed
    
    # Phase 4 (Curriculum): War Mode (Intense Combat)
    PHASE4_THREATS = 15              # Yoğun savaş: 15 hedef
    PHASE4_AMMO = 40                 # Bol mermi
    PHASE4_HEALTH = 3                # 3 can

    
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
