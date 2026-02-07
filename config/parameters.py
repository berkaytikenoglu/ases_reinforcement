
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
    PROJECTILE_RADIUS = 30       # ELITE TASARIM: 25 -> 30 (Zor ama İmkansız Değil - Tam Denge)
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
    PHASE1_EPISODES = 5000           # Standart: 5000 Episode (Updated)
    
    # Phase 2 (Curriculum): One Shot Mode
    PHASE2_EPISODES = 10000          # BARAJ %90: 10.000 Episode (Profesyonel Eğitim)
    PHASE2_THREATS = 5               # MULTI-TARGET: 5 hedef
    PHASE2_AMMO = 40                 # BUFFER: 20 -> 40 (Target başı 8 mermi - Hassasiyet için geniş yer)
    PHASE2_HEALTH = 3                # REALİSTİK Buffer: 5 çok fazlaydı, 3'e çekildi (2 hata payı)
    PHASE2_RELOAD_TIME = 15          # HIZLI RELOAD: 30 -> 15 (İkinci şans için zaman)
    PHASE2_THREAT_SPEED_MIN = 20.0   
    PHASE2_THREAT_SPEED_MAX = 35.0   

    # Phase 3 (Curriculum): IFF Recognition (Friend or Foe)
    PHASE3_EPISODES = 10000          # Elite Standart: 10.000 Episode (IFF Uzmanlığı)
    PHASE3_THREATS = 5               # SPEED RUN: 8 -> 5 (Daha kısa maçlar, daha hızlı eğitim)
    PHASE3_FRIENDLY_RATIO = 0.2      # 5 * 0.2 = 1 Dost (Tam kararında)
    PHASE3_HORIZONTAL_RATIO = 0.5    # %50 yatay uçuş
    PHASE3_AMMO = 60                 # USER STRATEGY: 40 -> 60 (Bol mermi ile rahat eğitim + Tasarruf Bonusu)
    PHASE3_HEALTH = 3                # 3 can (biraz tolerans)
    PHASE3_RELOAD_TIME = 15          # SPEEDY: 30 -> 15 (Faz 2 ile Eşitlendi - Seri Atış)
    
    # Phase 3 UAV Speed (Horizontal) - BALANCED FOR LEARNING
    PHASE3_UAV_SPEED_MIN = 25.0      # Balistik füze kadar hızlı (40 -> 25)
    PHASE3_UAV_SPEED_MAX = 40.0      # Yakalanabilir hız (60 -> 40)
    
    # Phase 4 (Curriculum): War Mode (Intense Combat)
    PHASE4_THREATS = 15              # Yoğun savaş: 15 hedef
    PHASE4_AMMO = 60                 # Bol mermi (Updated from 40)
    PHASE4_HEALTH = 3                # 3 can

    
    # Episode Kuralları
    THREATS_PER_EPISODE = 10     # Normal: 10 tehdit/episode
    MAX_CONCURRENT_THREATS = 3   # Teke tek
    SPAWN_INTERVAL = 60          # Normal: 60 frame aralık (User Request: More waiting)
    
    # Rüzgar (Stokastik)
    WIND_CHANGE_INTERVAL = 200
    MAX_WIND_FORCE = 0.5
    
    # Eğitim / RL (PPO Hyperparameters) - PRECISION TUNING (STABLE > AGGRESSIVE)
    LEARNING_RATE = 0.0003           # STABİLİZE: 0.0005 çok hızlı saçmalayabiliyor, 0.0003 ideal.
    MAX_EPISODES = 5000              
    GAMMA = 0.99
    
    # Advanced PPO
    LAMBDA = 0.95
    CLIP_RANGE = 0.2              # HASSAS GÜNCELLEME: 0.3 çok sert, 0.2 ile ağır ağır doğruyu bulur.
    ENTROPY_COEF = 0.01           # KEŞİF: 0.005 çok az, 0.01 ile daha fazla açı dener (Precision için şart).
    VALUE_COEF = 0.5
    
    ADVANTAGE_NORMALIZE = True
    LR_SCHEDULE = "linear"
    
    # BATCH & ROLLOUT
    ROLLOUT_STEPS = 2048
    MINIBATCH_SIZE = 64
    PPO_EPOCHS = 10
