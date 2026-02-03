
class Params:
    # Ekran ve Simülasyon Ayarları
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 60
    
    # Fizik
    GRAVITY = 30.0
    DT = 0.1
    
    # Savunma Sistemi (Ajan)
    AMMO_CAPACITY = 100
    RELOAD_TIME = 10
    PROJECTILE_SPEED = 100.0
    
    # Tehdit (Meteor/Hedef)
    THREAT_SPEED_MIN = 40.0
    THREAT_SPEED_MAX = 60.0
    
    # Episode Kuralları
    THREATS_PER_EPISODE = 50      # Her episode'da toplam tehdit sayısı
    MAX_CONCURRENT_THREATS = 3    # Aynı anda maksimum tehdit sayısı
    SPAWN_INTERVAL = 30           # Kaç frame'de bir yeni tehdit spawn olsun (düşük = daha sık)
    
    # Rüzgar (Stokastik)
    WIND_CHANGE_INTERVAL = 200
    MAX_WIND_FORCE = 0.5
    
    # Eğitim / RL
    LEARNING_RATE = 0.0003
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995

