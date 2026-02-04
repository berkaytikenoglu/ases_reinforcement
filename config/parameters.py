
class Params:
    # Ekran ve Simülasyon Ayarları
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    FPS = 60
    
    # Fizik
    GRAVITY = 30.0
    DT = 0.1
    
    # Savunma Sistemi (Ajan)
    AMMO_CAPACITY = 60
    AGENT_HEALTH = 6
    RELOAD_TIME = 5
    PROJECTILE_SPEED = 100.0
    
    # Tehdit (Meteor/Hedef)
    THREAT_SPEED_MIN = 15.0       # Daha yavaş (kolay hedef)
    THREAT_SPEED_MAX = 25.0       # Daha yavaş
    
    # Episode Kuralları
    THREATS_PER_EPISODE = 20      # Hard Mode: Gerçek sınav!
    MAX_CONCURRENT_THREATS = 3    # Biraz daha kaos (2->3)
    SPAWN_INTERVAL = 70           # Daha seyrek tehdit
    
    # Rüzgar (Stokastik)
    WIND_CHANGE_INTERVAL = 200
    MAX_WIND_FORCE = 0.5
    
    # Eğitim / RL
    LEARNING_RATE = 0.0003
    MAX_EPISODES = 1000
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
