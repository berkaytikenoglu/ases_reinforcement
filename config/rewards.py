
class Rewards:
    """Ödül ve Ceza Değerleri - RL Eğitimi için (Heterojen Strateji)"""
    
    # ===== POZİTİF ÖDÜLLER (DENGELİ SNIPER) =====
    HIT_REWARD = 2500.0             # Vuruş değerli olmalı ki denesin (500->2500)
    EARLY_HIT_BONUS = 500.0         
    EPISODE_WIN_BONUS = 50000.0     # Büyük hedef sabit
    AIM_BONUS = 50.0                
    SURVIVAL_BONUS = 0.1            
    ACCURACY_STREAK_BONUS = 100.0    
    ENGAGE_BONUS = 5.0              
    AMMO_EFFICIENCY_BONUS = 200.0   
    
    # ===== HETEROJEN STRATEJİ ÖDÜLLER =====
    AREA_DEFENSE_HIT_BONUS = 500.0      # Yüksek irtifa vuruş bonusu (DEVASA!)
    RAPID_FIRE_PRECISION_BONUS = 300.0  # Yakın mesafe hassas vuruş bonusu
    STRATEGY_MATCH_BONUS = 100.0        # Doğru strateji kullanma bonusu
    
    # ===== STRATEJİ MOD EŞİKLERİ =====
    AREA_DEFENSE_DISTANCE = 350     # Piksel - yüksek irtifa modu (parçalayıcı)
    RAPID_FIRE_DISTANCE = 200       # Piksel - kritik yakınlık modu (odaklı seri atış)
    
    # ===== NEGATİF CEZALAR (AĞIR!) =====
    # ===== DİNAMİK CEZALAR (MESAFEYE GÖRE) =====
    # ===== DİNAMİK CEZALAR (MESAFEYE GÖRE) =====
    MISS_PENALTY_LONG_RANGE = -500.0  # Tolere edilebilir hata (-2000 -> -500)
    MISS_PENALTY_CLOSE_RANGE = -200.0 
    
    FIRE_PENALTY = -25.0            # Makul ateş maliyeti (-100 -> -25)
    
    MISS_PENALTY = -50.0            # (Varsayılan - kullanılmayacak)
    LATE_ENGAGEMENT_PENALTY = -500.0 
    GROUND_HIT_PENALTY = -5000.0    # ASLA GEÇİT VERME! (-2500 -> -5000)
    GROUND_HIT_MULTIPLIER = 3.0     # Hata affedilmez
    DEATH_PENALTY = -3000.0         # Ajan ölürse (Çok ağır)
    WASTED_AMMO_PENALTY = -200.0    # Tehdit yokken ateş (Yasak!)
    COOLDOWN_VIOLATION = -50.0      # Cooldown'da ateş (Sabırsızlık cezası)
    RISKY_ANGLE_PENALTY = -20.0     # Tehlikeli açı
    BAD_AIM_PENALTY = -80.0         # Kötü nişanla ateş (AĞIR!)
    TIME_STEP_PENALTY = -0.05       # Zaman cezası (hafif)
    
    # ===== GUIDANCE (Hassas Nişan) =====
    AIM_ANGLE_THRESHOLD = 0.25      # Radyan - iyi nişan toleransı (Gevşetildi)
    BAD_AIM_THRESHOLD = 0.50        # Radyan - kötü nişan eşiği (Gevşetildi)
    EARLY_ENGAGEMENT_DISTANCE = 450 # Piksel - erken angajman (Dome'a girer girmez!)
    LATE_ENGAGEMENT_DISTANCE = 200  # Piksel - geç angajman (yakın = ceza)
    
    # ===== ESKİ UYUMLULUK =====
    BASE_HIT_REWARD = 100.0
    HIT_HEIGHT_MULTIPLIER = 0.1
    GAME_OVER_PENALTY = -300.0
    FRIENDLY_FIRE_PENALTY = -5000.0
    COOLDOWN_VIOLATION_PENALTY = -5.0
    PATIENCE_BONUS = 0.0

