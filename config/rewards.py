
class Rewards:
    """Ödül ve Ceza Değerleri - RL Eğitimi için"""
    
    # ===== POZİTİF ÖDÜLLER =====
    # ===== POZİTİF ÖDÜLLER =====
    HIT_REWARD = 150.0              # Tehdidi vurma ödülü
    EPISODE_WIN_BONUS = 1000.0      # Tüm tehditleri yok ederse bonus
    AIM_BONUS = 5.0                 # Doğru açıyla ateş etme bonusu
    SURVIVAL_BONUS = 1.0            # Her adımda hayatta kalma bonusu
    
    # ===== NEGATİF CEZALAR =====
    MISS_PENALTY = -15.0            # Atış yapıp ıskalama cezası (Artırıldı: Mermiyi boşa harcama!)
    GROUND_HIT_PENALTY = -500.0     # Tehdit yere çarptığında (ilk ceza)
    GROUND_HIT_MULTIPLIER = 1.5     # Her hatada ceza katlanır
    DEATH_PENALTY = -2000.0         # Ajan 3 kez vurulup ölürse
    WASTED_AMMO_PENALTY = -30.0     # Tehdit yokken ateş etme cezası (Çok Ciddi Ceza)
    COOLDOWN_VIOLATION = -5.0       # Cooldown'dayken ateş etmeye çalışma
    RISKY_ANGLE_PENALTY = -3.0      # Tehlikeli açıyla ateş etme
    TIME_STEP_PENALTY = -0.05       # Her adımda küçük zaman cezası
    
    # ===== GUIDANCE (Eğitimi Hızlandırma) =====
    AIM_ANGLE_THRESHOLD = 0.3       # Radyan - hedef açı toleransı
    
    # ===== ESKİ UYUMLULUK (Geçici) =====
    BASE_HIT_REWARD = 100.0
    HIT_HEIGHT_MULTIPLIER = 0.1
    GAME_OVER_PENALTY = -300.0
    FRIENDLY_FIRE_PENALTY = -5000.0
    COOLDOWN_VIOLATION_PENALTY = -5.0
