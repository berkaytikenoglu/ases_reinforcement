
class Rewards:
    """Ödül ve Ceza Değerleri - RL Eğitimi için (Heterojen Strateji)"""
    
    # ===== POZİTİF ÖDÜLLER (ULTIMATE SNIPER) =====
    HIT_REWARD = 5000.0             # HEDEFİ 12'DEN VUR! (+5000) - Vuruş başarısı için dev teşvik
    EARLY_HIT_BONUS = 1500.0        # KAPIDA KARŞILA! (+1500)
    EPISODE_WIN_BONUS = 100000.0    # KAZANMAK HER ŞEYDİR! (+100k)
    AIM_BONUS = 0.0                 # BEDAVA ÖDÜL YOK! Sadece isabet.
    SURVIVAL_BONUS = 0.1            
    ACCURACY_STREAK_BONUS = 200.0    
    ENGAGE_BONUS = 0.0              # Sıkmak marifet değil, vurmak marifet.
    AMMO_EFFICIENCY_BONUS = 500.0   # Mermi başına +500 (Tasarruf çok önemli)
    GOOD_SHOT_BONUS = 1000.0        # CESARET ÖDÜLÜ: Hassas nişanda denemeyi teşvik (+500 -> +1000)
    DISTANCE_MULTIPLIER = 1.0       # İPTAL: 2.0 -> 1.0 (Kumarı önlemek için)
    
    # ===== HETEROJEN STRATEJİ ÖDÜLLER =====
    AREA_DEFENSE_HIT_BONUS = 500.0      # Yüksek irtifa vuruş bonusu (DEVASA!)
    RAPID_FIRE_PRECISION_BONUS = 300.0  # Yakın mesafe hassas vuruş bonusu
    STRATEGY_MATCH_BONUS = 100.0        # Doğru strateji kullanma bonusu
    
    # ===== STRATEJİ MOD EŞİKLERİ =====
    AREA_DEFENSE_DISTANCE = 350     # Piksel - yüksek irtifa modu (parçalayıcı)
    RAPID_FIRE_DISTANCE = 200       # Piksel - kritik yakınlık modu (odaklı seri atış)
    
    # ===== NEGATİF CEZALAR (AĞIR!) =====
    # ===== DİNAMİK CEZALAR (ULTIMATE SCALE) =====
    # ===== DİNAMİK CEZALAR (MESAFEYE GÖRE) =====
    MISS_PENALTY_LONG_RANGE = -1000.0   # ARTTIRILDI: -500 -> -1000 (Uzaktan sallamak yok!)
    MISS_PENALTY_CLOSE_RANGE = -1000.0 
    
    
    MISS_PENALTY = -50.0            
    LATE_ENGAGEMENT_PENALTY = -500.0 
    GROUND_HIT_PENALTY = -5000.0    # Yerden yemenin cezası (-10k -> -5k)
    GROUND_HIT_MULTIPLIER = 2.0  
    DEATH_PENALTY = -50000.0        # Ölüm cezası (-300k -> -50k) - Makul seviye
    
    # EYLEM CEZALARI (Maliyetler)
    FIRE_PENALTY = -100.0           # Mermi maliyeti arttı! (-5 -> -100) - Tetiğe basmak pahalı.
    AMMO_DEPLETED_PENALTY = -10000.0 # Mermi bitti = FAIL (Büyük ceza)
    WASTED_AMMO_PENALTY = -5000.0   # AĞIR CEZA: -500 -> -5000 (Boşa sıkmak YASAK!)
    COOLDOWN_VIOLATION = -50.0      # Cooldown'da ateş (Sabırsızlık cezası)
    RISKY_ANGLE_PENALTY = -20.0     # Tehlikeli açı
    BAD_AIM_PENALTY = -500.0        # CEZA HAFİFLETİLDİ (-2000 -> -500) - Denemekten korkma!
    TIME_STEP_PENALTY = -0.05       # Zaman cezası (hafif)
    
    # ===== IFF (IDENTIFICATION FRIEND OR FOE) =====
    # ===== IFF (IDENTIFICATION FRIEND OR FOE) =====
    FRIENDLY_FIRE_PENALTY = -30000.0  # OPTİMİZE: -50k -> -30k (Korkuyu azalttık, öğrenmeyi hızlandırdık)
    ENEMY_HIT_BONUS = 15000.0         # TEŞVİK: +10k -> +15k (Vurmak daha cazip)
    UAV_ESCAPE_PENALTY = -200.0       # İHA KAÇIRMA: Hafif Ceza
    FRIENDLY_PASS_BONUS = 500.0       # Dostu vurmadan geçirme bonusu
    
    # ===== GUIDANCE (Hassas Nişan) =====
    AIM_ANGLE_THRESHOLD = 0.15      # Radyan - Bonus için çok hassas nişan (0.25 -> 0.15)
    BAD_AIM_THRESHOLD = 0.40        # TOLERANS GENİŞLETİLDİ (0.3 -> 0.4) - Biraz hata payı tanı.
    EARLY_ENGAGEMENT_DISTANCE = 600 # Piksel - kapsam genişletildi (450->600)
    LATE_ENGAGEMENT_DISTANCE = 200  # Piksel - geç angajman (yakın = ceza)

