"""
================================================================================
MODÜL ADI : kalman_filter.py
PROJE     : İHA Kamera Görüntülerinde Kalman Filtresi ve YZ Destekli Hedef Takibi
AMAÇ      : Görüntü düzleminde hareket eden bir hedefin konum ve hızını 4 boyutlu
            durum uzayında kestiren Çoklu-Sensör Doğrusal Kalman Filtresi sınıfı.
            - 1. Sensör (Kamera / Elektro-Optik): Düşük gürültülü (hassas) ölçüm.
            - 2. Sensör (Radar / Telemetri)     : Yüksek gürültülü yedek ölçüm.
            Kamera bulut/engel (oklüzyon) nedeniyle kapandığında filtre otomatik
            olarak radar ölçümü ve sabit-hız hareket modeli ile takibi sürdürür.
================================================================================
"""

import numpy as np


class MultiSensorKalmanFilter:
    """
    Çoklu-Sensör Doğrusal Kalman Filtresi (Linear Kalman Filter) Sınıfı.

    Durum Vektörü (State Vector):
        x = [px, py, vx, vy]^T   (4 satır x 1 sütun vektörü)
        - px : Hedefin görüntüdeki yatay merkez koordinatı (piksel)
        - py : Hedefin görüntüdeki dikey merkez koordinatı (piksel)
        - vx : Hedefin yatay (x) eksenindeki hız bileşeni (piksel / kare)
        - vy : Hedefin dikey (y) eksenindeki hız bileşeni (piksel / kare)
    """

    def __init__(self, dt=1.0, process_noise=1.5, cam_noise=3.0, radar_noise=18.0):
        # dt: İki ardışık video karesi arasındaki örnekleme zaman adımı (1 kare = 1.0 birim)
        self.dt = float(dt)

        # x: Başlangıç durum vektörü [px=0, py=0, vx=0, vy=0]^T
        self.x = np.zeros((4, 1), dtype=float)

        # F: Durum Geçiş Matrisi (State Transition Matrix - 4x4)
        # Sabit Hız (Constant Velocity) kinematik denklemlerini matris formunda tutar:
        #   px_yeni = 1*px_eski + 0*py_eski + dt*vx_eski + 0*vy_eski
        #   py_yeni = 0*px_eski + 1*py_eski + 0*vx_eski + dt*vy_eski
        #   vx_yeni = vx_eski  (Sabit hız varsayımı)
        #   vy_yeni = vy_eski
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0,     0.0],
            [0.0, 0.0, 0.0,     1.0]
        ], dtype=float)

        # H: Ölçüm Gözlem Matrisi (Observation Matrix - 2x4)
        # Sensörler bize doğrudan hızı (vx, vy) vermez; yalnızca [px, py] konumunu ölçer.
        # Bu nedenle 4 boyutlu durum vektörünü 2 boyutlu ölçüm uzayına izdüşürür:
        #   z_beklenen = H * x = [px, py]^T
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=float)

        # P: Durum Tahmin Hata Kovaryans Matrisi (Error Covariance Matrix - 4x4)
        # Filtrenin kendi tahminine ne kadar "güvenmediğini" (belirsizliği) gösterir.
        # Başlangıçta hedef konumu bilinmediği için köşegen elemanlar yüksek (500.0) seçilir.
        self.P = np.eye(4, dtype=float) * 500.0

        # Q: Süreç Gürültüsü Kovaryans Matrisi (Process Noise Covariance - 4x4)
        # Hedefin sabit hız modelinden sapma ihtimalini (ani fren, gaz, keskin dönüş) modeller.
        # Q büyütülürse filtre manevralara daha hızlı tepki verir ama gürültüye duyarlılaşır.
        self.Q = np.eye(4, dtype=float) * float(process_noise)

        # R_cam ve R_radar: Sensör Ölçüm Gürültü Kovaryans Matrisleri (2x2)
        # Varyans = (Standart Sapma)^2 kuralı gereği sigma^2 ile ölçeklenir:
        # - Kamera hassas olduğu için R_cam = 3^2 = 9.0 (düşük belirsizlik -> yüksek güven)
        # - Radar gürültülü olduğu için R_radar = 18^2 = 324.0 (yüksek belirsizlik -> düşük güven)
        self.R_cam = np.eye(2, dtype=float) * (float(cam_noise) ** 2)
        self.R_radar = np.eye(2, dtype=float) * (float(radar_noise) ** 2)

        # initialized: Filtrenin ilk hedef tespitiyle kilitlenip kilitlenmediği bilgisi
        self.initialized = False

    def initialize(self, x, y):
        """
        Hedef ilk kez tespit edildiğinde çağrılır.
        Durum vektörünü tespit edilen (x, y) noktasına kilitler, başlangıç hızını 0 atar
        ve başlangıç belirsizliğini (P) düşürür.
        """
        self.x = np.array([[float(x)], [float(y)], [0.0], [0.0]], dtype=float)
        self.P = np.eye(4, dtype=float) * 100.0
        self.initialized = True

    def predict(self):
        """
        1. AŞAMA: ZAMAN GÜNCELLEMESİ (TIME UPDATE / PREDICT)
        Henüz yeni sensör ölçümü gelmeden, fiziksel hareket modelini (F) kullanarak
        hedefin bir sonraki karede nerede olacağını önceden kestirir (A Priori Tahmin).
        """
        # Denklem 1: Önsel durum tahmini -> x_{k|k-1} = F * x_{k-1|k-1}
        self.x = self.F @ self.x

        # Denklem 2: Önsel hata kovaryansı -> P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        # Zaman ilerledikçe ve ölçüm alınmadıkça Q kadar belirsizlik artar.
        self.P = self.F @ self.P @ self.F.T + self.Q

        return float(self.x[0, 0]), float(self.x[1, 0])

    def _update(self, z, R):
        """
        2. AŞAMA: ÖLÇÜM GÜNCELLEMESİ (MEASUREMENT UPDATE / CORRECTION)
        Sensörden gelen z = [u, v]^T ölçümünü, önsel tahminle birleştirerek
        düzeltilmiş (A Posteriori) konum ve hız kestirimini hesaplar.
        """
        # Gelen 2 elemanlı ölçümü (2x1) sütun vektörüne çevir
        z_vec = np.array(z, dtype=float).reshape(2, 1)

        # Denklem 3: İnovasyon / Ölçüm Artığı (y)
        # Sensörün ölçtüğü konum ile filtrenin tahmin ettiği konum arasındaki fark:
        # y = z_k - H * x_{k|k-1}
        y = z_vec - (self.H @ self.x)

        # Denklem 4: İnovasyon Kovaryansı (S)
        # Tahmin belirsizliği ile sensör gürültüsünün toplamı:
        # S = H * P_{k|k-1} * H^T + R
        S = self.H @ self.P @ self.H.T + R

        # Denklem 5: Optimal Kalman Kazancı (Kalman Gain - K)
        # Sensöre mi yoksa kendi tahminimize mi daha çok güveneceğimizi belirleyen ağırlık:
        # K = P_{k|k-1} * H^T * S^(-1)
        # R (sensör gürültüsü) çok büyükse K sıfıra yaklaşır (gürültülü sensör bastırılır).
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Denklem 6: Sonsal Durum Güncellemesi (A Posteriori State Estimate)
        # Önsel tahmine, Kalman kazancı ile ağırlıklandırılmış ölçüm farkı eklenir:
        # x_{k|k} = x_{k|k-1} + K * y
        self.x = self.x + (K @ y)

        # Denklem 7: Sonsal Hata Kovaryansı Güncellemesi
        # Yeni ölçüm alındığı için sistemdeki konum belirsizliği azalır:
        # P_{k|k} = (I - K * H) * P_{k|k-1}
        I = np.eye(4, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def update_camera(self, z_cam):
        """Kamera sensöründen gelen hassas piksel ölçümü (R_cam) ile düzeltme yapar."""
        self._update(z_cam, self.R_cam)

    def update_radar(self, z_radar):
        """Radar sensöründen gelen gürültülü konum ölçümü (R_radar) ile düzeltme yapar."""
        self._update(z_radar, self.R_radar)

    def get_state(self):
        """Filtrenin güncel kestirdiği [px, py, vx, vy] değerlerini Python float olarak döner."""
        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
            float(self.x[3, 0])
        )
