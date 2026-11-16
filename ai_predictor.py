"""
================================================================================
MODÜL ADI : ai_predictor.py
PROJE     : İHA Kamera Görüntülerinde Kalman Filtresi ve YZ Destekli Hedef Takibi
AMAÇ      : Sabit-hız (doğrusal) Kalman filtresinin tek başına öngöremeyeceği
            eğrisel manevraları (S-kıvrımı, viraj dönüşü) öğrenmek için hedefin
            son 8 karelik yer değiştirme vektörlerinden çevrimiçi (online) eğitim
            yapan ve gelecek 10 karelik rotayı çizen PyTorch MLP sinir ağı.
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn


class TrajectoryMLP(nn.Module):
    """
    Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP) Yörünge Ağı:
      - Giriş Katmanı  : Son 'seq_len' (8) karenin (dx, dy) farkları -> 8 * 2 = 16 giriş
      - 1. Gizli Katman: 32 nöron + ReLU aktivasyon fonksiyonu (doğrusal olmayanlık)
      - 2. Gizli Katman: 32 nöron + ReLU aktivasyon fonksiyonu
      - Çıkış Katmanı  : Gelecek 'pred_len' (10) karenin (dx, dy) tahmini -> 10 * 2 = 20 çıkış
    """

    def __init__(self, seq_len=8, pred_len=10):
        super().__init__()
        self.seq_len = seq_len    # Geçmişe dönük bakılacak pencere uzunluğu (8 kare)
        self.pred_len = pred_len  # Geleceğe dönük tahmin edilecek adım sayısı (10 kare)

        # Tam bağlantılı (Fully Connected) yapay sinir ağı katman dizilimi
        self.net = nn.Sequential(
            nn.Linear(seq_len * 2, 32),   # 16 giriş özelliğini 32 gizli nörona bağlar
            nn.ReLU(),                    # Negatif değerleri sıfırlayarak eğrisel dönüşleri öğrenir
            nn.Linear(32, 32),            # 32 nörondan 32 nörona derin özellik çıkarımı
            nn.ReLU(),
            nn.Linear(32, pred_len * 2)   # Gelecek 10 adımın (dx, dy) vektörlerini üretir
        )

    def forward(self, x):
        """İleri yayılım (Forward Pass): Girdi tensörünü ağdan geçirip tahmini döndürür."""
        return self.net(x)


class OnlineTrajectoryPredictor:
    """
    Çevrimiçi (Online) Öğrenme Yöneticisi:
    Önceden büyük bir veri setiyle eğitilmeye ihtiyaç duymaz. Takip edilen her
    hedef (ID) için bağımsız oluşturulur; hedef sahnede ilerledikçe kendi geçmiş
    hareketinden anlık ağırlık güncellemesi (backpropagation) yaparak öğrenir.
    """

    def __init__(self, seq_len=8, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.history = []  # Kalman filtresinden gelen (x, y) konumlarının tutulduğu tampon

        # Sinir ağı modeli, Adam optimizasyon algoritması ve Ortalama Kare Hata (MSE) kaybı
        self.model = TrajectoryMLP(seq_len=seq_len, pred_len=pred_len)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def add_point(self, x, y):
        """Kalman filtresinin ürettiği güncel (est_x, est_y) konumunu geçmiş listesine ekler."""
        self.history.append([float(x), float(y)])

        # Bellek şişmesini önlemek ve hedefin çok eski manevralarını unutmasını sağlamak için
        # kayan tampon (sliding buffer) yalnızca son 120 kareyi hafızada tutar.
        if len(self.history) > 120:
            self.history.pop(0)

    def train_and_predict(self):
        """
        1) Geçmiş konumlardan hız/yer değiştirme vektörlerini (dx, dy) çıkarır.
        2) Yeterli veri biriktiyse kayan pencere yöntemiyle ağı 5 iterasyon eğitir.
        3) Son 8 hareket adımını ağa vererek gelecek 10 karelik koordinatları üretir.
        """
        # Eğitim ve tahmin için en az seq_len + 2 nokta birikmesi beklenir
        if len(self.history) < self.seq_len + 2:
            return []

        pts = np.array(self.history, dtype=np.float32)

        # ÖNEMLİ MÜHENDİSLİK KARARI:
        # Doğrudan (x, y) piksel koordinatı yerine ardışık kareler arası farklar (dx, dy)
        # kullanılır. Böylece model ekranın sol üstündeki bir dönüşü öğrendiğinde,
        # sağ altındaki benzer bir dönüşü de konumdan bağımsız (translation-invariant) bilir.
        deltas = np.diff(pts, axis=0)

        # Kayan Pencere (Sliding Window) ile çevrimiçi eğitim verisi oluşturma:
        if len(deltas) >= self.seq_len + self.pred_len:
            X_train, Y_train = [], []
            num_samples = len(deltas) - self.seq_len - self.pred_len + 1

            for i in range(num_samples):
                # Girdi (X): i'den başlayan 8 karelik yer değiştirme dizisi
                x_window = deltas[i : i + self.seq_len].flatten()
                # Hedef (Y): Hemen ardından gelen 10 karelik gerçek yer değiştirme dizisi
                y_window = deltas[
                    i + self.seq_len : i + self.seq_len + self.pred_len
                ].flatten()
                X_train.append(x_window)
                Y_train.append(y_window)

            X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
            Y_tensor = torch.tensor(np.array(Y_train), dtype=torch.float32)

            # Gerçek zamanlı hızı korumak için her karede 5 hızlı gradyan adımı atılır
            self.model.train()
            for _ in range(5):
                self.optimizer.zero_grad()                   # Eski türevleri sıfırla
                predictions = self.model(X_tensor)           # İleri yayılım
                loss = self.criterion(predictions, Y_tensor) # Tahmin-gerçek arası MSE hatası
                loss.backward()                              # Geri yayılım (türev hesabı)
                self.optimizer.step()                        # Ağırlıkları güncelle

        # Çıkarım (Inference) Aşaması: Gradyan hesaplamasını kapatarak gelecek 10 adımı tahmin et
        self.model.eval()
        with torch.no_grad():
            recent_deltas = deltas[-self.seq_len:].flatten()
            inp = torch.tensor(recent_deltas, dtype=torch.float32).unsqueeze(0)
            pred_deltas = (
                self.model(inp)
                .squeeze(0)
                .numpy()
                .reshape(self.pred_len, 2)
            )

        # Tahmin edilen (dx, dy) adımlarını hedefin son Kalman konumuna kümülatif ekle
        curr = pts[-1].copy()
        future_points = []
        for dx, dy in pred_deltas:
            curr = curr + np.array([dx, dy])
            future_points.append((int(curr[0]), int(curr[1])))

        return future_points
