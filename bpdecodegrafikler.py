import matplotlib.pyplot as plt
import numpy as np

# Örnek veri (gerçek çalışmalardan alınan verilerle değiştirilebilir)
n_values = [200, 400, 600, 800, 1000, 1200, 1500, 1800, 2000, 2500]
message_counts = [1500, 4200, 7300, 10000, 13500, 17000, 22000, 28000, 33000, 42000]
durations = [10.5, 20.1, 29.8, 39.2, 51.0, 62.4, 79.5, 97.2, 110.3, 138.0]

plt.figure(figsize=(12, 5))

# Mesaj karmaşıklığı grafiği
plt.subplot(1, 2, 1)
plt.plot(n_values, message_counts, marker='o', label='Toplam Mesaj')
plt.plot(n_values, [n * np.log2(n) for n in n_values], '--', label='O(n log n)')
plt.xlabel('Düğüm Sayısı (n)')
plt.ylabel('Mesaj Sayısı')
plt.title('Mesaj Karmaşıklığı')
plt.grid(True)
plt.legend()

# Zaman karmaşıklığı grafiği
plt.subplot(1, 2, 2)
plt.plot(n_values, durations, marker='s', color='orange', label='Toplam Süre (sn)')
plt.plot(n_values, [n / 10 for n in n_values], '--', color='gray', label='O(n)')
plt.xlabel('Düğüm Sayısı (n)')
plt.ylabel('Zaman (saniye)')
plt.title('Zaman Karmaşıklığı')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
