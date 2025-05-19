import pandas as pd

# Verileri bir DataFrame olarak tanımlayalım
df = pd.DataFrame({
    "n (Düğüm Sayısı)": n_values,
    "Tahmini Mesaj Sayısı": message_counts.astype(int),
    "Encode Sayısı": encode_counts,
    "Toplam Maliyet": total_costs.astype(int)
})

# Görsel olarak kullanıcıya sun
import ace_tools as tools; tools.display_dataframe_to_user(name="Maliyet Hesapları", dataframe=df)

# Grafik çizimi
plt.figure(figsize=(10, 6))
plt.plot(n_values, total_costs, marker='o')
plt.title("Toplam Maliyet vs Düğüm Sayısı (n)")
plt.xlabel("Düğüm Sayısı (n)")
plt.ylabel("Toplam Maliyet (birim)")
plt.grid(True)
plt.tight_layout()
plt.show()
