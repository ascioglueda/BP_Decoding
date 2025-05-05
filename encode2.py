import sys
sys.path.insert(1, '.')
from source import DawnSimVis
import random
import numpy as np

SOURCE = 0
k = 1000  # Orijinal veri boyutu
n = 1200  # Düğüm sayısı

# Rastgele veri oluştur (0 ve 1'lerden oluşan bit dizisi)
data = np.random.randint(0, 2, k, dtype=np.uint8)

# Robust Soliton dağılımı
def soliton_distribution(k, delta=0.5):
    R = k / sum(1.0 / i for i in range(1, k + 1))
    probabilities = [R / i for i in range(1, k + 1)]
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

# LT kodlama fonksiyonu
def encode_lt(data, probabilities):
    degree = np.random.choice(np.arange(1, k + 1), p=probabilities)
    selected_indices = random.sample(range(k), min(degree, k))
    encoded_symbol = data[selected_indices[0]]
    for i in selected_indices[1:]:
        encoded_symbol ^= data[i]
    return selected_indices, encoded_symbol, degree

class Node(DawnSimVis.BaseNode):
    def __init__(self, simulator, node_id, pos, tx_range):
        super().__init__(simulator, node_id, pos, tx_range)
        self.msg_received = False
        self.parent = None
        self.children = []
        self.visited = False
        self.selected_indices = None
        self.encoded_symbol = None
        self.degree = None
        self.received_packets = []
        self.known_values = {}

    def run(self):
        if self.id == SOURCE:
            self.change_color(1, 0, 0)
            probabilities = soliton_distribution(k)
            self.selected_indices, self.encoded_symbol, self.degree = encode_lt(data, probabilities)
            self.log(f'Düğüm {self.id} kodladı: indeksler = {self.selected_indices}, derece = {self.degree}')
            pck = {
                'type': 'encoded',
                'sender': self.id,
                'indices': self.selected_indices,
                'symbol': self.encoded_symbol,
                'degree': self.degree
            }
            self.send(DawnSimVis.BROADCAST_ADDR, pck)
            self.msg_received = True
            self.visited = True
        self.set_timer(20, self.perform_decoding)

    def on_receive(self, pck):
        if pck['type'] == 'encoded':
            self.received_packets.append((pck['sender'], pck['indices'], pck['symbol'], pck['degree']))
            if not self.visited:
                self.parent = pck['sender']
                self.selected_indices = pck['indices']
                self.encoded_symbol = pck['symbol']
                self.degree = pck['degree']
                self.visited = True
                self.change_color(0, 0, 1)
                self.log(f'Düğüm {self.id}, {self.parent} düğümünden aldı: indeksler = {self.selected_indices}, derece = {self.degree}')

                parent_node = self.sim.nodes[self.parent]
                if parent_node:
                    parent_node.children.append(self.id)

                probabilities = soliton_distribution(k)
                self.selected_indices, self.encoded_symbol, self.degree = encode_lt(data, probabilities)
                self.set_timer(1, self.cb_msg_send, {
                    'type': 'encoded',
                    'sender': self.id,
                    'indices': self.selected_indices,
                    'symbol': self.encoded_symbol,
                    'degree': self.degree
                })

    def cb_msg_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        self.change_color(0, 1, 0)
        self.log(f'Düğüm {self.id} mesaj gönderdi')

    def perform_decoding(self):
        self.log(f'Düğüm {self.id}, {len(self.received_packets)} paket ile dekodlamaya başlıyor')

        packets = self.received_packets[:]
        recovered = 0

        while packets:
            progress = False
            for i, (sender, indices, symbol, degree) in enumerate(packets[:]):
                remaining_indices = [idx for idx in indices if idx not in self.known_values]
                if not remaining_indices:
                    packets.pop(i)
                    continue
                current_degree = len(remaining_indices)
                if current_degree == 1:
                    idx = remaining_indices[0]
                    value = symbol
                    for known_idx in indices:
                        if known_idx in self.known_values and known_idx != idx:
                            value ^= self.known_values[known_idx]
                    self.known_values[idx] = value
                    self.log(f'Düğüm {self.id}, indeks {idx} = {value} çözdü')
                    packets.pop(i)
                    recovered += 1
                    progress = True
                elif current_degree == 0:
                    packets.pop(i)
                else:
                    new_symbol = symbol
                    for known_idx in indices:
                        if known_idx in self.known_values:
                            new_symbol ^= self.known_values[known_idx]
                    packets[i] = (sender, remaining_indices, new_symbol, current_degree)

            if not progress:
                break

        success_rate = len(self.known_values) / k
        unresolved_symbols = k - len(self.known_values)
        self.log(f'Düğüm {self.id} dekodlama tamamlandı: {recovered} değer kurtarıldı')
        self.log(f'Bilinen değerler: {len(self.known_values)}/{k}')
        self.log(f'Başarı oranı: {success_rate:.3f}')
        self.log(f'Çözülemeyen sembol sayısı: {unresolved_symbols}')

    def finish(self):
        if self.parent is not None:
            self.log(f'Düğüm {self.id} -> ebeveyn = {self.parent}')
        if self.children:
            self.log(f'Düğüm {self.id} -> çocuklar = {self.children}')
        if self.known_values:
            self.log(f'Düğüm {self.id} -> bilinen_değerler = {len(self.known_values)} değer')

def create_network():
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Simülatör oluştur
sim = DawnSimVis.Simulator(
    duration=300,
    timescale=1,
    visual=True,
    terrain_size=(2500, 2000),
    title='LT Kod ile İnanç Yayılımı Dekodlama'
)

# Ağı oluştur
create_network()

# Simülasyonu çalıştır
sim.run()

# 🔍 Global başarı oranını hesapla
all_known_indices = set()
for node in sim.nodes.values():
    all_known_indices.update(node.known_values.keys())

print(f'\n📊 Global başarı oranı: {len(all_known_indices)}/{k}\n')
