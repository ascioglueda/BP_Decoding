import sys
sys.path.insert(1, '.')
from source import DawnSimVis
import random
import numpy as np
from collections import deque

SOURCE = 0
k = 1000  # Orijinal veri boyutu
n = 2500  # Düğüm sayısı

# Veri oluşturma (0 ve 1'lerden oluşan bit dizisi)
data = np.random.randint(0, 2, k, dtype=np.uint8)

# Robust Soliton dağılımı
def soliton_distribution(k, delta=0.5):
    R = k / sum(1.0 / i for i in range(1, k + 1))
    probabilities = [R / i for i in range(1, k + 1)]
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # Normalizasyon
    return probabilities

# Soliton dağılımını hesapla
probabilities = soliton_distribution(k)

# Encoding işlemi
def encode_lt(data, probabilities):
    degree = np.random.choice(np.arange(1, k + 1), p=probabilities)
    selected_indices = random.sample(range(k), min(degree, k))
    encoded_symbol = data[selected_indices[0]]
    for i in selected_indices[1:]:
        encoded_symbol ^= data[i]  # XOR işlemi
    return selected_indices, encoded_symbol

# Decoding işlemi (Belief Propagation)
def decode_lt(requesting_node_id, received_packets, k):
    """Belief Propagation ile LT kod çözümü."""
    print(f"\nDüğüm {requesting_node_id} decode işlemi için mesaj gönderiyor...")

    # Paketleri kopyala
    received_data = [(list(indices), value) for indices, value in received_packets]

    known_values = np.full(k, -1, dtype=np.int8)  # Bilinmeyenler -1
    symbol_queue = deque()

    # Başlangıçta derecesi 1 olan sembolleri sıraya al
    for node_id, (indices, value) in enumerate(received_data):
        if len(indices) == 1:
            idx = indices[0]
            if known_values[idx] == -1:
                symbol_queue.append((idx, value))

    iteration = 0
    while symbol_queue:
        iteration += 1
        symbol_idx, value = symbol_queue.popleft()
        if known_values[symbol_idx] != -1:
            continue  # Zaten biliniyorsa atla
        known_values[symbol_idx] = value
        # Diğer tüm kod sembollerini bu bilgi ile güncelle
        for i in range(len(received_data)):
            indices, val = received_data[i]
            if symbol_idx in indices:
                indices.remove(symbol_idx)
                val ^= value
                received_data[i] = (indices, val)
                if len(indices) == 1:
                    new_idx = indices[0]
                    if known_values[new_idx] == -1:
                        symbol_queue.append((new_idx, val))
    return known_values

class Node(DawnSimVis.BaseNode):

    def __init__(self, simulator, node_id, pos, tx_range):
        super().__init__(simulator, node_id, pos, tx_range)
        self.msg_received = False
        self.parent = None
        self.children = []
        self.visited = False
        self.selected_indices = None
        self.encoded_symbol = None
        self.received_packets = []
        self.known_bits = [None] * k
        self.degree = 0

    def run(self):
        if self.id == SOURCE:
            self.change_color(1, 0, 0)
            probabilities = soliton_distribution(k)
            self.selected_indices, self.encoded_symbol = encode_lt(data, probabilities)
            self.degree = len(self.selected_indices)
            pck = {'type': 'encoded', 'sender': self.id, 'indices': (self.selected_indices, self.encoded_symbol)}
            self.send(DawnSimVis.BROADCAST_ADDR, pck)
            self.msg_received = True
            self.visited = True

    def on_receive(self, pck):
        if pck['type'] == 'encoded':
            self.received_packets.append((pck['sender'], pck['indices']))
            self.degree = len(pck['indices'][0])

            if not self.visited:
                self.parent = pck['sender']
                self.selected_indices, self.encoded_symbol = pck['indices']
                self.visited = True
                self.change_color(0, 0, 1)
                self.log(f'Düğüm {self.id}, {self.parent}. düğümden paket aldı: indeksler = {self.selected_indices}')

                parent_node = self.sim.nodes[self.parent]
                if parent_node:
                    parent_node.children.append(self.id)

                probabilities = soliton_distribution(k)
                self.selected_indices, self.encoded_symbol = encode_lt(data, probabilities)

                self.set_timer(1, self.cb_msg_send, {
                    'type': 'encoded',
                    'sender': self.id,
                    'indices': (self.selected_indices, self.encoded_symbol)
                })

    def cb_msg_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        self.change_color(0, 1, 0)
        self.log(f'Mesaj gönderildi: {self.id}')

    def finish(self):
        if self.received_packets:
            self.log(f'Düğüm {self.id} -> Alinan Paketler: {[(sender, indices[0]) for sender, indices in self.received_packets]}')
            flattened = [packet for _, packet in self.received_packets]
            decoded_data = decode_lt(self.id, flattened, k)
            decoded_count = np.count_nonzero(decoded_data != -1)
            self.log(f'Cozulen bit sayisi: {decoded_count}/{k}')

def create_network():
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Simülatör nesnesi oluştur
sim = DawnSimVis.Simulator(
    duration=300,
    timescale=1,
    visual=True,
    terrain_size=(650, 650),
    title='Belief Propagation Decoding')

# Ağı oluştur
create_network()

# Simülasyonu başlat
sim.run()

