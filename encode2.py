import sys
sys.path.insert(1, '.')
from source import DawnSimVis
import random
import numpy as np

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

# Encoding işlemi
def encode_lt(data, probabilities):
    degree = np.random.choice(np.arange(1, k + 1), p=probabilities)
    selected_indices = random.sample(range(k), min(degree, k))
    encoded_symbol = data[selected_indices[0]]
    for i in selected_indices[1:]:
        encoded_symbol ^= data[i]  # XOR işlemi
    return selected_indices, encoded_symbol

# Decoding işlemi (Belief Propagation)
def decode(received_packets, k):
    decoded = [None] * k
    packets = []

    for _, (indices, value) in received_packets:
        packets.append({'indices': set(indices), 'value': value})

    from collections import deque
    queue = deque()

    # Başlangıçta derecesi 1 olan sembolleri sıraya al
    for packet in packets:
        if len(packet['indices']) == 1:
            queue.append(packet)

    while queue:
        packet = queue.popleft()
        indices = packet['indices']

        if not indices:
            continue

        idx = next(iter(indices))
        value = packet['value']

        # Eğer bu bit zaten çözüldüyse devam et
        if decoded[idx] is not None:
            continue

        decoded[idx] = value

        # Bu biti içeren diğer sembolleri güncelle
        for other_packet in packets:
            if idx in other_packet['indices']:
                other_packet['indices'].remove(idx)
                other_packet['value'] ^= value
                if len(other_packet['indices']) == 1:
                    queue.append(other_packet)

    # Başarı oranını ve çözülen bitleri raporla
    decoded_count = sum(x is not None for x in decoded)
    return decoded, decoded_count, [(i, b) for i, b in enumerate(decoded) if b is not None]



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

    def run(self):
        if self.id == SOURCE:
            self.change_color(1, 0, 0)  # Root kırmızı
            probabilities = soliton_distribution(k)
            self.selected_indices, self.encoded_symbol = encode_lt(data, probabilities)
            self.log(f'Düğüm {self.id} şifreleme yaptı: değer = {self.selected_indices}')
            pck = {'type': 'encoded', 'sender': self.id, 'indices': (self.selected_indices, self.encoded_symbol)}
            self.send(DawnSimVis.BROADCAST_ADDR, pck)
            self.msg_received = True
            self.visited = True

    def on_receive(self, pck):
        if pck['type'] == 'encoded':
            self.received_packets.append((pck['sender'], pck['indices']))

            if not self.visited:
                self.parent = pck['sender']
                self.selected_indices, self.encoded_symbol = pck['indices']
                self.visited = True
                self.change_color(0, 0, 1)  # Mesaj alındığında mavi renk
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
        self.change_color(0, 1, 0)  # Mesaj gönderildikten sonra yeşil
        self.log(f'Mesaj gönderildi: {self.id}')

    def finish(self):
        if self.received_packets:
            self.log(f'Düğüm {self.id} -> Alınan Paketler: {[(sender, indices[0]) for sender, indices in self.received_packets]}')
            decoded_data, count, _ = decode(self.received_packets, k)
            self.log(f'Şifre çözme tamamlandı. Çözülen bit sayısı: {count}/{k}')

def create_network():
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Simülatör nesnesi oluştur
sim = DawnSimVis.Simulator(
    duration=100,
    timescale=1,
    visual=True,
    terrain_size=(650, 650),
    title='Belief Propagation Decoding')

# Ağı oluştur
create_network()

# Simülasyonu başlat
sim.run()

all_packets = []
for node in sim.nodes:
    if hasattr(node, 'received_packets'):
        all_packets.extend(node.received_packets)

decoded_data, decoded_count, _ = decode(all_packets, k)

# Özet sonuçları göster
print(f"\nÇözülen bit sayısı: {decoded_count}/{k}")
basari_orani = decoded_count / k
print(f"Başarı oranı: {basari_orani:.2f}")
print(f"Çözülemeyen sembol sayısı: {k - decoded_count}")