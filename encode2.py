import sys
sys.path.insert(1, '.')
from source import DawnSimVis
import random
import numpy as np

SOURCE = 0
k = 1000  # Orijinal veri boyutu
n = 2500  # Düğüm sayısı

# Veri oluşturma (0 ve 1'lerden oluşan)
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
    for idx in selected_indices[1:]:
        encoded_symbol ^= data[idx]  # XOR işlemi
    return selected_indices, encoded_symbol

class Node(DawnSimVis.BaseNode):

    def __init__(self, simulator, node_id, pos, tx_range):
        super().__init__(simulator, node_id, pos, tx_range)
        self.msg_received = False #mesaj aldi mi almadi mi
        self.parent = None 
        self.visited = False #daha once ziyaret edildi mi
        self.selected_indices = None #encode edilen veriler
        self.encoded_symbol = None #hangi degerler xorlanmis

    def run(self):
        if self.id == SOURCE:
            self.change_color(1, 0, 0)  # Root kırmızı
            probabilities = soliton_distribution(k)
            self.selected_indices, self.encoded_symbol = encode_lt(data, probabilities)
            self.log(f'Node {self.id} encoded: value = {self.selected_indices}')
            pck = {'type': 'encoded', 'sender': self.id, 'indices': self.encoded_symbol}
            self.send(DawnSimVis.BROADCAST_ADDR, pck)
            self.msg_received = True
            self.visited = True

    def on_receive(self, pck):
        if pck['type'] == 'encoded' and not self.visited:
            self.parent = pck['sender']
            self.selected_indices = pck['indices']

            self.visited = True
            self.change_color(0, 0, 1)  # Mesaj alındığında mavi renk
            self.log(f'Node {self.id} paketi aldi: indices = {self.selected_indices}')

            # Yeni bir encode işlemi yap
            probabilities = soliton_distribution(k)
            new_indices = encode_lt(data, probabilities)
            self.encoded_symbol = new_indices
            
            # Mesajı 1 saniye gecikmeyle gönder
            new_pck = {'type': 'encoded', 'sender': self.id, 'indices': new_indices}
            self.set_timer(1, self.cb_msg_send, new_pck)

    def cb_msg_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        self.change_color(0, 1, 0)  # Mesaj gönderildikten sonra yeşil
        self.log(f'Mesaj gönderildi: {self.id}')

    def finish(self):
        pass

def create_network():
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Simulator nesnesi oluştur
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
