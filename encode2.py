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
    for i in selected_indices[1:]: #ilk veriyi aliyoruz
        encoded_symbol ^= data[i]  # XOR işlemi
    return selected_indices, encoded_symbol

def decode_lt_distributed(requesting_node_id, nodes, k):
    """Belief Propagation ile decode."""
    print(f"\nDüğüm {requesting_node_id} decode için request mesajı gönderiyor...")
    received_data = nodes.copy()  # Gelen veriler (düğümler)
    known_values = np.full(k, -1, dtype=np.int8)  # Bilinen değerleri tutacak
    symbol_queue = []  # Çözüme ulaşmış düğümleri tutan kuyruk

    # Received data içeriğini yazdır
    print("Received Data İçeriği:")
    for node_id, packet in enumerate(received_data):
        if packet is not None:
            indices, value = packet
            print(f"Düğüm {node_id}: indices = {indices}, value = {value}")

    # Derecesi 1 olanları bul
    print("\nDerecesi 1 olan düğümler:")
    degree_one_count = 0
    for node_id, (indices, value) in enumerate(received_data):
        if len(indices) == 1:  # Derecesi 1 olan düğümü bul
            degree_one_count += 1
            print(f"Düğüm {node_id}: XOR(indices={indices[0]}) = {value}")
            symbol_queue.append((node_id, indices[0], value))
    print(f"Toplam derecesi 1 olan düğüm sayısı: {degree_one_count}")

    iteration = 0
    while symbol_queue:
        iteration += 1
        node_id, symbol_idx, value = symbol_queue.pop(0)  # Kuyruktan bir düğüm çek
        if known_values[symbol_idx] == -1:  # Henüz çözülmemişse
            known_values[symbol_idx] = value  # Çözümü kaydet
        
        # Diğer düğümleri güncelle
        for other_node_id, packet in enumerate(received_data):
            if packet is None:  # Eğer veri yoksa atla
                continue
            indices, val = packet
            if symbol_idx in indices:  # Eğer bu sembol bu düğümde varsa
                indices.remove(symbol_idx)  # İndeksi kaldır
                val ^= value  # XOR işlemi yap
                received_data[other_node_id] = (indices, val)  # Yeni veri
                if len(indices) == 1:  # Eğer derecesi 1'e düştüyse
                    symbol_queue.append((other_node_id, indices[0], val))
                elif len(indices) == 0:  # Derece 0 olan düğüm silinir
                    received_data[other_node_id] = None

    # Çözümleme sonuçlarını rapor et
    unresolved_nodes = np.sum(known_values == -1)
    resolved_nodes = k - unresolved_nodes
    print(f"\nÇözümleme İstatistikleri:")
    print(f"Toplam veri sayısı: {k}")
    print(f"Çözülen veri sayısı: {resolved_nodes}")
    print(f"Çözülemeyen veri sayısı: {unresolved_nodes}")
    print(f"Çözümleme oranı: {resolved_nodes / k * 100:.2f}%")

    if unresolved_nodes > 0:
        print("Çözülemeyen düğümler var!")

    # Çözümleme sonuçlarını yazdırma
    print(f"Çözümlenen veriler (Node {requesting_node_id}): {known_values}")

    return known_values

class Node(DawnSimVis.BaseNode):

    def __init__(self, simulator, node_id, pos, tx_range):
        super().__init__(simulator, node_id, pos, tx_range)
        self.msg_received = False #mesaj alindi mi alinmadi mi
        self.parent = None
        self.children = [] 
        self.visited = False #dugum daha once ziyaret edildi mi
        self.selected_indices = None # k degerlerini tutar (orijinal veri) indekslerini tutar.[4, 15, 300]
        self.encoded_symbol = None #selected_indices hangi degerleri xorlanmis
        self.received_packets =[] #gelen tum encode verileri saklamak icin liste

    def run(self):
        if self.id == SOURCE:
            self.change_color(1, 0, 0)  # Root kırmızı
            probabilities = soliton_distribution(k)
            self.selected_indices, self.encoded_symbol = encode_lt(data, probabilities)
            self.log(f'Node {self.id} encoded: value = {self.selected_indices}')
            pck = {'type': 'encoded', 'sender': self.id, 'indices':(self.selected_indices, self.encoded_symbol)}
            self.send(DawnSimVis.BROADCAST_ADDR, pck)
            self.msg_received = True
            self.visited = True
        
    #self.id gonderen kim
    def on_receive(self, pck):
        if pck['type'] == 'encoded':
            # Gelen pakette hem göndereni hem veriyi kaydet
            if pck['type'] == 'encoded':
                self.received_packets.append((pck['sender'], pck['indices']))

            if not self.visited:
                self.parent = pck['sender']
                self.selected_indices, self.encoded_symbol =pck['indices']

                self.visited = True
                self.change_color(0, 0, 1)  # Mesaj alındığında mavi renk
                self.log(f'Node {self.id} paketi {self.parent}. dugumden aldi: indices = {self.selected_indices}')

                # Ebeveyne çocuk olarak kendini ekle
                parent_node = self.sim.nodes[self.parent]
                if parent_node:
                    parent_node.children.append(self.id)
                
                # Yeni bir encode işlemi yap
                probabilities = soliton_distribution(k)
                self.selected_indices, self.encoded_symbol = encode_lt(data, probabilities)

                self.set_timer(1, self.cb_msg_send, {
                    'type': 'encoded',
                    'sender': self.id,
                    'indices': (self.selected_indices, self.encoded_symbol)
                })
            #else:
                #self.log(f'Node {self.id} ekstra paket aldı: gönderen = {sender}, indices = {indices[0]}')
                
    def cb_msg_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        self.change_color(0, 1, 0)  # Mesaj gonderildikten sonra yesil
        self.log(f'Mesaj gönderildi: {self.id}')

    def finish(self):
        if self.parent is not None:
            self.log(f'Node {self.id} -> parent = {self.parent}')
        if self.children:
            self.log(f'Node {self.id} -> Children: {self.children}')
        if self.received_packets:
            self.log(f'Node {self.id} -> Received Packets: {[(sender, indices[0]) for sender, indices in self.received_packets]}')
        if self.id == SOURCE:
            # Simülasyonun sonunda, çözüm sonuçlarını yazdır
            decoded_data = decode_lt_distributed(self.id, [packet[1] for packet in self.received_packets], k)
            print(f'Final Decoded Data at Node {self.id}: {decoded_data}')

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