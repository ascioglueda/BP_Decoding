import random
import numpy as np
from source import DawnSimVis

# Simülasyon parametreleri
k = 1000  # Orijinal veri boyutu
n = 2500  # Kodlanmış sembol sayısı
SOURCE = 0  # Kaynak düğüm
REQUESTING_NODE = 1  # Talep eden düğüm

# Orijinal veri
original_data = np.random.randint(0, 2, k, dtype=np.uint8)

###########################################################
def soliton_distribution(k, delta=0.5):
    """Robust Soliton dagilimi."""
    R = k / sum(1.0 / i for i in range(1, k + 1))
    probabilities = [R / i for i in range(1, k + 1)]
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # Normalizasyon
    return probabilities

probabilities = soliton_distribution(k)

def encode_lt_from_root(data, n, probabilities):
    nodes = []
    for _ in range(n):
        degree = np.random.choice(np.arange(1, k + 1), p=probabilities)
        selected_indices = random.sample(range(k), min(degree, k))
        encoded_symbol = data[selected_indices[0]]
        for idx in selected_indices[1:]:
            encoded_symbol ^= data[idx]  # XOR işlemi
        nodes.append((selected_indices, encoded_symbol))
    return nodes

def decode_lt_distributed(requesting_node_id, nodes, k):
    print(f"\ndugum {requesting_node_id} decode icin request mesaji gonderiyor...")
    received_data = [(list(indices), value) for indices, value in nodes]  # Deep copy
    known_values = np.full(k, -1, dtype=np.int8)
    symbol_queue = []

    print("\nDerecesi 1 olan dugumler:")
    for node_id, (indices, value) in enumerate(received_data):
        if len(indices) == 1:
            print(f"dugum {node_id}: XOR({indices}) -> {value}")
            symbol_queue.append((node_id, indices[0], value))

    new_degree_one_symbols = []

    while symbol_queue:
        node_id, symbol_idx, value = symbol_queue.pop(0)

        if known_values[symbol_idx] != -1:
            continue  # Zaten biliniyorsa atla

        known_values[symbol_idx] = value

        for other_node_id, (indices, val) in enumerate(received_data):
            if indices is None or symbol_idx not in indices:
                continue

            # Yeni liste oluştur, orijinali değiştirme
            new_indices = [i for i in indices if i != symbol_idx]
            new_val = val ^ value
            received_data[other_node_id] = (new_indices, new_val)

            if len(new_indices) == 1 and known_values[new_indices[0]] == -1:
                print(f"    Yeni derecesi 1: Düğüm {other_node_id} -> v{new_indices[0]} = {new_val}")
                symbol_queue.append((other_node_id, new_indices[0], new_val))
                new_degree_one_symbols.append((other_node_id, new_indices[0], new_val))
            elif len(new_indices) == 0:
                received_data[other_node_id] = (None, None)

    return known_values, new_degree_one_symbols

###########################################################
print("Root (0) düğümden kodlama ve dağıtım yapılıyor...")
nodes = encode_lt_from_root(original_data, n, probabilities)
# Encode edilmiş verilerin tamamını yazdırmak yerine, sadece bilgi verelim
print(f"\n{n} adet sembol encode edildi. Örnek bir sembol: Düğüm 0 = XOR({nodes[0][0]}) -> {nodes[0][1]}")

requesting_node_id = 5
recovered_data, new_symbols = decode_lt_distributed(requesting_node_id, nodes, k)

# Çözülen ve çözülemeyen sembolleri yazdır
print("\nÇözülen Semboller (Indeks ve Değer):")
solved_indices = np.where(recovered_data != -1)[0]
for idx in solved_indices:
    print(f"Sembol {idx}: {recovered_data[idx]} (Orijinal: {original_data[idx]})")

print("\nÇözülemeyen Semboller (Indeksler):")
unsolved_indices = np.where(recovered_data == -1)[0]
if len(unsolved_indices) > 0:
    print(f"Toplam {len(unsolved_indices)} sembol çözülemedi: {unsolved_indices}")
else:
    print("Tüm semboller çözüldü!")

###########################################################
class Node(DawnSimVis.BaseNode):
    def init(self):
        self.received_packets = []
        self.known_values = np.full(k, -1, dtype=np.int8)  # Tüm düğümler dekodlama yapabilir
        self.decoded = False
        self.sent_packet = False
        self.probabilities = probabilities
        self.data = original_data if self.id == SOURCE else None
        self.nodes = []  # encode_lt_from_root için sembol listesi
        self.forwarded_packets = set()  # Aynı paketin tekrar gönderilmesini önlemek için
        self.is_green = False  # Düğümün yeşil olup olmadığını takip etmek için
        # Renk ayarları
        if self.id == SOURCE:
            self.change_color(1, 0, 0)  # Kırmızı
        else:
            self.change_color(0.5, 0.5, 0.5)  # Gri (başlangıçta)

    def run(self):
        if self.id == SOURCE:
            # Kaynak düğüm, encode_lt_from_root ile sembolleri hazırlar
            self.nodes = encode_lt_from_root(self.data, n, self.probabilities)
            self.encode_and_send()
        # Tüm düğümler dekodlama yapabilir
        self.set_timer(10, self.decode)
        # Diğer düğümler mesajları iletmeye hazır
        if self.id != SOURCE:
            self.set_timer(random.uniform(1, 5), self.forward_packets)

    def encode_and_send(self):
        if self.nodes:
            node_idx = random.randint(0, len(self.nodes) - 1)
            indices, value = self.nodes[node_idx]
            pck = {'indices': indices, 'value': value}
            self.send(DawnSimVis.BROADCAST_ADDR, pck)
            self.log(f'Sent LT-coded symbol: XOR({indices[:5]}...) -> {value}')
            self.set_timer(random.uniform(1, 2), self.encode_and_send)

    def on_receive(self, pck):
        indices = pck['indices']
        value = pck['value']
        # Tüm düğümler paketleri saklar
        packet_key = (tuple(indices), value)  # Aynı paketi tekrar eklememek için
        if packet_key not in self.forwarded_packets:
            self.received_packets.append((indices, value))
            self.log(f'Received packet: XOR({indices[:5]}...) -> {value}')
            # Derecesi-1 sembol alındığında yeşile dön
            if len(indices) == 1 and not self.is_green and self.id != SOURCE:
                self.change_color(0, 1, 0)  # Yeşil
                self.is_green = True
                self.log(f'Node {self.id} turned green due to degree-1 symbol: XOR({indices}) -> {value}')
            # Mesaj alındığında sarıya dön (eğer yeşil değilse)
            if not self.is_green and self.id != SOURCE:
                self.change_color(0.7, 0.7, 0)  # Sarı

    def forward_packets(self):
        if self.id != SOURCE and self.received_packets:
            pck = random.choice(self.received_packets)
            packet_key = (tuple(pck[0]), pck[1])
            if packet_key not in self.forwarded_packets:
                self.send(DawnSimVis.BROADCAST_ADDR, {'indices': pck[0], 'value': pck[1]})
                self.log(f'Forwarded packet: XOR({pck[0][:5]}...) -> {pck[1]}')
                self.forwarded_packets.add(packet_key)
        self.set_timer(random.uniform(0.5, 2), self.forward_packets)

    def decode(self):
        if self.decoded:
            return

        self.log('Starting belief propagation decoding...')

        # decode_lt_distributed ile dekodlama, yeni derecesi-1 sembolleri de döndürür
        self.known_values, new_degree_one_symbols = decode_lt_distributed(self.id, self.received_packets, k)

        # Yeni derecesi-1 semboller bulunduğunda yeşile dön (eğer daha önce yeşile dönmediyse)
        for node_id, symbol_idx, value in new_degree_one_symbols:
            if not self.is_green and self.id != SOURCE:
                self.change_color(0, 1, 0)  # Yeşil
                self.is_green = True
                self.log(f'Node {self.id} turned green due to new degree-1 symbol: v{symbol_idx} = {value}')

        # Başarı oranı ve çözülemeyen sembol sayısı
        success_rate = np.mean(self.known_values == original_data)
        unsolved = np.sum(self.known_values == -1)
        self.log(f'Decoding complete. Success rate: {success_rate:.2f}, Unsolved: {unsolved}')

        if np.array_equal(self.known_values, original_data):
            self.log('All data successfully decoded!')
            self.decoded = True
            if self.id == REQUESTING_NODE:
                self.log('Simülasyon basarili bir sekilde tamamlandi!')
                sim.stop()
        else:
            self.log('Retrying decoding...')
            self.set_timer(2, self.decode)

    def finish(self):
        if self.id == REQUESTING_NODE:
            if self.decoded:
                self.log(f'Simülasyon basariyla tamamlandi. Basari orani: {np.mean(self.known_values == original_data):.2f}')
            else:
                self.log(f'Simülasyon tamamlandi, ancak tum veri cozulemedi. Cozulemeyen semboller: {np.sum(self.known_values == -1)}')
        elif self.received_packets:
            self.log(f'Dugum {self.id} toplam {len(self.received_packets)} paket aldi.')

###########################################################
def create_network():
    # Place nodes in a 10x10 grid
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Simülasyon ayarları
sim = DawnSimVis.Simulator(
    duration=2000,
    timescale=1,
    visual=True,
    terrain_size=(650, 650),
    title='LT Kodlama - BP Çözümleme'
)

create_network()
sim.run()


