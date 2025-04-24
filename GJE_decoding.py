# -*- coding: utf-8 -*-
"""
Created on Apr 18, 2025
@author: Eda Aşçıoğlu
LT Kodlama + Belief Propagation Simülasyonu (Tez için Geliştirilmiş Versiyon)
"""

import random
import numpy as np
from source import DawnSimVis

# Simülasyon parametreleri
k = 1000  # Orijinal veri boyutu
n = 1200  # Kodlanmış sembol sayısı (overhead 1.2 için daha uygun)
SOURCE = 0  # Kaynak düğüm
REQUESTING_NODE = 1  # Talep eden düğüm
PACKET_LOSS_RATE = 0.05  # Paket kaybı oranı (gerçekçi kanal modeli)

# Orijinal veri
original_data = np.random.randint(0, 2, k, dtype=np.uint8)

###########################################################
def soliton_distribution(k, delta=0.05, c=0.1):
    """Robust Soliton dagilimi."""
    R = c * np.sqrt(k) * np.log(k / delta)
    rho = [1/k if d == 1 else 1/(d * (d-1)) for d in range(1, k+1)]
    tau = [R/(d * k) if d < int(k/R) else (R * np.log(R/delta)/k if d == int(k/R) else 0) for d in range(1, k+1)]
    probabilities = [rho[i-1] + tau[i-1] for i in range(1, k+1)]
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # Normalizasyon
    return probabilities

probabilities = soliton_distribution(k)

def encode_lt_from_root(data, n, probabilities, systematic_ratio=0.1):
    """LT kodlama (sistematik bağlantılar ile)."""
    nodes = []
    systematic_count = int(n * systematic_ratio)  # Sistematik sembol sayısı
    for i in range(n):
        if i < systematic_count:  # Sistematik semboller
            degree = 1
            selected_indices = [i % k]
        else:
            if random.random() < 0.3:  # %30 olasılıkla derece-1 sembol
                degree = 1
            else:
                degree = np.random.choice(np.arange(1, k + 1), p=probabilities)
            selected_indices = random.sample(range(k), min(degree, k))
        encoded_symbol = data[selected_indices[0]]
        for idx in selected_indices[1:]:
            encoded_symbol ^= data[idx]
        nodes.append((selected_indices, encoded_symbol))
    return nodes

def gauss_jordan_elimination(matrix, values):
    """Gauss-Jordan Eliminasyonu ile kalan sembolleri çöz (Aşırı belirlenmiş sistem için)."""
    try:
        if matrix.shape[0] < matrix.shape[1]:
            print("Denklem sayısı değişken sayısından az, GJE başarısız olabilir.")
            return None

        # En küçük kareler yöntemi ile çöz
        solution, residuals, rank, s = np.linalg.lstsq(matrix, values, rcond=None)
        
        # Çözümü ikili hale getir (0 veya 1)
        solution = np.round(solution).astype(np.int8) % 2
        return solution
    except np.linalg.LinAlgError:
        print("GJE çözümü başarısız: Lineer bağımlılık veya başka bir hata.")
        return None

def decode_lt_distributed(requesting_node_id, nodes, k, incremental=False):
    """BP dekodlama (GJE destekli)."""
    received_data = [(list(indices), value) for indices, value in nodes]
    known_values = np.full(k, -1, dtype=np.int8)
    symbol_queue = []

    # Derece-1 sembolleri bul
    for node_id, (indices, value) in enumerate(received_data):
        if len(indices) == 1:
            symbol_queue.append((node_id, indices[0], value))

    new_degree_one_symbols = []
    ripple_size_history = []

    # BP dekodlama
    while symbol_queue:
        node_id, symbol_idx, value = symbol_queue.pop(0)

        if known_values[symbol_idx] != -1:
            continue

        known_values[symbol_idx] = value

        for other_node_id, (indices, val) in enumerate(received_data):
            if indices is None or symbol_idx not in indices:
                continue

            new_indices = [i for i in indices if i != symbol_idx]
            new_val = val ^ value
            received_data[other_node_id] = (new_indices, new_val)

            if len(new_indices) == 1 and known_values[new_indices[0]] == -1:
                symbol_queue.append((other_node_id, new_indices[0], new_val))
                new_degree_one_symbols.append((other_node_id, new_indices[0], new_val))
        ripple_size_history.append(len(symbol_queue))

    # BP dekodlama durduysa GJE ile devam et
    unsolved_indices = np.where(known_values == -1)[0]
    if len(unsolved_indices) > 0:
        print(f"BP dekodlama durdu. Kalan {len(unsolved_indices)} sembol için GJE uygulanıyor...")
        valid_equations = []
        valid_values = []
        for indices, val in received_data:
            if indices is None:
                continue
            relevant_indices = [idx for idx in indices if idx in unsolved_indices]
            if relevant_indices:
                equation = np.zeros(len(unsolved_indices), dtype=np.int8)
                for idx in relevant_indices:
                    equation[np.where(unsolved_indices == idx)[0]] = 1
                valid_equations.append(equation)
                valid_values.append(val)

        if valid_equations:
            matrix = np.array(valid_equations)
            values = np.array(valid_values)
            print(f"GJE matrisi boyutu: {matrix.shape}, Değerler boyutu: {values.shape}")
            solution = gauss_jordan_elimination(matrix, values)
            if solution is not None:
                for i, idx in enumerate(unsolved_indices):
                    if i < len(solution):
                        known_values[idx] = solution[i]
            else:
                print("GJE çözümü başarısız oldu.")
        else:
            print("Çözülemeyen sembollere katkıda bulunan denklem bulunamadı.")

    # Hata analizi
    unsolved = np.sum(known_values == -1)
    if unsolved > 0:
        print(f"Çözülemeyen sembol sayısı: {unsolved}. Dalga tükenmiş olabilir.")
    print(f"Dalga boyutu geçmişi: {ripple_size_history}")

    return known_values, new_degree_one_symbols

###########################################################
print("Root (0) düğümden kodlama ve dagitim yapiliyor...")
nodes = encode_lt_from_root(original_data, n, probabilities)
print("\nEncode edilmis veriler:")
for i, (indices, value) in enumerate(nodes):
    print(f"Dugum {i} = XOR({indices}) -> {value}")
requesting_node_id = 5
recovered_data, new_symbols = decode_lt_distributed(requesting_node_id, nodes, k)

print("Başarı oranı:", np.mean(recovered_data == original_data))
print("Çözülemeyen sembol sayısı:", np.sum(recovered_data == -1))
if np.array_equal(original_data, recovered_data):
    print("Tüm veri başarıyla çözüldü!")
else:
    print("Bazı semboller çözülemedi.")

###########################################################
class Node(DawnSimVis.BaseNode):
    def init(self):
        self.received_packets = []
        self.known_values = np.full(k, -1, dtype=np.int8)
        self.decoded = False
        self.sent_packet = False
        self.probabilities = probabilities
        self.data = original_data if self.id == SOURCE else None
        self.nodes = []
        self.forwarded_packets = set()
        self.is_green = False
        self.simulation_ended = False
        self.decode_attempts = 0
        self.max_decode_attempts = 10
        self.start_time = None
        if self.id == SOURCE:
            self.change_color(1, 0, 0)
        else:
            self.change_color(0.5, 0.5, 0.5)

    def run(self):
        if self.id == SOURCE:
            self.nodes = encode_lt_from_root(self.data, n, self.probabilities)
            self.encode_and_send()
        if self.id != SOURCE:
            self.set_timer(random.uniform(1, 5), self.forward_packets)

    def encode_and_send(self):
        if self.simulation_ended or not self.nodes:
            return
        node_idx = random.randint(0, len(self.nodes) - 1)
        indices, value = self.nodes[node_idx]
        pck = {'indices': indices, 'value': value}
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        self.log(f'Sent LT-coded symbol: XOR({indices[:5]}...) -> {value}')
        self.set_timer(random.uniform(1, 2), self.encode_and_send)

    def on_receive(self, pck):
        if self.simulation_ended:
            return
        if random.random() < PACKET_LOSS_RATE:
            self.log(f'Packet dropped due to channel loss: XOR({pck["indices"][:5]}...) -> {pck["value"]}')
            return

        indices = pck['indices']
        value = pck['value']
        packet_key = (tuple(indices), value)
        if packet_key not in self.forwarded_packets:
            self.received_packets.append((indices, value))
            self.log(f'Received packet: XOR({indices[:5]}...) -> {value}')
            if len(indices) == 1:
                self.log(f'Node {self.id} received degree-1 symbol: XOR({indices}) -> {value}')
            if len(indices) == 1 and not self.is_green and self.id != SOURCE:
                self.change_color(0, 1, 0)
                self.is_green = True
                self.log(f'Node {self.id} turned green due to degree-1 symbol: XOR({indices}) -> {value}')
            if not self.is_green and self.id != SOURCE:
                self.change_color(0.7, 0.7, 0)

            if self.start_time is None:
                self.start_time = self.get_time()
            self.known_values, new_symbols = decode_lt_distributed(self.id, self.received_packets, k, incremental=True)
            success_rate = np.mean(self.known_values == original_data)
            for node_id, symbol_idx, value in new_symbols:
                if not self.is_green and self.id != SOURCE:
                    self.change_color(0, 1, 0)
                    self.is_green = True
                    self.log(f'Node {self.id} turned green due to new degree-1 symbol: v{symbol_idx} = {value}')
            if success_rate >= 0.95 or np.array_equal(self.known_values, original_data):
                self.decoded = True
                self.simulation_ended = True
                self.change_color(0, 0, 1)
                decode_time = self.get_time() - self.start_time
                self.log(f'Decoding successful on-the-fly! Success rate: {success_rate:.2f}, Decode time: {decode_time:.2f} seconds')

    def forward_packets(self):
        if self.simulation_ended or self.id == SOURCE or not self.received_packets:
            return
        pck = random.choice(self.received_packets)
        packet_key = (tuple(pck[0]), pck[1])
        if packet_key not in self.forwarded_packets:
            self.send(DawnSimVis.BROADCAST_ADDR, {'indices': pck[0], 'value': pck[1]})
            self.log(f'Forwarded packet: XOR({pck[0][:5]}...) -> {pck[1]}')
            self.forwarded_packets.add(packet_key)
        self.set_timer(random.uniform(0.5, 2), self.forward_packets)

    def finish(self):
        if self.id == REQUESTING_NODE:
            if self.decoded:
                self.log(f'Simulation successfully completed. Success rate: {np.mean(self.known_values == original_data):.2f}')
            else:
                self.log(f'Simulation completed, but not all data decoded. Unsolved symbols: {np.sum(self.known_values == -1)}')
        elif self.received_packets:
            self.log(f'Node {self.id} received {len(self.received_packets)} packets.')

###########################################################
def create_network():
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=100)

# Simülasyon ayarları
sim = DawnSimVis.Simulator(
    duration=2000,
    timescale=1,
    visual=True,
    terrain_size=(650, 650),
    title='LT Kodlama - BP Çözümleme (Geliştirilmiş)'
)

create_network()
sim.run()