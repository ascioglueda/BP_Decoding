import random
import sys
import numpy as np
import time
from source import DawnSimVis

SOURCE = 0
k = 1000
n = 100

message_count = 0
start_time = time.time()

data = np.random.randint(0, 2, k, dtype=np.uint8)

def robust_soliton_distribution(k, delta=0.05, c=0.2):
    R = c * np.log(k / delta) * np.sqrt(k)
    tau = np.zeros(k)
    for i in range(1, k):
        if i <= int(k / R) - 1:
            tau[i] = R / (i * k)
        elif i == int(k / R):
            tau[i] = R * np.log(R / delta) / k
    ideal = np.array([1 / k] + [1 / (i * (i - 1)) for i in range(2, k + 1)])
    distribution = ideal + tau
    distribution /= distribution.sum()
    return distribution

def encode_lt(data, distribution):
    if random.random() < 0.3:
        degree = 1
    else:
        degree = np.random.choice(np.arange(1, k + 1), p=distribution)
    selected_indices = random.sample(range(k), min(degree, k))
    encoded_symbol = data[selected_indices[0]]
    for i in selected_indices[1:]:
        encoded_symbol ^= data[i]
    return selected_indices, encoded_symbol

def decode_lt(received_packets, k, data, total_hops=0, avg_hops=0):
    equations = [(list(indices), val) for indices, val in received_packets]
    known_values = np.full(k, -1, dtype=np.int8)
    symbol_queue = []
    used_packet_ids = set()

    for i, (indices, val) in enumerate(equations):
        if len(indices) == 1:
            symbol_queue.append((i, indices[0], val))

    while symbol_queue:
        eq_id, index, value = symbol_queue.pop(0)
        if known_values[index] != -1:
            continue
        known_values[index] = value
        used_packet_ids.add(eq_id)

        new_queue = []
        for i, (indices, val) in enumerate(equations):
            if index in indices:
                indices.remove(index)
                val ^= value
                equations[i] = (indices, val)
                if len(indices) == 1:
                    new_queue.append((i, indices[0], val))

        symbol_queue.extend(new_queue)

    # Ekstra: başlangıçta derecesi 1 olanları yazdır
    degree_1_indices = [indices[0] for indices, _ in received_packets if len(indices) == 1]
    if degree_1_indices:
        print(f"Başlangıçta derecesi 1 olan semboller: {sorted(degree_1_indices)}")

    # Sonuçlar ve istatistik
    result = np.zeros(k, dtype=np.uint8)
    undecoded = []

    for i in range(k):
        if known_values[i] != -1:
            result[i] = known_values[i]
        else:
            undecoded.append(i)

    correct_bits = np.sum(result == data)
    success_rate = (correct_bits / k) * 100

    print("\n================== DECODING RAPORU ==================")
    print(f"Toplam root’a gelen paket sayısı: {len(received_packets)}")
    print(f"TÜM SEMBOLLERDEN COZUM: {correct_bits}/{k} bit doğru çözüldü (%{success_rate:.2f} başarı oranı)")
    print("=====================================================\n")
    decode_time = time.time() - start_time
    msg_complexity = n * np.log2(n)

    # Başlık satırı yoksa oluştur
    import os
    csv_file = "decode_results.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write("k,n,message_count,success_rate,decode_time,msg_complexity,total_hops,avg_hops\n")

    with open(csv_file, "a") as f:
        f.write(f"{k},{n},{message_count},{success_rate:.2f},{decode_time:.2f},{msg_complexity:.2f},{total_hops},{avg_hops:.2f}\n")

    return result

class Node(DawnSimVis.BaseNode):
    def init(self):
        self.currstate = 'IDLE'
        self.parent = None
        self.children = []
        self.others = set()
        self.sent_probes = False
        self.received_reject = set()
        self.received_ack = set()
        self.received_packets = []
        self.collected_messages = set()
        self.encode_count = 0
        self.sent_links = set()
        self.neighbors = []
        self.log(f"Node {self.id} started in IDLE state.")
        self.hop_count_to_root = None
        if self.id == SOURCE:
            self.total_hops_list = []
            
    # İki düğüm arasındaki mesafeyi hesaplar
    def distance_to(self, other):
        x1, y1 = self.pos
        x2, y2 = other.pos
        return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

    # Hop sayısı hesaplama fonksiyonu
    def calculate_hop_count(self):
        # Hop count hesapla
        hop_count = 0
        node = self
        while node.parent is not None:
            hop_count += 1
            parent_id = node.parent
            parent_node = next((n for n in self.sim.nodes if n.id == parent_id), None)
            if parent_node is None:
                break
            node = parent_node

        self.hop_count_to_root = hop_count
        self.log(f"Node {self.id} has hop count {hop_count}")  

    # Her düğümün başlangıçta yapması gereken işlemleri (örn. probe gönderme) başlatır
    def run(self):
        global message_count
        if self.id == SOURCE:
            self.change_color(1, 0, 0)
            self.currstate = 'XPLORING'
            self.sent_probes = True
            self.log(f"Root node {self.id} sent probe message.")
            for node in self.sim.nodes:
                if node.id != self.id and self.distance_to(node) <= self.tx_range:
                    link = (self.id, node.id)
                    if link not in self.sent_links:
                        self.send(node.id, {'type': 'probe', 'sender': self.id})
                        message_count += 1
                        self.sent_links.add(link)
            self.set_timer(20, self.force_start_encoding)

    # Eğer zamanlayıcı tamamlanırsa ve hala encoding yapılmadıysa başlatır
    def force_start_encoding(self):
        if self.currstate != 'TERM':
            self.currstate = 'TERM'
            self.start_encoding()

    # Root node için LT encode işlemlerini başlatır ve her düğüme paket gönderir
    def start_encoding(self):
        global message_count
        probabilities = robust_soliton_distribution(k)

        selected_indices, encoded_symbol = encode_lt(data, probabilities)
        self.received_packets.append((selected_indices.copy(), encoded_symbol))
        self.log(f"Root node {self.id} saved a symbol for itself with indices: {selected_indices}")

        for target_node in range(1, n):
            selected_indices, encoded_symbol = encode_lt(data, probabilities)
            self.encode_count += 1
            pck = {
                'type': 'encoded',
                'sender': self.id,
                'indices': selected_indices,
                'symbol': encoded_symbol,
                'target': target_node
            }
            self.received_packets.append((selected_indices.copy(), encoded_symbol))
            self.set_timer(2 + self.encode_count * 0.01, self.send_encoded_packet, pck)

        total_encoding_time = 2 + (n - 1) * 0.01
        self.set_timer(total_encoding_time + 10, self.start_decoding)

    # Root node'da decoding işlemini başlatır ve hop istatistiklerini hesaplar
    def start_decoding(self):
        self.log("Root node is starting global decoding...")

        if self.id == SOURCE:
            hop_values = [n.hop_count_to_root for n in self.sim.nodes if n.hop_count_to_root is not None]
            total_hops = sum(hop_values)
            avg_hops = total_hops / len(hop_values) if hop_values else 0

            print(f"Root’a gelen paketlerin toplam hop sayısı: {total_hops}")
            #print(f"Ortalama hop sayısı: {avg_hops:.2f}")

            # HOP bilgilerini gönder
            decode_lt(self.received_packets, k, data, total_hops, avg_hops)

    # Zamanlayıcı ile belirli bir hedef düğüme LT paket gönderir
    def send_encoded_packet(self, pck):
        global message_count
        target_node = pck['target']
        self.send(target_node, pck)
        message_count += 1
        self.log(f"Node {self.id} sent encoded packet to node {target_node} with indices: {pck['indices']}")

    # Gelen her paketi işler: probe, ack, reject veya encoded türüne göre davranır
    def on_receive(self, pck):
        global message_count
        sender_id = pck['sender']
        msg_type = pck['type']
        message_count += 1

        if self.neighbors == []:
            self.neighbors = [node.id for node in self.sim.nodes if node.id != self.id and self.distance_to(node) <= self.tx_range]

        if self.currstate == 'IDLE' and msg_type == 'probe':
            self.parent = sender_id
            self.currstate = 'XPLORING'
            self.change_color(0, 0, 1)
            for neighbor in self.neighbors:
                if neighbor != self.parent:
                    self.send(neighbor, {'type': 'probe', 'sender': self.id})
                    message_count += 1
            self.sent_probes = True
            return

        if self.currstate == 'XPLORING':
            if msg_type == 'probe' and sender_id != self.parent:
                self.send(sender_id, {'type': 'reject', 'sender': self.id})
                message_count += 1
            elif msg_type == 'ack':
                self.received_ack.add(sender_id)
                if sender_id not in self.children:
                    self.children.append(sender_id)
            elif msg_type == 'reject':
                self.received_reject.add(sender_id)
                self.others.add(sender_id)

            expected = set(self.neighbors) - ({self.parent} if self.parent is not None else set())
            if self.sent_probes and (self.received_ack | self.received_reject) == expected:
                if self.id != SOURCE:
                    self.send(self.parent, {'type': 'ack', 'sender': self.id})
                    self.change_color(0, 1, 0)
                    self.currstate = 'TERM'
                    self.calculate_hop_count() 

                else:
                    self.collected_messages.update(self.received_ack | self.received_reject)
                    if len(self.collected_messages) >= n - 1:
                        self.currstate = 'TERM'
                        self.set_timer(10, self.start_encoding)

        elif msg_type == 'encoded':
            if self.id == SOURCE:
                if sender_id != self.id:
                    self.received_packets.append((pck['indices'], pck['symbol']))
                    self.log(f"Root received encoded packet back from node {sender_id}")
                    sender_node = next((n for n in self.sim.nodes if n.id == sender_id), None)
                    if sender_node and sender_node.hop_count_to_root is not None:
                        self.total_hops_list.append(sender_node.hop_count_to_root)
                        self.log(f"Root stored hop count {sender_node.hop_count_to_root} from node {sender_id}")  # <-- Bu satırı EKLE
            else:
                if self.parent is not None:
                    self.send(self.parent, pck)
                    self.log(f"Node {self.id} forwarded encoded packet to parent {self.parent}")

def create_network():
    rows = int(np.sqrt(n))
    cols = (n + rows - 1) // rows
    spacing = 60
    for x in range(cols):
        for y in range(rows):
            if (x * rows + y) < n:
                px = 50 + x * spacing + random.uniform(-20, 20)
                py = 50 + y * spacing + random.uniform(-20, 20)
                sim.add_node(Node, pos=(px, py), tx_range=100)

sim = DawnSimVis.Simulator(
    duration=500,
    timescale=1,
    visual=True,
    terrain_size=(650, 650),
    title='Belief Propagation Decoding'
)

create_network()
sim.run()