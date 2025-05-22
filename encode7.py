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

def decode_lt(received_packets, k, data):
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

    degree_1_indices = [indices[0] for indices, _ in received_packets if len(indices) == 1]
    if degree_1_indices:
        print(f"Başlangıçta derecesi 1 olan semboller: {sorted(degree_1_indices)}")

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
    print(f"Kullanılan (faydalı) paket sayısı: {len(used_packet_ids)}")
    print(f"Çözülebilen sembol sayısı: {k - len(undecoded)}")
    print(f"Çözülemeyen sembol sayısı: {len(undecoded)}")
    if undecoded:
        print(f"Çözülemeyen sembol indeksleri: {undecoded[:10]} ...")
    print(f"TÜM SEMBOLLERDEN COZUM: {correct_bits}/{k} bit doğru çözüldü (%{success_rate:.2f} başarı oranı)")
    print("=====================================================\n")
    decode_time = time.time() - start_time
    msg_complexity = n * np.log2(n)

    import os
    csv_file = "decode_results.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write("k,n,message_count,success_rate,decode_time,msg_complexity\n")

    with open(csv_file, "a") as f:
        f.write(f"{k},{n},{message_count},{success_rate:.2f},{decode_time:.2f},{msg_complexity:.2f}\n")

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
        self.neighbors = [node.id for node in self.sim.nodes if node.id != self.id and self.distance_to(node) <= self.tx_range]
        self.ack_sent = False
        self.log(f"Node {self.id} initialized with neighbors: {self.neighbors}")

    def distance_to(self, other):
        x1, y1 = self.pos
        x2, y2 = other.pos
        return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

    def run(self):
        global message_count
        if self.id == SOURCE:
            self.change_color(1, 0, 0)
            self.currstate = 'XPLORING'
            self.sent_probes = True
            self.log(f"Root node {self.id} sent probe message.")
            for node_id in self.neighbors:
                self.send(node_id, {'type': 'probe', 'sender': self.id})
                message_count += 1
            self.set_timer(20, self.force_start_encoding)

    def force_start_encoding(self):
        if self.currstate != 'TERM':
            self.currstate = 'TERM'
            self.start_encoding()

    def start_encoding(self):
        global message_count
        probabilities = robust_soliton_distribution(k)
        selected_indices, encoded_symbol = encode_lt(data, probabilities)
        self.received_packets.append((selected_indices.copy(), encoded_symbol))
        self.log(f"Root node {self.id} saved a symbol for itself with indices: {selected_indices}")

        pck = {
            'type': 'fwd_encoded',
            'sender': self.id,
            'indices': selected_indices,
            'symbol': encoded_symbol,
        }
        self.forward_packet(pck)
        self.set_timer(100, self.start_decoding)

    def forward_packet(self, pck):
        for child_id in self.children:
            self.send(child_id, pck)

    def start_decoding(self):
        self.log("Root node is starting global decoding...")
        decode_lt(self.received_packets, k, data)

    def on_receive(self, pck):
        global message_count
        sender_id = pck['sender']
        msg_type = pck['type']
        message_count += 1

        if self.currstate == 'IDLE' and msg_type == 'probe':
            self.parent = sender_id
            self.currstate = 'XPLORING'
            self.change_color(0, 0, 1)
            for neighbor in self.neighbors:
                if neighbor != self.parent:
                    self.send(neighbor, {'type': 'probe', 'sender': self.id})
                    message_count += 1
            self.set_timer(5, self.force_ack_if_needed)

        elif self.currstate == 'XPLORING':
            if msg_type == 'probe' and sender_id != self.parent:
                self.send(sender_id, {'type': 'reject', 'sender': self.id})
            elif msg_type == 'ack':
                self.received_ack.add(sender_id)
                if sender_id not in self.children:
                    self.children.append(sender_id)
            elif msg_type == 'reject':
                self.received_reject.add(sender_id)
                self.others.add(sender_id)

        elif msg_type == 'fwd_encoded':
            msg_key = (tuple(pck['indices']), pck['symbol'])
            if msg_key in self.collected_messages:
                return
            self.collected_messages.add(msg_key)
            self.received_packets.append((pck['indices'], pck['symbol']))
            self.log(f"Node {self.id} received fwd_encoded from {sender_id}")
            for child_id in self.children:
                self.set_timer(0.5, self.send, child_id, pck)
            if self.parent is not None:
                back_packet = {
                    'type': 'back_encoded',
                    'sender': self.id,
                    'indices': pck['indices'],
                    'symbol': pck['symbol']
                }
                self.set_timer(0.5, self.send, self.parent, back_packet)

        elif msg_type == 'back_encoded':
            if self.id == SOURCE:
                self.received_packets.append((pck['indices'], pck['symbol']))
                self.log(f"Root received back_encoded from node {sender_id}")

    def force_ack_if_needed(self):
        if not self.ack_sent and self.parent is not None:
            self.send(self.parent, {'type': 'ack', 'sender': self.id})
            self.currstate = 'TERM'
            self.change_color(0, 1, 0)
            self.ack_sent = True
            self.log(f"Node {self.id} forced to send ACK to parent {self.parent}")

    def finish(self):
        pass

def create_network():
    rows = int(np.sqrt(n))
    cols = (n + rows - 1) // rows
    spacing = 60
    for x in range(cols):
        for y in range(rows):
            if (x * rows + y) < n:
                px = 50 + x * spacing + random.uniform(-20, 20)
                py = 50 + y * spacing + random.uniform(-20, 20)
                sim.add_node(Node, pos=(px, py), tx_range=75)

sim = DawnSimVis.Simulator(
    duration=500,
    timescale=1,
    visual=True,
    terrain_size=(650, 650),
    title='Belief Propagation Decoding'
)

create_network()
sim.run()
