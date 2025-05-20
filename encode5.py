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
    if random.random() < 0.1:
        degree = 1
    else:
        degree = np.random.choice(np.arange(1, k + 1), p=distribution)
    selected_indices = random.sample(range(k), min(degree, k))
    encoded_symbol = data[selected_indices[0]]
    for i in selected_indices[1:]:
        encoded_symbol ^= data[i]
    return selected_indices, encoded_symbol

def decode_lt(received_packets, k, data):
    degree_1_indices = [indices[0] for indices, _ in received_packets if len(indices) == 1]
    if degree_1_indices:
        print(f"Başlangıçta derecesi 1 olan semboller: {sorted(degree_1_indices)}")
    known_values = np.full(k, -1, dtype=np.int8)
    symbol_queue = []
    equations = received_packets.copy()

    for node_id_, (indices, value) in enumerate(equations):
        if len(indices) == 1:
            symbol_queue.append((node_id_, indices[0], value))

    while symbol_queue:
        node_id_, index, value = symbol_queue.pop(0)
        if known_values[index] != -1:
            continue
        known_values[index] = value
        for other_node_id, (indices, val) in enumerate(equations):
            if other_node_id == node_id_:
                continue
            if index in indices:
                indices.remove(index)
                val ^= value
                equations[other_node_id] = (indices, val)
                if len(indices) == 1:
                    symbol_queue.append((other_node_id, indices[0], val))

    result = np.zeros(k, dtype=np.uint8)
    for i in range(k):
        result[i] = known_values[i] if known_values[i] != -1 else 0

    correct_bits = np.sum(result == data)
    success_rate = (correct_bits / k) * 100
    print(f"\nTÜM SEMBOLLERDEN COZUM: {correct_bits}/{k} bit doğru çözüldü (%{success_rate:.2f} başarı oranı)")
    print(f"Toplam mesaj: {message_count}")
    print(f"Toplam süre: {time.time() - start_time:.2f} saniye")
    print(f"Zaman karmaşıklığı tahmini: O({n})")
    print(f"Mesaj karmaşıklığı tahmini: O({n * np.log2(n):.2f})")
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
        self.log(f"Node {self.id} started in IDLE state.")

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
            for node in self.sim.nodes:
                if node.id != self.id and self.distance_to(node) <= self.tx_range:
                    link = (self.id, node.id)
                    if link not in self.sent_links:
                        self.send(node.id, {'type': 'probe', 'sender': self.id})
                        message_count += 1
                        self.sent_links.add(link)
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

    def start_decoding(self):
        self.log("Root node is starting global decoding...")
        decode_lt(self.received_packets, k, data)

    def send_encoded_packet(self, pck):
        global message_count
        target_node = pck['target']
        self.send(target_node, pck)
        message_count += 1
        self.log(f"Node {self.id} sent encoded packet to node {target_node} with indices: {pck['indices']}")

    def on_receive(self, pck):
        global message_count
        sender_id = pck['sender']
        msg_type = pck['type']
        message_count += 1
        self.log(f"Node {self.id} received '{msg_type}' packet from node {sender_id}.")

        if self.currstate == 'IDLE' and msg_type == 'probe':
            self.parent = sender_id
            self.currstate = 'XPLORING'
            self.change_color(0, 0, 1)
            self.log(f"Node {self.id} set {self.parent} as parent.")
            for node in self.sim.nodes:
                if node.id != self.id and node.id != self.parent and self.distance_to(node) <= self.tx_range:
                    link = (self.id, node.id)
                    if link not in self.sent_links:
                        self.send(node.id, {'type': 'probe', 'sender': self.id})
                        message_count += 1
                        self.sent_links.add(link)
            self.sent_probes = True

        elif self.currstate == 'XPLORING':
            if msg_type == 'probe' and sender_id != self.parent:
                self.send(sender_id, {'type': 'reject', 'sender': self.id})
                message_count += 1
            elif msg_type == 'ack':
                self.received_ack.add(sender_id)
                self.children.append(sender_id)
            elif msg_type == 'reject':
                self.received_reject.add(sender_id)
                self.others.add(sender_id)

            if self.sent_probes and (self.received_ack or self.received_reject):
                if self.id != SOURCE:
                    self.send(self.parent, {'type': 'ack', 'sender': self.id})
                    message_count += 1
                    self.change_color(0, 1, 0)
                    self.currstate = 'TERM'
                else:
                    self.collected_messages.add(sender_id)
                    if len(self.collected_messages) >= n - 1:
                        self.currstate = 'TERM'
                        self.set_timer(10, self.start_encoding)

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