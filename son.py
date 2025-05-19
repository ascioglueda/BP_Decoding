import sys
sys.path.insert(1, '.')
from source import DawnSimVis
import random
import math
import numpy as np

TX_RANGE = 75
SOURCE = 0
n = 100
k = 1000

# Orijinal veri
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
    degree = np.random.choice(np.arange(1, k + 1), p=distribution)
    selected_indices = random.sample(range(k), min(degree, k))
    encoded_symbol = data[selected_indices[0]]
    for i in selected_indices[1:]:
        encoded_symbol ^= data[i]
    return selected_indices, encoded_symbol

distribution = robust_soliton_distribution(k)

class Node(DawnSimVis.BaseNode):

    def init(self):
        self.currstate = 'IDLE'
        self.parent = None
        self.childs = set()
        self.others = set()
        self.waiting_for = set()
        self.received_acks = set()
        self.neighbors = []
        self.started = False
        self.encoded_packets = []

    def find_neighbors(self):
        neighbors = []
        x1, y1 = self.pos
        for node in self.sim.nodes:
            if node.id == self.id:
                continue
            x2, y2 = node.pos
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if distance <= TX_RANGE:
                neighbors.append(node.id)
        return neighbors

    def run(self):
        self.set_timer(0.1, self.start_tree)

    def start_tree(self):
        self.neighbors = self.find_neighbors()
        if self.id == SOURCE:
            self.start_exploring()

    def start_exploring(self):
        if self.started:
            return
        self.started = True
        self.currstate = 'XPLORD'
        self.waiting_for = set(self.neighbors) - ({self.parent} if self.parent else set())
        for n in self.waiting_for:
            self.send(n, {'type': 'probe', 'sender': self.id})
        if self.id == SOURCE:
            self.change_color(1, 0, 0)

    def start_encoding(self):
        self.log("🔥 Encoding started...")
        for i in range(n):  # her düğüme 1 paket
            indices, symbol = encode_lt(data, distribution)
            self.encoded_packets.append((indices, symbol))
            self.log(f"[ENCODE] Packet {i}: indices={indices}, symbol={symbol}")
            for j in range(n):
                self.send(j, {
                    'type': 'encoded',
                    'indices': indices,
                    'symbol': symbol,
                    'packet_id': i
                })
        self.log(f"✅ Encoding finished. Total packets: {len(self.encoded_packets)}")

    def on_receive(self, pck):
        msg = pck['type']
        sender = pck['sender']

        if self.currstate == 'IDLE' and msg == 'probe':
            self.parent = sender
            self.start_exploring()
            return

        if self.currstate == 'XPLORD':
            if msg == 'probe':
                self.send(sender, {'type': 'reject', 'sender': self.id})
                self.change_color(0, 0, 1)
            elif msg == 'ack':
                self.childs.add(sender)
                self.received_acks.add(sender)
                self.change_color(0, 1, 0)
                self.start_exploring()
            elif msg == 'reject':
                self.others.add(sender)
                self.change_color(0, 0, 1)

            if self.received_acks.union(self.others) == self.waiting_for:
                if self.id != SOURCE and self.parent is not None:
                    self.send(self.parent, {'type': 'ack', 'sender': self.id})
                self.currstate = 'TERM'

                # ENCODING'i root başlatır
                if self.id == SOURCE:
                    self.set_timer(0.5, self.start_encoding)

        elif msg == 'encoded':
            self.log(f"[RECV] Encoded packet {pck['packet_id']} received from root")

    def finish(self):
        if self.currstate == 'IDLE':
            self.change_color(0.2, 0.2, 0.2)
            self.log("Disconnected node.")
        else:
            self.log(f"Parent: {self.parent}")
            self.log(f"Children: {sorted(self.childs)}")
            if self.id == SOURCE:
                self.log(f"📦 Total encoded packets: {len(self.encoded_packets)}")

###########################################################
def create_network():
    grid_size = math.ceil(math.sqrt(n))
    spacing = 60
    count = 0
    for x in range(grid_size):
        for y in range(grid_size):
            if count >= n:
                return
            px = 50 + x * spacing + random.uniform(-15, 15)
            py = 50 + y * spacing + random.uniform(-15, 15)
            sim.add_node(Node, pos=(px, py), tx_range=TX_RANGE)
            count += 1

sim = DawnSimVis.Simulator(
    duration=100,
    timescale=1,
    visual=True,
    terrain_size=(650, 650),
    title='Spanning Tree + LT Encoding'
)

create_network()
sim.run()
