import sys
sys.path.insert(1, '.')
from source import DawnSimVis
import random
import numpy as np
from collections import defaultdict
from scipy.special import expit

SOURCE = 0
k = 1000  # Orijinal veri boyutu
n = 2500  # Düğüm sayısı

# Veri oluşturma
data = np.random.randint(0, 2, k, dtype=np.uint8)

# Robust Soliton dağılımı
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

# Encoding işlemi
def encode_lt(data, probabilities):
    degree = np.random.choice(np.arange(1, k + 1), p=probabilities)
    selected_indices = random.sample(range(k), min(degree, k))
    encoded_symbol = data[selected_indices[0]]
    for i in selected_indices[1:]:
        encoded_symbol ^= data[i]
    return selected_indices, encoded_symbol

# Belief Propagation Decoder
def belief_propagation_decode(received_packets, k, max_iter=100):
    print("\nBP decode işlemi başlatılıyor...")
    variable_nodes = defaultdict(list)
    check_nodes = []

    for indices, value in received_packets:
        check_nodes.append((indices, value))
        for idx in indices:
            variable_nodes[idx].append(len(check_nodes) - 1)

    beliefs = np.zeros(k)
    messages = {}

    for iteration in range(max_iter):
        updated = False

        # Check nodes -> Variable nodes
        for check_id, (indices, value) in enumerate(check_nodes):
            for target in indices:
                msg = value
                for other in indices:
                    if other != target and beliefs[other] != 0:
                        msg ^= int(beliefs[other] > 0)
                messages[(check_id, target)] = 1 if msg == 1 else -1

        # Variable nodes -> Update beliefs
        for var_id in range(k):
            msg_sum = sum(messages.get((check_id, var_id), 0) for check_id in variable_nodes[var_id])
            new_belief = msg_sum
            if abs(new_belief - beliefs[var_id]) > 1e-6:
                updated = True
            beliefs[var_id] = new_belief

        if not updated:
            print(f"İterasyon {iteration + 1}'de güncelleme durdu.")
            break

    decoded = np.where(beliefs > 0, 1, 0).astype(np.uint8)
    return decoded

# Node sınıfı
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
            self.change_color(1, 0, 0)
            probabilities = robust_soliton_distribution(k)
            self.selected_indices, self.encoded_symbol = encode_lt(data, probabilities)
            self.log(f'Node {self.id} encoded: value = {self.selected_indices}')
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
                self.change_color(0, 0, 1)
                self.log(f'Node {self.id} paketi {self.parent}. düğümden aldı: indices = {self.selected_indices}')

                if self.parent is not None and self.parent < len(self.sim.nodes):
                    parent_node = self.sim.nodes[self.parent]
                    if parent_node:
                        parent_node.children.append(self.id)

                probabilities = robust_soliton_distribution(k)
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
        if self.parent is not None:
            self.log(f'Node {self.id} -> parent = {self.parent}')
        if self.children:
            self.log(f'Node {self.id} -> Children: {self.children}')
        if self.received_packets:
            self.log(f'Node {self.id} -> Received Packets: {[(sender, indices[0]) for sender, indices in self.received_packets]}')

# Ağ oluşturma
def create_network():
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=100)

# Simülatör başlat
sim = DawnSimVis.Simulator(
    duration=50,
    timescale=1,
    visual=True,
    terrain_size=(650, 650),
    title='Belief Propagation Decoding'
)

create_network()
sim.run()

# Decode sonrası
all_received_packets = []
for node in sim.nodes:
    if node.id != SOURCE:
        for sender_id, (indices, value) in node.received_packets:
            all_received_packets.append((indices, value))

decoded_data = belief_propagation_decode(all_received_packets, k)

# Kontrol
if np.array_equal(decoded_data, data):
    print("\nBaşarıyla çözüldü!")
else:
    print("\nVeri tam çözülemedi.")
    unsolved_bits = np.where(decoded_data != data)[0]
    print(f"Çözülemeyen bitler (indeksler): {unsolved_bits}")
    print(f"Çözülemeyen bit sayısı: {len(unsolved_bits)}")
