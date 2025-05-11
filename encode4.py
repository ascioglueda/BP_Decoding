import random
import sys
sys.path.insert(1, '.')
from source import DawnSimVis
import numpy as np

SOURCE = 0  # Root node ID
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
def encode_lt(data, distribution):
    degree = np.random.choice(np.arange(1, k + 1), p=distribution)
    selected_indices = random.sample(range(k), min(degree, k))
    encoded_symbol = data[selected_indices[0]]
    for i in selected_indices[1:]:
        encoded_symbol ^= data[i]
    return selected_indices, encoded_symbol

# Belief Propagation Decoder
def decode_lt(received_packets, k):
    known_values = np.full(k, -1, dtype=np.int8)
    symbol_queue = []
    equations = received_packets.copy()

    # İlk tek bilinmeyenli denklemleri kuyrukla başlat
    for node_id, (indices, value) in enumerate(equations):
        if len(indices) == 1:
            symbol_queue.append((node_id, indices[0], value))

    while symbol_queue:
        node_id, index, value = symbol_queue.pop(0)
        if known_values[index] != -1:
            continue
        known_values[index] = value
        for other_node_id, (indices, val) in enumerate(equations):
            if other_node_id == node_id:
                continue
            if index in indices:
                indices.remove(index)
                val ^= value
                equations[other_node_id] = (indices, val)
                if len(indices) == 1:
                    symbol_queue.append((other_node_id, indices[0], val))

    # Bilinmeyenleri sıfır yap
    result = np.zeros(k, dtype=np.uint8)
    for i in range(k):
        result[i] = known_values[i] if known_values[i] != -1 else 0
    return result

###########################################################
class Node(DawnSimVis.BaseNode):
    def init(self):
        self.currstate = 'IDLE'
        self.parent = None
        self.children = []
        self.childs = set()
        self.others = set()
        self.sent_probes = False
        self.received_reject = set()
        self.received_ack = set()
        self.received_packets = []
        self.visited = False

        self.log(f"Node {self.id} started in IDLE state.")

    def run(self):
        if self.id == SOURCE:
            self.change_color(1, 0, 0)  # Kırmızı: Root
            self.currstate = 'XPLORING'
            self.sent_probes = True
            self.log(f"Root node {self.id} sent probe message.")
            self.cb_flood_send({'type': 'probe', 'sender': self.id})

            # 5 kez encoded paket gönder
            probabilities = robust_soliton_distribution(k)
            for i in range(5):
                selected_indices, encoded_symbol = encode_lt(data, probabilities)
                self.log(f"Node {self.id} encoded: indices = {selected_indices}")
                encoded_packet = {
                    'type': 'encoded',
                    'sender': self.id,
                    'indices': (selected_indices, encoded_symbol)
                }
                self.set_timer(1 + i * 0.5, self.cb_msg_send, encoded_packet)

    def cb_flood_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        self.log(f"Node {self.id} broadcasted message.")

    def cb_msg_send(self, pck):
        # Mesajları yalnızca children ve parent'a gönder
        if self.parent is not None:
            self.send(self.parent, pck)
        for child in self.children:
            self.send(child, pck)
        self.log(f"Node {self.id} sent packet to parent and children.")

    def on_receive(self, pck):
        sender_id = pck['sender']
        msg_type = pck['type']
        self.log(f"Node {self.id} received '{msg_type}' from {sender_id}.")

        # Spanning Tree oluşturma
        if self.currstate == 'IDLE' and msg_type == 'probe':
            self.parent = sender_id
            self.currstate = 'XPLORING'
            self.change_color(0, 0, 1)  # Mavi: XPLORING
            self.log(f"Node {self.id} set {self.parent} as parent.")
            self.set_timer(1.5, self.cb_flood_send, {'type': 'probe', 'sender': self.id})
            self.sent_probes = True

        elif self.currstate == 'XPLORING':
            if msg_type == 'probe' and sender_id != self.parent:
                self.send(sender_id, {'type': 'reject', 'sender': self.id})
                self.log(f"Node {self.id} sent reject to {sender_id}.")
            elif msg_type == 'ack':
                self.received_ack.add(sender_id)
                self.childs.add(sender_id)
                self.log(f"Node {self.id} added {sender_id} as child.")
            elif msg_type == 'reject':
                self.received_reject.add(sender_id)
                self.others.add(sender_id)
                self.log(f"Node {self.id} marked {sender_id} as other.")

            if self.sent_probes and (self.received_ack or self.received_reject):
                if self.id != SOURCE:
                    self.send(self.parent, {'type': 'ack', 'sender': self.id})
                    self.change_color(0, 1, 0)  # Yeşil: TERM
                    self.currstate = 'TERM'
                    self.log(f"Node {self.id} transitioning to TERM state.")

                    # TERM durumuna geçtikten sonra 5 kez encoded paket gönder
                    probabilities = robust_soliton_distribution(k)
                    for i in range(5):
                        selected_indices, encoded_symbol = encode_lt(data, probabilities)
                        self.set_timer(2 + i * 0.5, self.cb_msg_send, {
                            'type': 'encoded',
                            'sender': self.id,
                            'indices': (selected_indices, encoded_symbol)
                        })
                else:
                    self.currstate = 'TERM'
                    self.log(f"Root node {self.id} finished.")

        # Encoded paketleri al ve SOURCE'a gönder
        if msg_type == 'encoded' and 'indices' in pck and isinstance(pck['indices'], tuple):
            indices_data = pck['indices']
            if len(indices_data) == 2:
                selected_indices, encoded_symbol = indices_data
                self.received_packets.append((selected_indices.copy(), encoded_symbol))
                self.log(f"Node {self.id} received packet from {sender_id}: {selected_indices}")

                # Paketleri SOURCE'a gönder
                if self.id != SOURCE:
                    self.send(SOURCE, {
                        'type': 'encoded',
                        'sender': self.id,
                        'indices': (selected_indices.copy(), encoded_symbol)
                    })

                # Parent-child ilişkisini kur
                if sender_id not in self.children and sender_id != self.parent:
                    if self.parent is None or sender_id == self.parent:
                        self.parent = sender_id
                        if self.parent < len(self.sim.nodes):
                            parent_node = self.sim.nodes[self.parent]
                            if parent_node:
                                parent_node.children.append(self.id)

    def finish(self):
        if self.parent is not None:
            self.log(f'Node {self.id} -> Parent: {self.parent}')
        if self.children:
            self.log(f'Node {self.id} -> Children: {self.children}')
        
        # Sadece SOURCE düğümü decoding yapar ve sonucu loglar
        if self.id == SOURCE and self.received_packets:
            self.log(f"SOURCE Node {self.id} collected {len(self.received_packets)} encoded packets in total.")
            decoded_data = decode_lt(self.received_packets, k)
            correct_bits = np.sum(decoded_data == data)
            self.log(f"SOURCE Node decoded {correct_bits}/{k} bits correctly.")
            self.log(f"Decoded data (first 100 bits): {decoded_data[:100]}")
            if np.array_equal(decoded_data, data):
                self.change_color(1, 0, 1)  # Mor: Başarılı decoding
                self.log(f"SOURCE Node {self.id} successfully decoded the data! Overall decoding successful.")
            else:
                self.log(f"SOURCE Node {self.id} failed to decode the data. Overall decoding failed.")

###########################################################
def create_network():
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-10, 10)
            py = 50 + y * 60 + random.uniform(-10, 10)
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Set up the simulation
sim = DawnSimVis.Simulator(
    duration=100,  # Simulation duration (seconds)
    timescale=1,  # Time scale to control simulation speed
    visual=True,   # Enable visualization
    terrain_size=(650, 650),  # Terrain size
    title='Spanning Tree Simulation'  # Title
)

# Create the network
create_network()

# Start the simulation
sim.run()