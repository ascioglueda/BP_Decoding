import random
import sys
import numpy as np
from source import DawnSimVis  

SOURCE = 0  
k = 1000 
n = 100  

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
    distribution /= distribution.sum()  # Normalize edilir
    return distribution

def encode_lt(data, distribution):
    if random.random()<0.1:
        degree =1
    else:
        degree = np.random.choice(np.arange(1, k + 1), p=distribution)  # Rastgele derece secilir
    selected_indices = random.sample(range(k), min(degree, k))  # Secilen indeksler
    encoded_symbol = data[selected_indices[0]]
    for i in selected_indices[1:]:
        encoded_symbol ^= data[i]  # XOR islemi ile kodlama yapilir
    return selected_indices, encoded_symbol

# Belief Propagation Decoder
def decode_lt(received_packets, k, node_id, original_data):
    known_values = np.full(k, -1, dtype=np.int8) 
    symbol_queue = []
    equations = [(indices.copy(), val) for indices, val in received_packets] 

    print(f"Node {node_id}: Baslangicta derecesi 1 olan paketleri kontrol ediyor...")

    for packet_id, (indices, value) in enumerate(equations):
        if len(indices) == 1:
            print(f"  [EKLENDİ] Paket {packet_id}: indeks={indices[0]}, deger={value}")
            symbol_queue.append((packet_id, indices[0], value))
        else:
            print(f"  [ATLANDI] Paket {packet_id}: derece={len(indices)}, indeksler={indices}")

    # Belief Propagation dongusu
    while symbol_queue:
        packet_id, index, value = symbol_queue.pop(0)
        if known_values[index] != -1:
            continue
        known_values[index] = value
        print(f"  [coZuLDu] İndeks {index} icin deger {value}")

        for other_packet_id, (indices, val) in enumerate(equations):
            if other_packet_id == packet_id:
                continue
            if index in indices:
                indices.remove(index)
                val ^= value
                equations[other_packet_id] = (indices, val)
                if len(indices) == 1:
                    symbol_queue.append((other_packet_id, indices[0], val))
                    print(f"  [YENİ] Paket {other_packet_id}: indeks={indices[0]}, deger={val}")

    # Bilinmeyenler sifir kabul edilerek sonuc dizisi olusturulur
    result = np.zeros(k, dtype=np.uint8)
    for i in range(k):
        result[i] = known_values[i] if known_values[i] != -1 else 0

    # Dekodlama basarisini hesapla ve logla
    correct_bits = np.sum(result == original_data)
    success_rate = (correct_bits / k) * 100
    print(f"Node {node_id} Decoding Result: {correct_bits}/{k} bits correctly decoded ({success_rate:.2f}% success rate)")
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
        self.collected_messages = set()  # Root'un topladigi dugum ID'leri
        self.encode_count = 0  # Root icin encode mesaj sayaci
        self.log(f"Node {self.id} started in IDLE state.")

    def run(self):
        # Root dugumu calistirilirsa
        if self.id == SOURCE:
            self.change_color(1, 0, 0)  # Kirmizi renk
            self.currstate = 'XPLORING'
            self.sent_probes = True
            self.log(f"Root node {self.id} sent probe message.")
            self.cb_flood_send({'type': 'probe', 'sender': self.id})
            # Spanning tree tamamlanmazsa diye bir yedek zamanlayici ekleyelim
            self.set_timer(20, self.force_start_encoding)

    def cb_flood_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)

    def start_encoding(self):
        # Root encode islemini baslatir
        probabilities = robust_soliton_distribution(k)
        self.log(f"Node {self.id} starting encoding process. Collected {len(self.collected_messages)} messages out of {n-1}.")

        # Root kendine bir sembol saklar
        selected_indices, encoded_symbol = encode_lt(data, probabilities)
        self.received_packets.append((selected_indices.copy(), encoded_symbol))
        self.log(f"Root node {self.id} saved a symbol for itself with indices: {selected_indices}")

        # Her dugum icin 1 encode paketi gonder
        for target_node in range(1, n):  # 1'den (n-1)'e kadar tum dugumler
            selected_indices, encoded_symbol = encode_lt(data, probabilities)
            self.encode_count += 1
            pck = {
                'type': 'encoded',
                'sender': self.id,
                'indices': selected_indices,
                'symbol': encoded_symbol,
                'target': target_node  # Hedef dugum ID'si
            }
            # Paketleri sirayla gondermek icin zamanlayici kullan
            self.set_timer(2 + self.encode_count * 0.01, self.send_encoded_packet, pck)
        
        # Encode islemi bittikten 10 saniye sonra decode islemini baslat
        total_encoding_time = 2 + (n - 1) * 0.01  # n-1 diger dugum sayisidir
        self.set_timer(total_encoding_time + 10, self.start_decoding)

    def send_encoded_packet(self, pck):
        # Mesaji dogrudan hedef dugume gonder
        target_node = pck['target']
        self.send(target_node, pck)
        self.log(f"Node {self.id} sent encoded packet to node {target_node} with indices: {pck['indices']}")

    def force_start_encoding(self):
        # Eger spanning tree tamamlanmazsa, encode islemini zorla baslat
        if self.currstate != 'TERM':
            self.log(f"Node {self.id} forcing encoding start. Collected {len(self.collected_messages)} messages out of {n-1}.")
            self.currstate = 'TERM'
            self.start_encoding()

    def start_decoding(self):
        # Root icin decode islemini baslat
        if self.id == SOURCE and self.received_packets:
            self.log(f"Node {self.id} starting decoding process with {len(self.received_packets)} packets.")
            decode_lt(self.received_packets, k, self.id, data)

    def on_receive(self, pck):
        sender_id = pck['sender']
        msg_type = pck['type']
        self.log(f"Node {self.id} received '{msg_type}' packet from node {sender_id}.")

        # Spanning Tree olusturma
        if self.currstate == 'IDLE' and msg_type == 'probe':
            self.parent = sender_id
            self.currstate = 'XPLORING'
            self.change_color(0, 0, 1)  # Mavi renk
            self.log(f"Node {self.id} set {self.parent} as parent.")
            self.set_timer(1.5, self.cb_flood_send, {'type': 'probe', 'sender': self.id})
            self.sent_probes = True

        elif self.currstate == 'XPLORING':
            if msg_type == 'probe' and sender_id != self.parent:
                self.send(sender_id, {'type': 'reject', 'sender': self.id})
                self.log(f"Node {self.id} sent reject to {sender_id}.")
            elif msg_type == 'ack':
                self.received_ack.add(sender_id)
                self.children.append(sender_id)
                self.log(f"Node {self.id} added {sender_id} as child.")
            elif msg_type == 'reject':
                self.received_reject.add(sender_id)
                self.others.add(sender_id)
                self.log(f"Node {self.id} marked {sender_id} as other.")

            # TERM durumuna gecis
            if self.sent_probes and (self.received_ack or self.received_reject):
                if self.id != SOURCE:
                    self.send(self.parent, {'type': 'ack', 'sender': self.id})
                    self.change_color(0, 1, 0)  # Yesil renk
                    self.currstate = 'TERM'
                    self.log(f"Node {self.id} transitioned to TERM state.")
                else:
                    self.collected_messages.add(sender_id)
                    self.log(f"Root node {self.id} collected message from {sender_id}")
                    if len(self.collected_messages) >= n - 1:
                        self.currstate = 'TERM'
                        self.log(f"Root node {self.id} finished collecting messages. Starting encoding.")
                        self.set_timer(10, self.start_encoding)

        # Encode edilmis mesaj alindiginda
        if msg_type == 'encoded':
            # Paket yapisina gore veriyi dogru sekilde al
            if 'indices' in pck and 'symbol' in pck:
                selected_indices = pck['indices']
                encoded_symbol = pck['symbol']
            else:
                self.log(f"Node {self.id} received invalid encoded packet from {sender_id}")
                return
            # Alinan paketi kaydet
            self.received_packets.append((selected_indices.copy(), encoded_symbol))
            self.log(f"Node {self.id} received encoded packet from {sender_id} with indices: {selected_indices}")
            

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