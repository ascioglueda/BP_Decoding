# Gerekli kütüphaneler ve modüller içe aktarılıyor
import random
import sys
import numpy as np
from source import DawnSimVis  # Simülasyon ortamını sağlayan özel bir modül

SOURCE = 0  # Root düğüm
k = 1000  # Orijinal veri boyutu
n = 100  # Düğüm sayısı

# 0 ve 1'lerden oluşan rastgele veri oluşturuluyor
data = np.random.randint(0, 2, k, dtype=np.uint8)

# Robust Soliton dağılım fonksiyonu (LT kodları için dağılım hesaplar)
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

# LT (Luby Transform) kodlama fonksiyonu
def encode_lt(data, distribution):
    degree = np.random.choice(np.arange(1, k + 1), p=distribution)  # Rastgele derece seçilir
    selected_indices = random.sample(range(k), min(degree, k))  # Seçilen indeksler
    encoded_symbol = data[selected_indices[0]]
    for i in selected_indices[1:]:
        encoded_symbol ^= data[i]  # XOR işlemi ile kodlama yapılır
    return selected_indices, encoded_symbol

# LT kod çözme fonksiyonu (Belief Propagation Decoder)
def decode_lt(received_packets, k, node_id, original_data):
    known_values = np.full(k, -1, dtype=np.int8)  # Bilinmeyen değerler -1 olarak atanır
    symbol_queue = []
    equations = received_packets.copy()

    # Tek bilinmeyenli (tek indeksli) denklemler kuyruğa eklenir
    print(f"Node {node_id}: Baslangicta derecesi 1 olan paketler:")
    for packet_id, (indices, value) in enumerate(equations):
        if len(indices) == 1:
            print(f"  Paket {packet_id}: indeks={indices[0]}, değer={value}")
            symbol_queue.append((packet_id, indices[0], value))

    # Belief Propagation döngüsü
    while symbol_queue:
        packet_id, index, value = symbol_queue.pop(0)
        if known_values[index] != -1:
            continue
        known_values[index] = value
        for other_packet_id, (indices, val) in enumerate(equations):
            if other_packet_id == packet_id:
                continue
            if index in indices:
                indices.remove(index)
                val ^= value
                equations[other_packet_id] = (indices, val)
                if len(indices) == 1:
                    symbol_queue.append((other_packet_id, indices[0], val))

    # Bilinmeyenler sıfır kabul edilerek sonuç dizisi oluşturulur
    result = np.zeros(k, dtype=np.uint8)
    for i in range(k):
        result[i] = known_values[i] if known_values[i] != -1 else 0

    # Dekodlama başarısını hesapla ve logla
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
        self.collected_messages = set()  # Root'un topladığı düğüm ID'leri
        self.encode_count = 0  # Root için encode mesaj sayacı
        self.log(f"Node {self.id} started in IDLE state.")

    def run(self):
        # Root düğümü çalıştırılırsa
        if self.id == SOURCE:
            self.change_color(1, 0, 0)  # Kırmızı renk
            self.currstate = 'XPLORING'
            self.sent_probes = True
            self.log(f"Root node {self.id} sent probe message.")
            self.cb_flood_send({'type': 'probe', 'sender': self.id})
            # Spanning tree tamamlanmazsa diye bir yedek zamanlayıcı ekleyelim
            self.set_timer(20, self.force_start_encoding)

    def cb_flood_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)

    def start_encoding(self):
        # Root encode işlemini başlatır
        probabilities = robust_soliton_distribution(k)
        self.log(f"Node {self.id} starting encoding process. Collected {len(self.collected_messages)} messages out of {n-1}.")

        # Root kendine bir sembol saklar
        selected_indices, encoded_symbol = encode_lt(data, probabilities)
        self.received_packets.append((selected_indices.copy(), encoded_symbol))
        self.log(f"Root node {self.id} saved a symbol for itself with indices: {selected_indices}")

        # Her düğüm için 1 encode paketi gönder
        for target_node in range(1, n):  # 1'den (n-1)'e kadar tüm düğümler
            selected_indices, encoded_symbol = encode_lt(data, probabilities)
            self.encode_count += 1
            pck = {
                'type': 'encoded',
                'sender': self.id,
                'indices': selected_indices,
                'symbol': encoded_symbol,
                'target': target_node  # Hedef düğüm ID'si
            }
            # Paketleri sırayla göndermek için zamanlayıcı kullan
            self.set_timer(2 + self.encode_count * 0.01, self.send_encoded_packet, pck)
        
        # Encode işlemi bittikten 10 saniye sonra decode işlemini başlat
        total_encoding_time = 2 + (n - 1) * 0.01  # n-1 diğer düğüm sayısıdır
        self.set_timer(total_encoding_time + 10, self.start_decoding)

    def send_encoded_packet(self, pck):
        # Mesajı doğrudan hedef düğüme gönder
        target_node = pck['target']
        self.send(target_node, pck)
        self.log(f"Node {self.id} sent encoded packet to node {target_node} with indices: {pck['indices']}")

    def force_start_encoding(self):
        # Eğer spanning tree tamamlanmazsa, encode işlemini zorla başlat
        if self.currstate != 'TERM':
            self.log(f"Node {self.id} forcing encoding start. Collected {len(self.collected_messages)} messages out of {n-1}.")
            self.currstate = 'TERM'
            self.start_encoding()

    def start_decoding(self):
        # Root için decode işlemini başlat
        if self.id == SOURCE and self.received_packets:
            self.log(f"Node {self.id} starting decoding process with {len(self.received_packets)} packets.")
            decode_lt(self.received_packets, k, self.id, data)

    def on_receive(self, pck):
        sender_id = pck['sender']
        msg_type = pck['type']
        self.log(f"Node {self.id} received '{msg_type}' packet from node {sender_id}.")

        # Spanning Tree oluşturma
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

            # TERM durumuna geçiş
            if self.sent_probes and (self.received_ack or self.received_reject):
                if self.id != SOURCE:
                    self.send(self.parent, {'type': 'ack', 'sender': self.id})
                    self.change_color(0, 1, 0)  # Yeşil renk
                    self.currstate = 'TERM'
                    self.log(f"Node {self.id} transitioned to TERM state.")
                else:
                    self.collected_messages.add(sender_id)
                    self.log(f"Root node {self.id} collected message from {sender_id}")
                    if len(self.collected_messages) >= n - 1:
                        self.currstate = 'TERM'
                        self.log(f"Root node {self.id} finished collecting messages. Starting encoding.")
                        self.set_timer(10, self.start_encoding)

        # Encode edilmiş mesaj alındığında
        if msg_type == 'encoded':
            # Paket yapısına göre veriyi doğru şekilde al
            if 'indices' in pck and 'symbol' in pck:
                selected_indices = pck['indices']
                encoded_symbol = pck['symbol']
            else:
                self.log(f"Node {self.id} received invalid encoded packet from {sender_id}")
                return

            # Alınan paketi kaydet
            self.received_packets.append((selected_indices.copy(), encoded_symbol))
            self.log(f"Node {self.id} received encoded packet from {sender_id} with indices: {selected_indices}")
            

    def finish(self):
        pass

def create_network():
    rows = int(np.sqrt(n))  # Yaklaşık kare kök ile satır sayısı
    cols = (n + rows - 1) // rows  # Sütun sayısı (n'e tam uyması için yuvarlama)
    spacing = 60
    for x in range(cols):
        for y in range(rows):
            if (x * rows + y) < n:  # n sayısına kadar düğüm ekle
                px = 50 + x * spacing + random.uniform(-20, 20)  # x ekseni + rastgele kaydırma
                py = 50 + y * spacing + random.uniform(-20, 20)  # y ekseni + rastgele kaydırma
                sim.add_node(Node, pos=(px, py), tx_range=100)

# Simülasyon ortamını başlatan ayarlar
sim = DawnSimVis.Simulator(
    duration=500,  # Süre, 99 mesaj için yeterli
    timescale=1,
    visual=True,
    terrain_size=(650, 650),
    title='Belief Propagation Decoding'
)

# Ağı oluştur ve simülasyonu başlat
create_network()
sim.run()