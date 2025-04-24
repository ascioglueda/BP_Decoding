import random
import sys
sys.path.insert(1, '.')
from source import DawnSimVis

ROUND = 'round'
CH_BLACK = 'ch_black'
CH_GRAY = 'ch_gray'
UNDECIDE = 'undecide'
NO_CHANGE = 'no_change'
DISCOVER = 'discover'
DISCOVER_ACK = 'discover_ack'

###########################################################
class Node(DawnSimVis.BaseNode):
    def init(self):
        self.color = 'white'              # Düğüm beyaz olarak başlar
        self.spans = {}                   # spans[di]: degree + 1
        self.neigh_cols = {}              # Komşu renkleri
        self.recvd_cols = set()           # Alınan renkler
        self.received = set()             # Alınan mesajların göndericileri
        self.lost_neighs = set()          # Kayıp komşular
        self.curr_neighs = set()          # Şimdiki komşular
        self.neighbors = set()            # Tüm komşular
        self.round_recvd = False          # Round mesajı alındı mı
        self.round_over = False           # Tur bitti mi
        self.finished = False             # Karar aşaması tamamlandı mı
        self.n_recvd = 0                  # ch_gray/no_change mesaj sayısı
        self.round_k = 0                  # Mevcut tur sayısı
        self.undecide_count = 0           # Kararsız tur sayısı
        self.MAX_UNDECIDE_ROUNDS = 3      # Maksimum kararsız tur sayısı
        self.update_color()
        self.log(f"Node {self.id} initialized as white.")

    def update_color(self):
        if self.color == 'black':
            self.change_color(0, 0, 0)  # Siyah
        elif self.color == 'gray':
            self.change_color(0.5, 0.5, 0.5)  # Gri
        else:
            self.change_color(1, 1, 1)  # Beyaz

    def run(self):
        # Komşu keşfi için discover mesajı gönder
        self.send(DawnSimVis.BROADCAST_ADDR, {'type': DISCOVER, 'sender': self.id})
        yield self.timeout(2.0)  # Komşuların yanıt vermesi için bekle

        # Span'ı başlat
        self.spans[self.id] = len(self.neighbors) + 1
        self.neigh_cols[self.id] = 'white'

        # Ana döngü
        max_rounds = 200
        while self.round_k < max_rounds:
            self.round_k += 1
            self.round_over = False
            self.round_recvd = False
            self.finished = False
            self.received.clear()
            self.recvd_cols.clear()
            self.lost_neighs.clear()
            self.n_recvd = 0

            # Siyah değilse ve tüm komşuları gri/siyah değilse tur başlat
            if self.color != 'black' and not self.is_terminated():
                self.broadcast_round()
                self.round_recvd = True

            # Mesaj işleme için bekle
            yield self.timeout(3.0)

            # Karar aşaması
            if not self.finished and self.round_recvd:
                self.process_decision_phase()

            # Turun tamamlanması için bekle
            timeout = self.now + 2.0
            while not self.round_over and self.now < timeout:
                yield self.timeout(0.2)

            # Turu tamamla
            self.round_over = True
            self.curr_neighs -= self.lost_neighs
            self.n_recvd = 0
            self.round_recvd = False
            self.finished = False
            self.received.clear()
            self.recvd_cols.clear()
            self.lost_neighs.clear()
            self.log(f"Node {self.id} completed round {self.round_k}")

            # Sonlandırma kontrolü
            if self.is_terminated():
                self.log(f"Node {self.id} terminated: color={self.color}")
                break

        self.log(f"Node {self.id} finished with color {self.color}")

    def broadcast_round(self):
        pck = {
            'type': ROUND,
            'round': self.round_k,
            'sender': self.id,
            'spans': self.spans.get(self.id, 0),
            'color': self.color
        }
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        self.log(f"Node {self.id} sent round {self.round_k} with spans={self.spans.get(self.id, 0)}")

    def send_state_to_neighbors(self, msg_type):
        pck = {'type': msg_type, 'sender': self.id, 'round': self.round_k}
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        self.log(f"Node {self.id} sent {msg_type} to neighbors")

    def process_decision_phase(self):
        if self.color == 'black':
            self.finished = True
            return

        # Tüm komşulardan mesaj alındıysa
        if self.received == self.curr_neighs:
            white_neighs = [nid for nid in self.curr_neighs if self.neigh_cols.get(nid, 'white') == 'white']
            neighbor_spans = [(self.spans.get(nid, 0), nid) for nid in white_neighs]
            my_span = self.spans.get(self.id, 0)

            if not white_neighs:
                if self.color == 'gray':
                    self.color = 'black'
                    self.spans[self.id] = 0
                    self.send_state_to_neighbors(CH_BLACK)
                    self.update_color()
                    self.log(f"Node {self.id} became black: no white neighbors")
                elif self.color == 'white':
                    has_black_neighbor = any(self.neigh_cols.get(nid, 'white') == 'black' for nid in self.curr_neighs)
                    if has_black_neighbor:
                        self.color = 'gray'
                        self.spans[self.id] = 0
                        self.send_state_to_neighbors(CH_GRAY)
                        self.update_color()
                        self.log(f"Node {self.id} became gray: black neighbor detected")
                    else:
                        self.color = 'black'
                        self.spans[self.id] = 0
                        self.send_state_to_neighbors(CH_BLACK)
                        self.update_color()
                        self.log(f"Node {self.id} became black: no black or white neighbors")
                self.finished = True
                return

            max_span, max_id = max(neighbor_spans, key=lambda x: (x[0], -x[1])) if neighbor_spans else (0, -1)

            if self.color == 'white':
                if my_span > max_span or (my_span == max_span and self.id < max_id):
                    self.color = 'black'
                    self.spans[self.id] = 0
                    self.send_state_to_neighbors(CH_BLACK)
                    self.update_color()
                    self.log(f"Node {self.id} became black: spans={my_span}, max_neighbor_span={max_span}")
                    self.undecide_count = 0
                else:
                    has_black_neighbor = any(self.neigh_cols.get(nid, 'white') == 'black' for nid in self.curr_neighs)
                    if has_black_neighbor:
                        self.color = 'gray'
                        self.spans[self.id] = 0
                        self.send_state_to_neighbors(CH_GRAY)
                        self.update_color()
                        self.log(f"Node {self.id} became gray: black neighbor detected")
                        self.undecide_count = 0
                    else:
                        self.send_state_to_neighbors(UNDECIDE)
                        self.log(f"Node {self.id} sent undecide: spans={my_span}, max_neighbor_span={max_span}")
                        self.undecide_count += 1
                        # Kararsız kalma limiti
                        if self.undecide_count >= self.MAX_UNDECIDE_ROUNDS:
                            self.color = 'black'
                            self.spans[self.id] = 0
                            self.send_state_to_neighbors(CH_BLACK)
                            self.update_color()
                            self.log(f"Node {self.id} forced black: too many undecide rounds")
                            self.undecide_count = 0
            else:
                self.send_state_to_neighbors(NO_CHANGE)
                self.log(f"Node {self.id} sent no_change: already {self.color}")
            self.finished = True
        else:
            self.log(f"Node {self.id} waiting for neighbors: received={self.received}, expected={self.curr_neighs}")

    def on_receive(self, pck):
        msg_type = pck.get('type')
        sender = pck.get('sender')
        round_num = pck.get('round', 0)

        # Komşu keşfi
        if msg_type == DISCOVER:
            self.neighbors.add(sender)
            self.curr_neighs.add(sender)
            self.neigh_cols[sender] = 'white'
            self.spans[sender] = 0
            self.send(sender, {'type': DISCOVER_ACK, 'sender': self.id})
            self.log(f"Node {self.id} discovered neighbor {sender}")
            return

        if msg_type == DISCOVER_ACK:
            self.neighbors.add(sender)
            self.curr_neighs.add(sender)
            self.neigh_cols[sender] = 'white'
            self.spans[sender] = 0
            self.log(f"Node {self.id} received discover_ack from {sender}")
            return

        # Yanlış tur veya bilinmeyen gönderici
        if round_num != self.round_k or sender not in self.curr_neighs:
            return

        self.log(f"Node {self.id} received {msg_type} from {sender} in round {self.round_k}")

        if msg_type == ROUND:
            self.received.add(sender)
            self.spans[sender] = pck.get('spans', self.spans.get(sender, 0))
            self.neigh_cols[sender] = pck.get('color', 'white')
            self.round_recvd = True

        elif msg_type == CH_BLACK:
            self.received.add(sender)
            self.recvd_cols.add('black')
            if self.neigh_cols.get(sender, 'white') != 'black':
                self.spans[self.id] = max(0, self.spans.get(self.id, 0) - 1)
                self.lost_neighs.add(sender)
            self.neigh_cols[sender] = 'black'
            self.log(f"Node {self.id} updated {sender} to black, spans={self.spans[self.id]}")
            if self.color == 'white':
                self.color = 'gray'
                self.spans[self.id] = 0
                self.send_state_to_neighbors(CH_GRAY)
                self.update_color()
                self.log(f"Node {self.id} became gray: black neighbor detected")
                self.undecide_count = 0

        elif msg_type == CH_GRAY:
            self.n_recvd += 1
            if self.neigh_cols.get(sender, 'white') != 'gray':
                self.spans[self.id] = max(0, self.spans.get(self.id, 0) - 1)
            self.neigh_cols[sender] = 'gray'
            self.log(f"Node {self.id} updated {sender} to gray, spans={self.spans[self.id]}")

        elif msg_type == UNDECIDE:
            self.received.add(sender)
            self.n_recvd += 1
            self.log(f"Node {self.id} received undecide from {sender}")

        elif msg_type == NO_CHANGE:
            self.received.add(sender)
            self.n_recvd += 1
            self.log(f"Node {self.id} received no_change from {sender}")

        # Tur tamamlama kontrolü
        if self.finished and self.n_recvd >= len(self.curr_neighs):
            self.round_over = True
            self.curr_neighs -= self.lost_neighs
            self.n_recvd = 0
            self.round_recvd = False
            self.finished = False
            self.received.clear()
            self.recvd_cols.clear()
            self.lost_neighs.clear()
            self.log(f"Node {self.id} completed round {self.round_k}")

    def is_terminated(self):
        if self.color == 'white':
            return False
        for nid in self.curr_neighs:
            if self.neigh_cols.get(nid, 'white') == 'white':
                return False
        return True

    def finish(self):
        self.log(f"Node {self.id} finished in state {self.color}")

###########################################################
def create_network():
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Simülatör oluştur
sim = DawnSimVis.Simulator(
    duration=2000,
    timescale=0.1,
    visual=True,
    terrain_size=(650, 650),
    title='Span_MDS Algorithm (Corrected)'
)

# Ağı oluştur
create_network()

# Simülasyonu başlat
sim.run()