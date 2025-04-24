import random
import sys
sys.path.insert(1, '.')
from source import DawnSimVis


ROUND = 'round'
CH_BLACK = 'ch_black'
CH_GRAY = 'ch_gray'
UNDECIDE = 'undecide'
NO_CHANGE = 'no_change'

###########################################################
class Node(DawnSimVis.BaseNode):
    def init(self):
        
        self.color = 'white'              # Düğüm beyaz olarak başlar
        self.spans = {}                   # spans[di]: degree + 1
        self.neigh_cols = {}              # Komşu renkleri
        self.recvd_cols = set()           # Alınan renkler
        self.received = set()             
        self.lost_neighs = set()          # Kayıp komşular
        self.curr_neighs = set()          # Şimdiki komşular
        self.round_recvd = False          # Round message alış
        self.round_over = False           # Round bitiş
        self.finished = False             
        self.n_recvd = 0                  # ch_gray/no_change mesaj sayısı
        self.round_k = 0                  # Mevcut tur sayısı

        # Komşuları ve dereceleri belirleme
        for _, node in self.neighbor_distance_list:
            if DawnSimVis.distance(self.pos, node.pos) <= self.tx_range:
                self.curr_neighs.add(node.id)
                degree = len([n for d, n in node.neighbor_distance_list if d <= node.tx_range])
                self.spans[node.id] = degree + 1
                self.neigh_cols[node.id] = 'white'
        self.spans[self.id] = len(self.curr_neighs) + 1
        self.neigh_cols[self.id] = 'white'
        self.update_color()
        self.log(f"Initialized with spans={self.spans[self.id]}, neighbors={self.curr_neighs}")

    def update_color(self):
        
        if self.color == 'black':
            self.change_color(0, 0, 0)  # Black for dominating set
        elif self.color == 'gray':
            self.change_color(0.5, 0.5, 0.5)  # Gray
        else:
            self.change_color(1, 1, 1)  # White for undecided

    def run(self):
        # Turları yönetme
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

            # Eğer düğüm beyaz ise, span'ları göndermek için round başlatır true ayarlar
            if self.color == 'white':
                self.broadcast_round()
                self.round_recvd = True

            # Mesaj değişimine izin vermek için 6 saniye bekler
            yield self.timeout(6.0)

            # Eğer bitmemişse ve round_recvd True ise, process_decision_phase çağrılır
            if not self.finished and self.round_recvd:
                self.process_decision_phase()

            # Round bitene kadar bekle
            timeout = self.now + 3.0
            while not self.round_over and self.now < timeout:
                yield self.timeout(0.3)

            # Bir sonraki tur için durumu sıfırlar
            self.round_over = True
            self.curr_neighs -= self.lost_neighs
            self.n_recvd = 0
            self.round_recvd = False
            self.finished = False
            self.received.clear()
            self.recvd_cols.clear()
            self.lost_neighs.clear()
            self.log(f"Completed round {self.round_k}")

            
            all_decided = True
            white_nodes = []
            for node in self.sim.nodes:
                if node.color == 'white':
                    all_decided = False
                    white_nodes.append(node.id)
            if all_decided:
                self.log("All nodes are black or gray, terminating.")
                break
            if white_nodes and self.color == 'white':
                self.log(f"Still white, neighbors={self.curr_neighs}, neigh_cols={self.neigh_cols}, recvd_cols={self.recvd_cols}")

        self.log(f"Terminated in state {self.color}")

    def broadcast_round(self):
        # Bir turu başlatmak için düğümün kapsamlarını tüm komşularına gönderir.
        pck = {
            'type': ROUND,
            'round': self.round_k,
            'sender': self.id,
            'spans': self.spans.get(self.id, 0)
        }
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        self.delayed_exec(0.5, self.send, DawnSimVis.BROADCAST_ADDR, pck)
        self.log(f"Sent round {self.round_k} with spans={self.spans.get(self.id, 0)}")

    def send_state_to_neighbors(self, msg_type):
        # Komşulara bir durum mesajı gönder
        pck = {'type': msg_type, 'sender': self.id, 'round': self.round_k}
        for (dist, neighbor) in self.neighbor_distance_list:
            if dist <= self.tx_range and neighbor.id in self.curr_neighs:
                self.send(neighbor.id, pck)
                self.delayed_exec(0.5, self.send, neighbor.id, pck)
        self.log(f"Sent {msg_type} to neighbors")

    def process_decision_phase(self):
        # Düğümlerin alınan mesajlara göre renklerine karar verir
        
        if self.received != self.curr_neighs:
            self.log(f"Decision phase: waiting for neighbors, received={self.received}, expected={self.curr_neighs}")
            
            white_neighs = [nid for nid in self.curr_neighs if self.neigh_cols.get(nid, 'white') == 'white']
            if not white_neighs:  
                if self.color == 'white':
                    has_black_neighbor = any(self.neigh_cols.get(nid, 'white') == 'black' for nid in self.curr_neighs)
                    if has_black_neighbor:
                        self.color = 'gray'
                        self.spans[self.id] -= 1
                        self.send_state_to_neighbors(CH_GRAY)
                        self.update_color()
                        self.log(f"Forced gray due to black neighbor, neigh_cols={self.neigh_cols}")
                    else:
                        self.color = 'black'  # Become black if no white neighbors and no black neighbors
                        self.spans[self.id] = 0
                        self.send_state_to_neighbors(CH_BLACK)
                        self.update_color()
                        self.log(f"Forced black: no white neighbors, spans={self.spans[self.id]}")
                self.finished = True
                return

        if 'black' in self.recvd_cols:
            if self.color == 'white':
                self.color = 'gray'
                self.spans[self.id] -= 1
                self.send_state_to_neighbors(CH_GRAY)
                self.update_color()
                self.log(f"Became gray due to black neighbor, recvd_cols={self.recvd_cols}")
            else:
                self.send_state_to_neighbors(NO_CHANGE)
                self.log(f"Sent no_change (already {self.color})")
        else:
            
            white_neighs = [nid for nid in self.curr_neighs if self.neigh_cols.get(nid, 'white') == 'white']
            neighbor_spans = [(self.spans.get(nid, 0), nid) for nid in white_neighs]
            if not white_neighs:
                max_span, max_id = 0, -1
            else:
                max_span, max_id = max(neighbor_spans, key=lambda x: (x[0], -x[1])) 
            my_span = self.spans.get(self.id, 0)
            if self.color == 'white':
                if my_span > max_span or (my_span == max_span and self.id < max_id):
                    self.color = 'black'
                    self.spans[self.id] = 0
                    self.send_state_to_neighbors(CH_BLACK)
                    self.update_color()
                    self.log(f"Became black (spans={my_span}, max_neighbor_span={max_span})")
                else:
                    
                    has_black_neighbor = any(self.neigh_cols.get(nid, 'white') == 'black' for nid in self.curr_neighs)
                    if has_black_neighbor:
                        self.color = 'gray'
                        self.spans[self.id] -= 1
                        self.send_state_to_neighbors(CH_GRAY)
                        self.update_color()
                        self.log(f"Forced gray due to black neighbor, neigh_cols={self.neigh_cols}")
                    else:
                        self.send_state_to_neighbors(NO_CHANGE)
                        self.log(f"Sent no_change (spans={my_span}, max_neighbor_span={max_span})")
        self.finished = True

    def on_receive(self, pck):
        # Gelen mesajı işle
        msg_type = pck.get('type')
        sender = pck.get('sender')
        round_num = pck.get('round', 0)

        # Yanlış turdan gelen mesajlar göz ardı edilir
        if round_num != self.round_k or sender not in self.curr_neighs:
            return

        self.log(f"Received {msg_type} from {sender} in round {self.round_k}")

          # Round mesajı alındıysa
        if msg_type == ROUND:
            self.received.add(sender)
            if self.color == 'white':
                self.spans[sender] = pck.get('spans', self.spans.get(sender, 0))
                white_neighs = [nid for nid in self.curr_neighs if self.neigh_cols.get(nid, 'white') == 'white']
                neighbor_spans = [(self.spans.get(nid, 0), nid) for nid in white_neighs]
                if not white_neighs:
                    max_span, max_id = 0, -1
                else:
                    max_span, max_id = max(neighbor_spans, key=lambda x: (x[0], -x[1]))  
                my_span = self.spans.get(self.id, 0)
                if my_span > max_span or (my_span == max_span and self.id < max_id):
                    self.color = 'black'
                    self.spans[self.id] = 0
                    self.send_state_to_neighbors(CH_BLACK)
                    self.update_color()
                    self.log(f"Became black (spans={my_span}, max_neighbor_span={max_span})")
                else:
                    self.send_state_to_neighbors(UNDECIDE)
                    self.log(f"Sent undecide (spans={my_span}, max_neighbor_span={max_span})")
                self.round_recvd = True
            else:
                
                self.send_state_to_neighbors(NO_CHANGE)
                self.log(f"Sent no_change to {sender} (I am {self.color})")

        
        elif msg_type == CH_BLACK:
            self.received.add(sender)
            self.recvd_cols.add('black')
            if self.neigh_cols.get(sender, 'white') == 'white' and self.spans.get(self.id, 0) > 0:
                self.spans[self.id] -= 1
                self.lost_neighs.add(sender)
            self.neigh_cols[sender] = 'black'
            self.log(f"Received ch_black from {sender}, spans={self.spans[self.id]}")

        
        elif msg_type == UNDECIDE:
            self.received.add(sender)
            self.log(f"Received undecide from {sender}")

        
        elif msg_type == CH_GRAY:
            self.n_recvd += 1
            if self.neigh_cols.get(sender, 'white') == 'white' and self.spans.get(self.id, 0) > 0:
                self.spans[self.id] -= 1
            self.neigh_cols[sender] = 'gray'
            self.log(f"Received ch_gray from {sender}, spans={self.spans[self.id]}")

        
        elif msg_type == NO_CHANGE:
            self.n_recvd += 1
            self.received.add(sender)
            self.log(f"Received no_change from {sender}")

        
        if self.finished and self.n_recvd >= len(self.curr_neighs):
            self.round_over = True
            self.curr_neighs -= self.lost_neighs
            self.n_recvd = 0
            self.round_recvd = False
            self.finished = False
            self.received.clear()
            self.recvd_cols.clear()
            self.lost_neighs.clear()
            self.log(f"Completed round {self.round_k}")

    def is_terminated(self):
       
        if self.color == 'white':
            return False
        for nid in self.curr_neighs:
            if self.neigh_cols.get(nid, 'white') == 'white':
                return False
        return True

    def finish(self):
        """Log final state."""
        self.log(f"Finished in state {self.color}")

###########################################################
def create_network():
    """Create a 10x10 grid of nodes with random offsets."""
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=75)

sim = DawnSimVis.Simulator(
    duration=2000,  
    timescale=0.1,
    visual=True,
    terrain_size=(650, 650),
    title='Span_MDS Algorithm (Revised)'
)
create_network()

sim.run()