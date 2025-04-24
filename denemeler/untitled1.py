import sys
sys.path.insert(1, '.')
from source import DawnSimVis
import random

MSG_IN = "IN"

class Node(DawnSimVis.BaseNode):
    def init(self):
        self.state = "UNDECIDED" 
        self.change_color(1, 1, 1)  # Beyaz
        self.neighbors = [n.id for (dist, n) in self.neighbor_distance_list if dist <= self.tx_range]
        self.neighbor_in = False
        # Karar verme süresi
        delay = random.uniform(0.5, 2.0)
        self.set_timer(delay, self.make_decision)

    def run(self):
        pass

    def make_decision(self):
        if not self.neighbor_in:
            self.state = "IN"
            self.change_color(0, 0, 0)  # Siyah
            self.log("Decided IN (dominator)")
            self.send_message_to_neighbors(MSG_IN)
        else:
            self.state = "OUT"
            self.change_color(0.5, 0.5, 0.5)  # Gri
            self.log("Decided OUT (dominated)")

    def on_receive(self, pck):
        msg_type = pck.get("type")
        sender = pck.get("sender")
        if msg_type == MSG_IN:
            self.neighbor_in = True
            self.log(f"Received IN from Node {sender}")

    def finish(self):
        self.log(f"Finished with state {self.state}")

    def send_message_to_neighbors(self, msg_type):
        pck = {"type": msg_type, "sender": self.id}
        self.send(DawnSimVis.BROADCAST_ADDR, pck)

# Simülatör ayarları
sim = DawnSimVis.Simulator(
    duration=100,
    timescale=1,
    visual=True,
    terrain_size=(650, 650),
    title='Distributed Dominating Set Simulation'
)

def create_network():
    """
    NxN grid üzerinde düğümleri yerleştirirken
    her birine (±offset) rastgelelik ekleyerek
    düzenli ama tamamen benzer olmayan bir dağılım oluşturuyoruz.
    """
    N = 10     # 10x10 grid
    offset = 15  # Her düğüm için maksimum ±offset
    spacing = 60 # Düğümler arasındaki temel ızgara aralığı
    
    node_count = N * N
    for i in range(N):
        for j in range(N):
            # Grid tabanlı konum
            x_grid = 50 + i * spacing
            y_grid = 50 + j * spacing
            # Az miktarda jitter ekleyelim
            x_jitter = random.uniform(-offset, offset)
            y_jitter = random.uniform(-offset, offset)
            px = x_grid + x_jitter
            py = y_grid + y_jitter
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Ağ oluşturma
create_network()

sim.run()
