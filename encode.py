import sys
sys.path.insert(1, '.')
from source import DawnSimVis
import random

SOURCE = 0

###########################################################
class Node(DawnSimVis.BaseNode):

    ###################
    # Node başlatılırken ilk ayarlamalar
    def init(self):
        self.flood_received = False
        self.parent = None  # Spanning tree için üst düğüm
        self.visited = False  # Mesajın işlenip işlenmediğini kontrol etmek için

    ###################
    # Simülasyon başında root mesaj yayını
    def run(self):
        if self.id == SOURCE:
            self.change_color(1, 0, 0)  # Root kırmızı
            # Başlangıç mesajını yayınla, msg_id olmadan
            pck = {'type': 'flood', 'sender': self.id}
            self.cb_flood_send(pck)
            self.flood_received = True
            self.visited = True

    ###################
    # Mesaj alındığında çağrılır
    def on_receive(self, pck):
        if pck['type'] == 'flood' and not self.visited:
            self.visited = True
            self.parent = pck['sender']  # Üst düğümü kaydet
            self.change_color(0, 1, 0)  # Ziyaret edilen düğüm yeşil
            pck = {'type': 'flood', 'sender': self.id}
            self.set_timer(1, self.cb_flood_send, pck)
            self.flood_received = True

    ###################
    # Zamanlayıcıyla tetiklenen yayın fonksiyonu
    def cb_flood_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)  # Mesajı tüm komşulara yayınla
        self.log(f'Flood msg sent from {self.id}')

    ###################
    # Düğüm sonu
    def finish(self):
        pass

###########################################################
def create_network():
    # 100x100 grid üzerinde düğümleri yerleştir
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Simulator nesnesi oluştur
sim = DawnSimVis.Simulator(
    duration=100,
    timescale=1,
    visual=True,  # Görselleştirme açık
    terrain_size=(650, 650),  # Simülasyon alanı
    title='Spanning Tree with Flooding (No msg_id)')

# Ağı oluştur
create_network()

# Simülasyonu başlat
sim.run()