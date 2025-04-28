import sys
sys.path.insert(1, '.')
from source import DawnSimVis
import random
import numpy as np

SOURCE = 0
k = 1000
n = 2500
#Robust Soliton dagilimi
def soliton_distribution(k, delta=0.5):
    R = k / sum(1.0 / i for i in range(1, k + 1))
    probabilities = [R / i for i in range(1, k + 1)]
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # Normalizasyon
    return probabilities

###########################################################
class Node(DawnSimVis.BaseNode):

    ###################
    def init(self):
        self.msg_received = False # mesaji aldi mi
        self.parent = None
        self.visited = False  # Mesaj işlenip işlenmedi
        self.encoded_packet = None # gelen veriyi tutuyoruz
        self.used_indices = None # hangi verilerle xorlandigini tutuyoruz

    ###################
    def run(self):
        if self.id == SOURCE:
            self.change_color(1, 0, 0)  # Root kırmızı
            pck = {'type': 'msg', 'sender': self.id}
            self.cb_msg_send(pck)
            self.msg_received = True
            self.visited = True
            data=[random.randint(0,255) for _ in range(k)] # k tane eleman 
            for target_id in range(1,len(self.sim.nodes)):
                selected_indices = random.sample(range(k),random.randint(1,5))
                encoded_value = 0
                for i in selected_indices:
                    encoded_value^=data[i]
                
                #gonderilecek paket icerigi
                packet = {
                    'type':'encoded',
                    'sender':self.id,
                    'encoded_value': encoded_value,
                    'indices': selected_indices
                }

                self.send(target_id,packet)
    

    ###################
    def on_receive(self, pck):
        if pck['type'] == 'encoded' and not self.visited: #daha once islenmemisse not self.visited
            self.visited = True
            self.parent = pck['sender']

            self.encoded_packet = pck['encoded_value']#gelen encode verisi
            self.used_indices = pck['indices']#hangi paketler xorlandi

            if self.id != SOURCE:
                self.change_color(0, 0, 1)  
            self.log(f'Node {self.id}paketi aldi: value ={self.encoded_packet},indices ={self.used_indices}')

            new_pck = {'type': 'msg', 'sender': self.id}
            self.set_timer(1, self.cb_msg_send, new_pck)
            self.msg_received = True

        ###################
    def cb_msg_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        if self.id != SOURCE:
            self.change_color(0, 1, 0)
        self.log(f'Mesaj gönderildi: {self.id}')

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
    visual=True,  
    terrain_size=(650, 650),
    title='Belief Propagation Decoding')

# Ağı oluştur
create_network()

# Simülasyonu başlat
sim.run()


