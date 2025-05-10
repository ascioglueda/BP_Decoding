import sys
sys.path.insert(1, '.')
from source import DawnSimVis
import random
import numpy as np
from collections import defaultdict

# Simülasyon parametreleri
SOURCE = 0
k = 1000   # Veri boyutu
n = 100    # Düğüm sayısı (10x10 grid için)

# Mesaj tipleri
MSG_PROBE = 'probe'
MSG_ACK = 'ack'
MSG_REJECT = 'reject'

class Node(DawnSimVis.BaseNode):
    def init(self):
        self.parent = None  # Spanning tree'deki ebeveyn düğüm
        self.children = set()  # Spanning tree'deki çocuk düğümler
        self.visited = False  # Probe mesajı alındı mı
        self.node_id = self.id  # Düğüm kimliği
        self.neighbors = set()  # Komşu düğümler
        self.pending_probes = set()  # Bekleyen probe mesajları

    def run(self):
        # Kaynak düğüm (id=0) ağaç oluşumunu başlatır
        if self.node_id == SOURCE:
            self.visited = True
            self.discover_neighbors()
        # Diğer düğümler rastgele bir gecikmeyle komşularını keşfeder
        else:
            delay = random.uniform(0.1, 0.5)
            self.set_timer(delay, self.discover_neighbors)

    def discover_neighbors(self):
        # Komşuları keşfetmek için probe mesajı gönder
        self.neighbors = set(self.get_neighbor_ids())
        for neighbor_id in self.neighbors:
            self.send(neighbor_id, {'type': MSG_PROBE, 'sender': self.node_id})

    def on_receive(self, pck):
        msg_type = pck['type']
        sender_id = pck['sender']

        if msg_type == MSG_PROBE:
            if not self.visited:
                # İlk probe mesajı: göndereni ebeveyn olarak seç
                self.visited = True
                self.parent = sender_id
                self.children.add(sender_id)
                self.send(sender_id, {'type': MSG_ACK, 'sender': self.node_id})
                # Diğer komşulara probe mesajı gönder
                self.discover_neighbors()
            else:
                # Zaten ziyaret edilmiş, reject gönder
                self.send(sender_id, {'type': MSG_REJECT, 'sender': self.node_id})

        elif msg_type == MSG_ACK:
            # ACK alındı, göndereni çocuk olarak ekle
            self.children.add(sender_id)
            self.pending_probes.discard(sender_id)

        elif msg_type == MSG_REJECT:
            # REJECT alındı, göndereni çocuk listesinden çıkar
            self.pending_probes.discard(sender_id)

    def finish(self):
        # Simülasyon sonunda ağaç yapısını kontrol et
        if self.node_id == SOURCE:
            print(f"Root Node {self.node_id}: Children = {self.children}")
        else:
            print(f"Node {self.node_id}: Parent = {self.parent}, Children = {self.children}")

def create_network():
    # 10x10 ızgara üzerinde düğümleri oluştur
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Simülatör oluştur
sim = DawnSimVis.Simulator(
    duration=30,
    timescale=1,
    visual=True,
    terrain_size=(650, 650),
    title='Spanning Tree with Probe, ACK, Reject'
)

create_network()
sim.run()