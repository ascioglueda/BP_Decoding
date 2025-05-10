# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 13:37:01 2025

@author: EDA AŞÇIOĞLU
"""

import random
import sys
sys.path.insert(1, '.')
from source import DawnSimVis

SOURCE = 0  # Root node ID

###########################################################
class Node(DawnSimVis.BaseNode):

    ###################
    def init(self):
        self.currstate = 'IDLE'  # Current state: IDLE, XPLORING, TERM
        self.parent = None  # Parent node ID
        self.childs = set()  # Child node IDs
        self.others = set()  # Other neighbor IDs
        self.sent_probes = False  # Track if probe has been sent
        self.received_rejects_from = set()  # Neighbors from which reject was received
        self.received_acks_from = set()  # Neighbors from which ack was received
        self.change_color(0.5, 0.5, 0.5)  # Gray color for IDLE state
        self.log(f"Node {self.id} started in IDLE state.")

    ###################
    def run(self):
        if self.id == SOURCE:
            # Root node starts the tree construction process
            self.change_color(1, 0, 0)  # Red color for root
            self.currstate = 'XPLORING'
            pck = {'type': 'probe', 'sender': self.id}
            self.cb_flood_send(pck)  # Send probe to all neighbors
            self.sent_probes = True
            self.log(f"Root node {self.id} sent probe message.")

    ###################
    def cb_flood_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        self.log(f"Node {self.id} sent probe message.")

    ###################
    def on_receive(self, pck):
        sender_id = pck['sender']
        msg_type = pck['type']
        self.log(f"Node {self.id} ({self.currstate}) received message: {msg_type} from {sender_id}")

        if self.currstate == 'IDLE' and msg_type == 'probe':
            # When the first probe is received, set the sender as parent
            self.parent = sender_id
            self.change_color(0, 0, 1)  # Blue color for XPLORING state
            self.log(f"Node {self.id} set {self.parent} as parent.")
            self.currstate = 'XPLORING'
            pck = {'type': 'probe', 'sender': self.id}
            self.set_timer(1.5, self.cb_flood_send, pck)  # Send probe to neighbors with a small delay
            self.sent_probes = True

        elif self.currstate == 'XPLORING':
            if msg_type == 'probe' and sender_id != self.parent:
                # If probe is received (not from parent), send reject
                self.send(sender_id, {'type': 'reject', 'sender': self.id})
                self.log(f"Node {self.id} sent reject to {sender_id}.")
            
            elif msg_type == 'ack':
                # When ack is received, add sender as child
                self.received_acks_from.add(sender_id)
                self.childs.add(sender_id)
                self.log(f"Node {self.id} added {sender_id} as child.")
            
            elif msg_type == 'reject':
                # When reject is received, add sender to other neighbors
                self.received_rejects_from.add(sender_id)
                self.others.add(sender_id)
                self.log(f"Node {self.id} added {sender_id} to other neighbors.")

            # Check if responses from all neighbors are received (simplified termination condition)
            if self.sent_probes and (self.received_acks_from or self.received_rejects_from):
                if self.id != SOURCE:
                    # If not the root, send ack to parent and transition to TERM state
                    self.send(self.parent, {'type': 'ack', 'sender': self.id})
                    self.log(f"Node {self.id} sent ack to {self.parent}, transitioning to TERM state.")
                    self.change_color(0, 1, 0)  # Green color for TERM state
                    self.currstate = 'TERM'
                else:
                    # Root node completed, but stays red
                    self.log(f"Root node {self.id} completed, transitioned to TERM state.")
                    self.currstate = 'TERM'

###########################################################
def create_network():
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Set up the simulation
sim = DawnSimVis.Simulator(
    duration=300,  # Simulation duration (seconds)
    timescale=1,  # Time scale to control simulation speed
    visual=True,   # Enable visualization
    terrain_size=(650, 650),  # Terrain size
    title='Spanning Tree Simulation'  # Title
)

# Create the network
create_network()

# Start the simulation
sim.run()