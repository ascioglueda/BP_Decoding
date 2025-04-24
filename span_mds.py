import random
import sys
sys.path.insert(1, '.')
from source import DawnSimVis

SOURCE = 0  # Root node ID for starting the Span MDS process

###########################################################
class Node(DawnSimVis.BaseNode):

    ###################
    def init(self):
        self.color_state = 'white'  
        self.change_color(1, 1, 1)  # Set visual color to white
        self.neighbors = set()      
        self.curr_neighs = set()    
        self.lost_neighs = set()    
        self.received = set()       
        self.recvd_cols = set()     
        self.neigh_cols = {}        
        self.neigh_spans = {}       
        self.neigh_ids = {}         
        self.neigh_finished = {}    # Dict for finished neighbors
        self.span = 0               # Set initial span to 0
        self.n_recvd = 0            # Count received messages
        self.round_recvd = False    # Flag for round messages
        self.finished = False       # Flag for round completion
        self.round_over = False     # Flag for round end
        self.sent_round = False     # Flag for sent round (unused)
        self.received_messages = [] # List for received messages
        self.undecide_count = 0     # Count undecide rounds
        self.MAX_UNDECIDE_ROUNDS = 3# Max undecide rounds (3)
        self.log(f"Node {self.id} initialized in white state.")  # Log node init

    ###################
    def run(self):
        self.cb_flood_send({'type': 'discover', 'sender': self.id})
        self.set_timer(2.0, self.start_round)

    ###################
    def start_round(self):
        if self.color_state != 'black' and not self.all_covered():
            self.received = set()
            self.recvd_cols = set()
            self.n_recvd = 0
            self.round_recvd = False
            self.finished = False
            self.received_messages = []
            self.round_over = False
            self.log(f"Node {self.id} starting a new round.")
            self.cb_flood_send({'type': 'round', 'sender': self.id, 'span': self.span, 'id': self.id, 'color': self.color_state})
            #self.sent_round = True
        else:
            self.round_over = True
            self.cb_flood_send({'type': 'done', 'sender': self.id})  # Notify neighbors
            self.log(f"Node {self.id} skips round: {self.color_state}, all covered: {self.all_covered()}")

    ###################
    def all_covered(self):
        # A node is covered if it is gray/black and all neighbors are gray/black
        # Additionally, neighbors must have no white neighbors themselves
        if self.color_state == 'white':
            return False
        for neigh in self.neighbors:
            if self.neigh_cols.get(neigh, 'white') == 'white':
                return False
            if not self.neigh_finished.get(neigh, False):
                return False
        return True

    ###################
    def cb_flood_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)
        self.log(f"Node {self.id} sent {pck['type']} message.")

    ###################
    def on_receive(self, pck):
        sender_id = pck['sender']
        msg_type = pck['type']
        self.log(f"Node {self.id} ({self.color_state}) received {msg_type} message from {sender_id}")

        self.received_messages.append(pck)

        if msg_type == 'discover':
            self.neighbors.add(sender_id)
            self.curr_neighs.add(sender_id)
            self.neigh_cols[sender_id] = 'white'
            self.neigh_spans[sender_id] = 0
            self.neigh_ids[sender_id] = sender_id
            self.neigh_finished[sender_id] = True
            self.send(sender_id, {'type': 'discover_ack', 'sender': self.id})
            self.span = len(self.neighbors) + 1

        elif msg_type == 'discover_ack':
            self.neighbors.add(sender_id)
            self.curr_neighs.add(sender_id)
            self.neigh_cols[sender_id] = 'white'
            self.neigh_spans[sender_id] = 0
            self.neigh_ids[sender_id] = sender_id
            self.neigh_finished[sender_id] = True
            self.span = len(self.neighbors) + 1

        if msg_type == 'round':
            if self.color_state != 'black':
                self.round_recvd = True
                self.received.add(sender_id)
                self.neigh_spans[sender_id] = pck['span']
                self.neigh_ids[sender_id] = pck['id']
                self.neigh_cols[sender_id] = pck['color']

                if len(self.received) == len(self.curr_neighs):
                    # Collect active neighbors (white or gray)
                    active_neighs = [neigh for neigh in self.curr_neighs if self.neigh_cols.get(neigh, 'white') != 'black']
                    if active_neighs or self.color_state != 'black':
                        # Include self if not black
                        spans = [(self.neigh_spans.get(neigh, 0), self.neigh_ids.get(neigh, neigh), self.neigh_cols.get(neigh, 'white')) for neigh in active_neighs]
                        if self.color_state != 'black':
                            spans.append((self.span, self.id, self.color_state))
                        
                        # Find max span
                        max_span = max(span for span, _, _ in spans)
                        
                        # Prioritize gray nodes with max span
                        max_span_nodes = [(span, id, col) for span, id, col in spans if span == max_span]
                        gray_max_span = [(span, id, col) for span, id, col in max_span_nodes if col == 'gray']
                        
                        if gray_max_span:
                            # Select gray node with highest ID
                            max_id = max(id for _, id, _ in gray_max_span)
                            if self.id == max_id and self.color_state == 'gray':
                                self.color_state = 'black'
                                self.change_color(0, 0, 0)
                                self.span = 0
                                self.cb_flood_send({'type': 'ch_black', 'sender': self.id})
                                self.log(f"Node {self.id} turned black (highest span gray node, span: {max_span}).")
                            else:
                                self.cb_flood_send({'type': 'undecide', 'sender': self.id})
                                self.log(f"Node {self.id} sent undecide (gray node with max span exists).")
                                if self.color_state == 'white':
                                    self.undecide_count += 1
                        else:
                            # No gray nodes; select highest ID among max span nodes
                            max_id = max(id for _, id, _ in max_span_nodes)
                            if self.id == max_id and self.color_state != 'black':
                                self.color_state = 'black'
                                self.change_color(0, 0, 0)
                                self.span = 0
                                self.cb_flood_send({'type': 'ch_black', 'sender': self.id})
                                self.log(f"Node {self.id} turned black (highest span, ID: {self.id}).")
                                self.undecide_count = 0
                            else:
                                self.cb_flood_send({'type': 'undecide', 'sender': self.id})
                                self.log(f"Node {self.id} sent undecide (span {self.span} not highest or lower ID).")
                                if self.color_state == 'white':
                                    self.undecide_count += 1
                    else:
                        # No active neighbors; gray node becomes black
                        if self.color_state == 'gray':
                            self.color_state = 'black'
                            self.change_color(0, 0, 0)
                            self.span = 0
                            self.cb_flood_send({'type': 'ch_black', 'sender': self.id})
                            self.log(f"Node {self.id} turned black (no active neighbors).")
                            self.undecide_count = 0
                        else:
                            self.cb_flood_send({'type': 'no_change', 'sender': self.id})
                            self.log(f"Node {self.id} sent no_change (no active neighbors).")
                    self.finished = True

                    # Force white node to turn black if stuck too long
                    if self.color_state == 'white' and self.undecide_count >= self.MAX_UNDECIDE_ROUNDS:
                        self.color_state = 'black'
                        self.change_color(0, 0, 0)
                        self.span = 0
                        self.cb_flood_send({'type': 'ch_black', 'sender': self.id})
                        self.log(f"Node {self.id} turned black (forced after {self.undecide_count} undecide rounds).")
                        self.undecide_count = 0
            self.set_timer(0.5, self.start_round)

        elif msg_type == 'ch_black':
            self.received.add(sender_id)
            self.recvd_cols.add('black')
            if self.neigh_cols.get(sender_id, 'white') != 'black':
                self.span -= 1
            self.neigh_cols[sender_id] = 'black'
            self.lost_neighs.add(sender_id)
            self.curr_neighs.discard(sender_id)
            self.log(f"Node {self.id} updated neighbor {sender_id} to black, new span: {self.span}.")
            if self.color_state == 'white':
                self.color_state = 'gray'
                self.change_color(0.5, 0.5, 0.5)
                self.cb_flood_send({'type': 'ch_gray', 'sender': self.id})
                self.log(f"Node {self.id} turned gray (black neighbor detected).")
                self.undecide_count = 0
            self.round_over = True
            #self.set_timer(0.5, self.start_round)

        elif msg_type == 'undecide':
            self.received.add(sender_id)
            self.n_recvd += 1
            self.log(f"Node {self.id} received undecide from {sender_id}.")
            self.round_over = True
            #self.set_timer(0.5, self.start_round)

        elif msg_type == 'ch_gray':
            self.n_recvd += 1
            if self.neigh_cols.get(sender_id, 'white') != 'gray':
                self.span -= 1
            self.neigh_cols[sender_id] = 'gray'
            self.log(f"Node {self.id} updated neighbor {sender_id} to gray, new span: {self.span}.")
            self.round_over = True
            #self.set_timer(0.5, self.start_round)

        elif msg_type == 'no_change':
            self.n_recvd += 1
            self.log(f"Node {self.id} received no_change from {sender_id}.")
            self.round_over = True
            #self.set_timer(0.5, self.start_round)

        elif msg_type == 'done':
            self.neigh_finished[sender_id] = True
            self.log(f"Node {self.id} marked neighbor {sender_id} as finished.")
            #self.set_timer(0.5, self.start_round)

        # Phase 1 Check: Decide color based on received messages
        if not self.finished and self.round_recvd and len(self.received) == len(self.curr_neighs):
            if 'black' in self.recvd_cols and self.color_state == 'white':
                self.color_state = 'gray'
                self.change_color(0.5, 0.5, 0.5)
                self.cb_flood_send({'type': 'ch_gray', 'sender': self.id})
                self.log(f"Node {self.id} turned gray (black neighbor detected).")
                self.undecide_count = 0
            else:
                self.cb_flood_send({'type': 'no_change', 'sender': self.id})
                self.log(f"Node {self.id} sent no_change (no color change required).")
            self.finished = True

        # Round Completion Check
        if self.finished and self.n_recvd >= len(self.curr_neighs):
            self.round_over = True
            self.curr_neighs -= self.lost_neighs
            self.n_recvd = 0
            self.round_recvd = False
            self.finished = False
            self.received = set()
            self.recvd_cols = set()
            self.lost_neighs = set()
            self.received_messages = []
            self.log(f"Node {self.id} completed a round.")

            #if self.all_covered():
                #self.log(f"Node {self.id} finished: All nodes covered.")
                #self.cb_flood_send({'type': 'done', 'sender': self.id})
            #else:
                #self.set_timer(0.5, self.start_round)

    ###################
    def is_terminated(self):
           
        if self.color == 'white':
            return False
        for nid in self.curr_neighs:
            if self.neigh_cols.get(nid, 'white') == 'white':
                return False
        return True

    def finish(self):
        self.log(f"Node {self.id} finished in {self.color_state} state, span: {self.span}.")

###########################################################
def create_network():
    for x in range(10):
        for y in range(10):
            px = 50 + x * 60 + random.uniform(-20, 20)
            py = 50 + y * 60 + random.uniform(-20, 20)
            sim.add_node(Node, pos=(px, py), tx_range=75)

# Set up the simulation
sim = DawnSimVis.Simulator(
    duration=20000,
    timescale=3,
    visual=True,
    terrain_size=(650, 650),
    title='Span MDS Simulation'
)

# Create the network
create_network()

# Start the simulation
sim.run()