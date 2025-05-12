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

        self.log(f"Node {self.id} started in IDLE state.")

    def run(self):
        if self.id == SOURCE:
            self.change_color(1, 0, 0)  # Kırmızı: Root
            self.currstate = 'XPLORING'
            self.sent_probes = True
            self.log(f"Root node {self.id} sent probe message.")
            self.cb_flood_send({'type': 'probe', 'sender': self.id})

            probabilities = robust_soliton_distribution(k)
            selected_indices, encoded_symbol = encode_lt(data, probabilities)
            self.log(f"Node {self.id} encoded: indices = {selected_indices}")
            received_packets = {
                'type': 'encoded',
                'sender': self.id,
                'indices': (selected_indices, encoded_symbol)
            }
            self.set_timer(1, self.cb_msg_send, received_packets)

    def cb_flood_send(self, pck):
        self.send(DawnSimVis.BROADCAST_ADDR, pck)

    def cb_msg_send(self, pck):
        if self.parent is not None:
            self.send(self.parent, pck)
        for child in self.children:
            self.send(child, pck)
        self.log(f"Node {self.id} sent packet to parent and children.")

    def on_receive(self, pck):
        sender_id = pck['sender']
        msg_type = pck['type']
        self.log(f"Node {self.id} received '{msg_type}' from {sender_id}.")

        # Spanning Tree oluşturma
        if self.currstate == 'IDLE' and msg_type == 'probe':
            self.parent = sender_id
            self.currstate = 'XPLORING'
            self.change_color(0, 0, 1)  # Mavi: XPLORING
            self.log(f"Node {self.id} set {self.parent} as parent.")
            self.set_timer(1.5, self.cb_flood_send, {'type': 'probe', 'sender': self.id})
            self.sent_probes = True

        elif self.currstate == 'XPLORING':
            if msg_type == 'probe' and sender_id != self.parent:
                self.send(sender_id, {'type': 'reject', 'sender': self.id})
                self.log(f"Node {self.id} sent reject to {sender_id}.")
            elif msg_type == 'ack':
                self.received_ack.add(sender_id)
                self.children.append(sender_id)  # 'childs' yerine 'children' kullandık
                self.log(f"Node {self.id} added {sender_id} as child.")
            elif msg_type == 'reject':
                self.received_reject.add(sender_id)
                self.others.add(sender_id)
                self.log(f"Node {self.id} marked {sender_id} as other.")

            if self.sent_probes and (self.received_ack or self.received_reject):
                if self.id != SOURCE:
                    self.send(self.parent, {'type': 'ack', 'sender': self.id})
                    self.change_color(0, 1, 0)  # Yeşil: TERM
                    self.currstate = 'TERM'
                    self.log(f"Node {self.id} transitioning to TERM state.")

                    probabilities = robust_soliton_distribution(k)
                    selected_indices, encoded_symbol = encode_lt(data, probabilities)
                    self.set_timer(2 , self.cb_msg_send, {
                        'type': 'encoded',
                        'sender': self.id,
                        'indices': (selected_indices, encoded_symbol)
                    })
                else:
                    self.currstate = 'TERM'
                    self.log(f"Root node {self.id} finished.")

        # Encoded paketleri al ve SOURCE'a gönder
        if msg_type == 'encoded' and 'indices' in pck:
            indices_data = pck['indices']
            if len(indices_data) == 2:
                selected_indices, encoded_symbol = indices_data
                self.received_packets.append((selected_indices.copy(), encoded_symbol))
                self.log(f"Node {self.id} received packet from {sender_id}: {selected_indices}")

                # Paketleri SOURCE'a gönder
                if self.id != SOURCE:
                    self.send(SOURCE, {
                        'type': 'encoded',
                        'sender': self.id,
                        'indices': (selected_indices.copy(), encoded_symbol)
                    })
                # Parent-child ilişkisini kur
                if sender_id not in self.children and sender_id != self.parent:
                    if self.parent is None or sender_id == self.parent:
                        self.parent = sender_id
                        if self.parent < len(self.sim.nodes):
                            parent_node = self.sim.nodes[self.parent]
                            if parent_node:
                                parent_node.children.append(self.id)

    def finish(self):
        #if self.parent is not None:
            #self.log(f'Node {self.id} -> Parent: {self.parent}')
        #if self.children:
            #self.log(f'Node {self.id} -> Children: {self.children}')
        # Sadece SOURCE düğümü decoding yapar ve sonucu loglar
        if self.id == SOURCE and self.received_packets:
            decoded_data = decode_lt(self.received_packets, k)
            correct_bits = np.sum(decoded_data == data)
            self.log(f"SOURCE Node decoded {correct_bits}/{k} bits correctly.")