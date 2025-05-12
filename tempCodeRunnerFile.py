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
                self.childs.add(sender_id)
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

                    # TERM durumuna geçtikten sonra sadece 1 encoded paket gönder
                    probabilities = robust_soliton_distribution(k)
                    selected_indices, encoded_symbol = encode_lt(data, probabilities)
                    self.set_timer(2, self.cb_msg_send, {
                        'type': 'encoded',
                        'sender': self.id,
                        'indices': (selected_indices, encoded_symbol)
                    })
                else:
                    self.currstate = 'TERM'
                    self.log(f"Root node {self.id} finished.")

        # Encoded paketleri al ve SOURCE'a gönder
        if msg_type == 'encoded' and 'indices' in pck and isinstance(pck['indices'], tuple):
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