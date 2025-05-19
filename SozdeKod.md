#--- Belief Propagation Decoding ---
1: input: k, n
2: data ← random_binary_array(k)
3: dist ← robust_soliton_distribution(k)

# --- ENCODE ---
4: for i = 1 to n do
5:     degree ← sample_degree(dist)
6:     indices ← select_random_indices(k, degree)
7:     symbol ← XOR(data[indices])
8:     send (symbol, indices) to node i

# --- COLLECT ---
9: received ← collect_all(symbol, indices) pairs
10: known ← [-1] * k
11: queue ← packets with 1 unknown index

# --- DECODE ---
12: while queue not empty do
13:     (i, val) ← queue.pop()
14:     if known[i] ≠ -1: continue
15:     known[i] ← val
16:     for packet in received do
17:         if i in packet.indices:
18:             remove i from indices
19:             packet.symbol ← packet.symbol XOR val
20:             if len(indices) == 1: queue.push()

# --- RESULT ---
21: result ← fill unknowns with 0
22: success ← count_correct_bits(result, data)
23: print("Success rate:", success, "%")


