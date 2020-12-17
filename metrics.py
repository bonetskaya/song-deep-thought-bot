def cer(trg, hyp):
    prev = None
    curr = [0] + list(range(1, len(hyp) + 1))
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(hyp) + 1)]
    for x in range(1, len(trg) + 1):
        prev, curr = curr, [x] + ([None] * len(hyp))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(hyp))
        for y in range(1, len(hyp) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(trg[x - 1] != hyp[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(trg[x - 1] != hyp[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    cer_s, cer_i, cer_d = curr_ops[len(hyp)]
    cer_n = len(trg)
    return (100.0 * (cer_s + cer_i + cer_d)) / cer_n