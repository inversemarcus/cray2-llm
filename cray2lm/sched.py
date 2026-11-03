"""List scheduler for straight-line CRAY-2 vector code.

Ops carry virtual registers.  The scheduler mirrors cray2sim's timing rules
(in-order issue, no chaining, unit reservation, register write/read
reservations) to choose an emission order and physical V/S registers.
It changes only instruction order and register names, never arithmetic.
"""

VL = 64
TRACE = False


class Op:
    def __init__(self, kind, fmt, dst=None, srcs=(), after=(), parcels=1, prio_bias=0):
        self.kind, self.fmt, self.dst, self.srcs = kind, fmt, dst, list(srcs)
        self.after = list(after)        # extra ordering deps (op objects)
        self.parcels = parcels
        self.prio_bias = prio_bias
        self.users = []
        self.done = False


class Timing:
    v_lat = {'L': 22, 'I': 22, 'F': 22, 'M': 22}
    gap = 2
    cm_lat = 59
    lm_lat = 4
    a_lat = 4
    issue = 2


def schedule(ops, vregs, sregs, pinned_in=None, pinned_out=None, T=Timing, window=16, reserve=2, rng=None, jitter=0):
    """ops: list of Op in a valid sequential order.  vregs/sregs: free physical
    register numbers.  pinned_in: {vname: phys} values live on entry;
    pinned_out: {vname: phys} required final location.
    Returns list of CAL lines and the predicted cycle count."""
    pinned_in = dict(pinned_in or {})
    pinned_out = dict(pinned_out or {})
    producer = {}
    for op in ops:
        if op.dst:
            producer[op.dst] = op
    for op in ops:
        for s in op.srcs:
            if s in producer:
                producer[s].users.append(op)
    # predecessor sets
    for op in ops:
        op.preds = [producer[s] for s in op.srcs if s in producer] + op.after
    # remaining-use counts per value
    uses = {}
    for op in ops:
        for s in op.srcs:
            uses[s] = uses.get(s, 0) + 1
    # priority: latency-weighted longest path to end
    def lat(op):
        return {'L': 86, 'I': 86, 'F': 86, 'M': 86, 'VLD': 130, 'SLD': 60, 'VLM': 70, 'VSM': 70, 'SLM': 5, 'A': 4, 'X': 2}[op.kind]
    prio = {}
    for op in reversed(ops):
        succ = [u for u in ops if op in u.preds]
        prio[op] = lat(op) + max([prio[u] for u in succ], default=0) + op.prio_bias + (rng.uniform(0, jitter) if rng else 0)
    # state
    t = 0
    unit_free = {'L': 0, 'I': 0, 'F': 0, 'M': 0, 'P': 0, 'LMP': 0}
    loc = {}                 # value -> ('V'|'S', phys)
    ready = {}               # value -> time written
    phys_rd = {}             # ('V',p) -> reads complete
    phys_wr = {}             # ('V',p) -> write complete
    holder = {}              # ('V',p) -> value currently held
    for v, p in pinned_in.items():
        loc[v] = ('V' if v.startswith('v') else 'S', p)
        ready[v] = 0
        holder[loc[v]] = v
    free = {'V': [p for p in vregs if ('V', p) not in holder],
            'S': [p for p in sregs if ('S', p) not in holder]}
    out = []
    pending = list(ops)
    issued = set()

    def regclass(v):
        return 'V' if v.startswith('v') else 'S'

    def earliest(op, dphys):
        e = t
        for s in op.srcs:
            e = max(e, ready.get(s, 0))
            if s in loc and loc[s][0] == 'V':
                e = max(e, phys_rd.get(loc[s], 0))   # a source register is reserved while read
        k = op.kind
        if k in ('L', 'I', 'F', 'M'):
            e = max(e, unit_free[k])
        if k in ('VLD', 'SLD'):
            e = max(e, unit_free['P'])
        if k in ('VLM', 'VSM'):
            e = max(e, unit_free['LMP'])
        if dphys is not None:
            e = max(e, phys_rd.get(dphys, 0), phys_wr.get(dphys, 0))
        return e

    while pending:
        best = None
        oldest_alloc = {}
        for op in pending:
            if op.dst:
                oldest_alloc.setdefault(regclass(op.dst), op)
        for op in pending[:window]:
            if any(p not in issued for p in op.preds):
                continue
            if op.dst and op.dst not in pinned_out:
                rc = regclass(op.dst)
                need = 1 if oldest_alloc.get(rc) is op else 1 + (reserve if rc == 'V' else 1)
                if len(free[rc]) < need:
                    continue
            dphys = None
            if op.dst:
                rc = regclass(op.dst)
                if op.dst in pinned_out and pinned_out[op.dst] is not None:
                    want = (rc, pinned_out[op.dst])
                    if want in holder:
                        continue
                    dphys = want
                else:
                    cands = [p for p in free[rc]]
                    if not cands:
                        continue
                    # choose the candidate free earliest
                    dphys = min(((rc, p) for p in cands), key=lambda q: max(phys_rd.get(q, 0), phys_wr.get(q, 0)))
            e = earliest(op, dphys)
            key = (e, -prio[op])
            if best is None or key < best[0]:
                best = (key, op, dphys)
        if best is None:
            raise RuntimeError('scheduler deadlock; pending %d' % len(pending))
        (e, _), op, dphys = best
        # bind and emit
        names = {}
        for s in op.srcs:
            names[s] = '%s%d' % loc[s]
        if op.dst:
            names[op.dst] = '%s%d' % dphys
        out.append(op.fmt.format(**names) + ("   ;t=%d" % e if TRACE else ""))
        k = op.kind
        if k in ('L', 'I', 'F', 'M'):
            unit_free[k] = e + VL + T.gap
            for s in op.srcs:
                if regclass(s) == 'V':
                    phys_rd[loc[s]] = max(phys_rd.get(loc[s], 0), e + VL)
            ready[op.dst] = e + T.v_lat[k] + VL
            phys_wr[dphys] = ready[op.dst]
        elif k == 'VLD':
            unit_free['P'] = e + VL + 8
            ready[op.dst] = e + VL + T.cm_lat + 3
            phys_wr[dphys] = ready[op.dst]
        elif k == 'VLM':
            unit_free['LMP'] = e + T.lm_lat + VL
            ready[op.dst] = e + T.lm_lat + VL
            phys_wr[dphys] = ready[op.dst]
        elif k == 'VSM':
            unit_free['LMP'] = e + T.lm_lat + VL
            for s_ in op.srcs:
                phys_rd[loc[s_]] = max(phys_rd.get(loc[s_], 0), e + VL)
        elif k == 'SLD':
            unit_free['P'] = max(unit_free['P'], e + 1)
            ready[op.dst] = e + T.cm_lat + 3
        elif k == 'SLM':
            ready[op.dst] = e + T.lm_lat
        elif k == 'A':
            if op.dst:
                ready[op.dst] = e + T.a_lat
        if op.dst:
            loc[op.dst] = dphys
            holder[dphys] = op.dst
            rc = dphys[0]
            if dphys[1] in free[rc]:
                free[rc].remove(dphys[1])
        # release sources whose uses are exhausted
        for s in op.srcs:
            uses[s] -= 1
            if uses[s] == 0 and s not in pinned_out:
                ph = loc[s]
                if holder.get(ph) == s:
                    del holder[ph]
                    free[ph[0]].append(ph[1])
        issued.add(op)
        pending.remove(op)
        t = e + T.issue + max(0, op.parcels - 2)
    return out, t
