"""common_lib: F2 linear algebra, gate/gadget model, scheduler, decompositions, cost model rules."""
import itertools, math
import numpy as np
from collections import Counter, defaultdict

# ---------------------------------------------------------------- F2 algebra
def f2_rref(M):
    """Row-reduce a uint8 matrix over F2. Returns (R, pivot_cols)."""
    M = (np.array(M, dtype=np.uint8) & 1).copy()
    rows, cols = M.shape
    piv = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if len(nz) == 0:
            continue
        p = r + nz[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
        others = np.nonzero(M[:, c])[0]
        others = others[others != r]
        if len(others):
            M[others] ^= M[r]
        piv.append(c)
        r += 1
    return M[:r], piv


def f2_rank(M):
    if len(M) == 0:
        return 0
    return len(f2_rref(M)[1])


def f2_nullspace(M, n=None):
    """Basis (rows) of {x : M x = 0} over F2."""
    M = np.array(M, dtype=np.uint8)
    if M.size == 0:
        return np.eye(n, dtype=np.uint8)
    n = M.shape[1]
    Rm, piv = f2_rref(M)
    free = [c for c in range(n) if c not in set(piv)]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        for i, p in enumerate(piv):
            if Rm[i, f]:
                v[p] = 1
        basis.append(v)
    return np.array(basis, dtype=np.uint8).reshape(len(basis), n)


def f2_in_rowspace(Rref, piv, v):
    """Is v in the row space of the RREF matrix Rref (pivots piv)?"""
    v = np.array(v, dtype=np.uint8) & 1
    w = v.copy()
    for i, p in enumerate(piv):
        if w[p]:
            w ^= Rref[i]
    return not w.any()


def f2_solve(A, b):
    """Solve x A = b (x row vector, A rows = generators). Returns x or None."""
    A = np.array(A, dtype=np.uint8)
    m, n = A.shape
    aug = np.concatenate([A, np.eye(m, dtype=np.uint8)], axis=1)
    Rm, piv = f2_rref(aug)
    w = np.concatenate([np.array(b, dtype=np.uint8) & 1, np.zeros(m, dtype=np.uint8)])
    for i, p in enumerate(piv):
        if p >= n:
            break
        if w[p]:
            w ^= Rm[i]
    if w[:n].any():
        return None
    return w[n:]




import itertools
from collections import Counter, defaultdict

ARITY = {'INIT_X': 1, 'INIT_Z': 1, 'MEAS_X': 1, 'MEAS_Z': 1, 'H': 1, 'X': 1, 'Z': 1, 'S': 1, 'SDG': 1,
         'T': 1, 'TDG': 1, 'CNOT': 2, 'CZ': 2, 'CS': 2, 'CSDG': 2, 'CCZ': 3}
NONCLIFFORD = {'T', 'TDG', 'CS', 'CSDG', 'CCZ'}


# ------------------------------------------------------------------ minimum-weight logical (MILP)
def min_weight_odd(Hc, lvec, cols):
    """min |x| over x in F2^cols with Hc x = 0 and lvec . x = 1 (exact MILP, HiGHS)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    H = np.array(Hc, dtype=float)[:, cols]
    H = H[np.any(H, axis=1)]
    l = np.array(lvec, dtype=float)[cols]
    m, n = H.shape
    # variables: x (n, binary), y (m, integer), z (1, integer)
    nv = n + m + 1
    c = np.concatenate([np.ones(n), np.zeros(m + 1)])
    A1 = np.concatenate([H, -2 * np.eye(m), np.zeros((m, 1))], axis=1)
    A2 = np.concatenate([l[None, :], np.zeros((1, m)), -2 * np.ones((1, 1))], axis=1)
    cons = [LinearConstraint(A1, 0, 0), LinearConstraint(A2, 1, 1)]
    integrality = np.ones(nv)
    ub = np.concatenate([np.ones(n), np.full(m + 1, np.inf)])
    res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(np.zeros(nv), ub))
    assert res.status == 0, res.message
    x = np.round(res.x[:n]).astype(np.uint8)
    out = np.zeros(len(lvec), dtype=np.uint8)
    out[cols] = x
    return int(x.sum()), out


# ------------------------------------------------------------------ gadgets
def pauli_check_gadget(anc, xq=(), zq=()):
    """Measure X(xq) Z(zq) with ancilla anc (standard CNOT circuit)."""
    g = [('INIT_X', anc)]
    g += [('CNOT', anc, q) for q in sorted(xq)]           # ancilla controls -> X on data
    g += [('CZ', anc, q) for q in sorted(zq)]             # controlled Z on data
    g += [('MEAS_X', anc)]
    return g


def z_check_gadget(anc, zq):
    g = [('INIT_Z', anc)] + [('CNOT', q, anc) for q in sorted(zq)] + [('MEAS_Z', anc)]
    return g


def clifford_x_gadget(anc, parts):
    """Measure a product of X-type Clifford operators S = S_1 S_2 ... with
    S_k = X(s_k) prod CZ(pairs_k): ancilla |+>, controlled-S (= CNOTs from anc and
    CCZ(anc,i,j) for each CZ pair; rightmost factor first), measure X.
    parts: list of (xqubits, czpairs) in operator order S_1 S_2 ..."""
    g = [('INIT_X', anc)]
    for xq, pairs in reversed(parts):
        g += [('CCZ', anc, i, j) for (i, j) in pairs]      # CZ part acts first within S_k
        g += [('CNOT', anc, q) for q in sorted(xq)]
    g += [('MEAS_X', anc)]
    return g


# ------------------------------------------------------------------ scheduling
def asap_layers(gates):
    """Greedy ASAP layering of a gate list respecting qubit order (no reordering of gates that
    share a qubit).  Returns list of layers (lists of gates)."""
    ready = defaultdict(int)
    layers = []
    for gt in gates:
        qs = gt[1:]
        t = max(ready[q] for q in qs)
        while len(layers) <= t:
            layers.append([])
        layers[t].append(gt)
        for q in qs:
            ready[q] = t + 1
    return layers


def gadget_support(gad):
    s = set()
    for gt in gad:
        s |= set(gt[1:])
    return s


def disjoint_groups(gadgets):
    """Greedy colouring: partition gadgets so that gadgets in a group touch disjoint qubits.
    Measurements within a group are then exactly simultaneous and the sequential-group
    schedule is valid for commuting (on the relevant subspace) generators."""
    groups = []          # list of (used_qubits, [gadgets])
    order = sorted(range(len(gadgets)), key=lambda i: -len(gadget_support(gadgets[i])))
    for i in order:
        sup = gadget_support(gadgets[i])
        for used, members in groups:
            if not (used & sup):
                used |= sup
                members.append(gadgets[i])
                break
        else:
            groups.append((set(sup), [gadgets[i]]))
    return [m for _, m in groups]


def layer_duration(layer, dur):
    return max(dur[g[0]] for g in layer)


def circuit_time(layers, dur):
    return sum(layer_duration(l, dur) for l in layers)


def round_schedule(gadgets, dur):
    """Upper-bound schedule of one round: disjoint groups executed sequentially, each group
    ASAP-layered.  Returns dict with depth (layers), time (weighted by dur), groups."""
    groups = disjoint_groups(gadgets)
    depth = 0
    time = 0
    for grp in groups:
        gl = asap_layers([gt for gad in grp for gt in gad])
        depth += len(gl)
        time += circuit_time(gl, dur)
    return {'groups': len(groups), 'depth_ub': depth, 'time_ub': time}


def round_lower_bound(gadgets, dur):
    """Lower bound on the duration of one round: every qubit executes its gates sequentially,
    so time >= max_q sum of durations of gates acting on q."""
    busy = defaultdict(float)
    cnt = defaultdict(int)
    for gad in gadgets:
        for gt in gad:
            for q in gt[1:]:
                busy[q] += dur[gt[0]]
                cnt[q] += 1
    return {'depth_lb': max(cnt.values()), 'time_lb': max(busy.values())}


def gate_counts(gates):
    c = Counter(g[0] for g in gates)
    return dict(c)


# ------------------------------------------------------------------ decompositions
# Explicit Clifford+T decompositions (verified numerically in the validation notebook).
def ccz_to_clifford_t(a, b, c):
    """CCZ with 7 T/T^dag and 6 CNOT (standard Toffoli decomposition, Nielsen & Chuang Fig. 4.9,
    with the target Hadamards removed)."""
    return [('CNOT', b, c), ('TDG', c), ('CNOT', a, c), ('T', c), ('CNOT', b, c), ('TDG', c),
            ('CNOT', a, c), ('T', b), ('T', c), ('CNOT', a, b), ('T', a), ('TDG', b), ('CNOT', a, b)]


def cs_to_clifford_t(a, b):
    """CS = diag(1,1,1,i) = T_a T_b CNOT(a,b) T^dag_b CNOT(a,b)."""
    return [('T', a), ('T', b), ('CNOT', a, b), ('TDG', b), ('CNOT', a, b)]


def csdg_to_clifford_t(a, b):
    return [('TDG', a), ('TDG', b), ('CNOT', a, b), ('T', b), ('CNOT', a, b)]


def decompose(gates):
    out = []
    for g in gates:
        if g[0] == 'CCZ':
            out += ccz_to_clifford_t(*g[1:])
        elif g[0] == 'CS':
            out += cs_to_clifford_t(*g[1:])
        elif g[0] == 'CSDG':
            out += csdg_to_clifford_t(*g[1:])
        else:
            out.append(g)
    return out


DUR_UNIT = {k: 1 for k in ARITY}            # every primitive = 1 time unit (ASSUMPTION, parametrised)


# ------------------------------------------------- commuting-class scheduler
DIAG_GATES = {'CZ', 'CCZ', 'CS', 'CSDG', 'T', 'TDG', 'S', 'SDG', 'Z'}


def gate_roles(gt):
    """Per-qubit role of an entangling/1q unitary: 'Z' (diagonal in Z basis), 'X' (X-type)."""
    name = gt[0]
    if name in DIAG_GATES:
        return {q: 'Z' for q in gt[1:]}
    if name == 'CNOT':
        return {gt[1]: 'Z', gt[2]: 'X'}
    if name == 'X':
        return {gt[1]: 'X'}
    return {q: '?' for q in gt[1:]}      # H, INIT, MEAS: never shared between gadgets


def gadget_roles(gad):
    roles = {}
    for gt in gad:
        for q, r in gate_roles(gt).items():
            if roles.get(q, r) != r:
                roles[q] = '?'
            else:
                roles[q] = r
    return roles


def commuting_classes(gadgets, n_orders=6, seed=0):
    """Best (fewest classes) of several greedy colourings (different gadget orders)."""
    import random
    best = None
    orders = [list(range(len(gadgets)))]
    orders.append(sorted(orders[0], key=lambda i: tuple(sorted(set(gadget_roles(gadgets[i]).values())))))
    rnd = random.Random(seed)
    for _ in range(n_orders):
        o = orders[0][:]
        rnd.shuffle(o)
        orders.append(o)
    for o in orders:
        cl = _commuting_classes_greedy([gadgets[i] for i in o])
        if best is None or len(cl) < len(best):
            best = cl
    return best


def _commuting_classes_greedy(gadgets):
    """Greedy colouring of the gadget conflict graph.  Two gadgets are compatible if on every
    shared qubit both act with the same role ('Z' or 'X'); then all their gates commute and the
    joint circuit is equivalent to measuring them one after another (sufficient condition)."""
    roles = [gadget_roles(g) for g in gadgets]
    classes = []     # list of (role_map, [idx])
    for i, rm in enumerate(roles):
        placed = False
        for cmap, members in classes:
            ok = all(cmap.get(q, r) == r and r != '?' for q, r in rm.items() if q in cmap)
            if ok:
                for q, r in rm.items():
                    cmap[q] = r
                members.append(i)
                placed = True
                break
        if not placed:
            classes.append((dict(rm), [i]))
    return [[gadgets[i] for i in m] for _, m in classes]


def class_layers(gadgets):
    """Layering of one commuting class: init layer, greedy layering of the (mutually commuting)
    entangling gates, measurement layer."""
    inits = [g for gad in gadgets for g in gad if g[0].startswith('INIT')]
    meas = [g for gad in gadgets for g in gad if g[0].startswith('MEAS')]
    body = [g for gad in gadgets for g in gad if not (g[0].startswith('INIT') or g[0].startswith('MEAS'))]
    # greedy: most-constrained gates first
    body = sorted(body, key=lambda g: -len(g))
    layers = []
    busy = []
    for g in body:
        qs = set(g[1:])
        for li, b in enumerate(busy):
            if not (b & qs) and layers[li][0][0] == g[0]:     # keep one gate type per layer
                b |= qs
                layers[li].append(g)
                break
        else:
            busy.append(set(qs))
            layers.append([g])
    return ([inits] if inits else []) + layers + ([meas] if meas else [])


def round_schedule_classes(gadgets, dur):
    classes = commuting_classes(gadgets)
    depth = 0
    time = 0
    for c in classes:
        ls = class_layers(c)
        depth += len(ls)
        time += circuit_time(ls, dur)
    return {'classes': len(classes), 'depth_ub': depth, 'time_ub': time}


import math
from collections import Counter

SOURCE_LIT = 'literature'
SOURCE_COUNT = 'counted (explicit circuit)'
SOURCE_ASSUME = 'assumption (parameter)'
SOURCE_UNRES = 'unresolved (excluded / illustrative)'

DEFAULT_PARAMS = dict(
    kappa0=1.0,      # FSC |+_L> preparation rounds = ceil(kappa0*d)   [ASSUMPTION; Kobayashi: not specified]
    kappa_in=1.0,    # switch-in rounds            = ceil(kappa_in*d)  [ASSUMPTION; Kobayashi: O(d)]
    kappa_out=1.0,   # FSC EC rounds after return  = ceil(kappa_out*d) [ASSUMPTION; Kobayashi: O(d)]
    tau_rounds=1.0,  # Litinski: 1 time step = tau_rounds*d code cycles [LITERATURE: d cycles]
    t_inj_FT=6,      # Litinski App. A (Li's injection): 3 + 3 code cycles, d-independent [LITERATURE]
    t_inj_NL=1,      # noiseless-minimal injection: 1 round [ASSUMPTION]
)


def rounds_for(model, kappa, d):
    if model == 'NL_min':
        return 1
    return int(math.ceil(kappa * d))



# ----------------------------------------------------------------- summaries
def summarize(rows):
    tot = Counter()
    for r in rows:
        for k, v in r['gates'].items():
            tot[k] += v
    peak = max(r['qubits'] for r in rows)
    rounds = sum(r['rounds'] for r in rows)
    tl = sum(r['time_lb'] for r in rows if r['time_lb'] is not None)
    tu = sum(r['time_ub'] for r in rows if r['time_ub'] is not None)
    v_round = sum(r['qubits'] * r['rounds'] for r in rows)
    v_tl = sum(r['qubits'] * r['time_lb'] for r in rows if r['time_lb'] is not None)
    v_tu = sum(r['qubits'] * r['time_ub'] for r in rows if r['time_ub'] is not None)
    nonclif = {k: v for k, v in tot.items() if k in NONCLIFFORD}
    tcount = 7 * tot.get('CCZ', 0) + 3 * (tot.get('CS', 0) + tot.get('CSDG', 0)) + tot.get('T', 0) + tot.get('TDG', 0)
    return dict(peak_qubits=peak, rounds=rounds, time_lb=tl, time_ub=tu, V_qubit_round=v_round,
                V_qubit_time_lb=v_tl, V_qubit_time_ub=v_tu, V_peak_x_rounds_UB=peak * rounds,
                gates=dict(tot), nonclifford=nonclif, T_count_if_decomposed=tcount)


def uncounted_costs(protocol, comparison):
    """Costs that are known to exist but are not included in the totals (listed, never summed as 0)."""
    out = [('classical decoding / control latency', 'both comparisons, both protocols')]
    if protocol == 'Kobayashi_D4':
        out.append(('JIT-decoder classical processing', 'not modelled'))
        if comparison == 'B':
            out.append(('FSC -> standard rotated surface-code patch conversion', 'UNRESOLVED: no protocol constructed'))
    else:
        out.append(('twist-based Z_m (x) Y_0 seam checks between |m> and |0>', 'O(d) measurement qubits, not constructed'))
    return out


# ------------------------------------------------------------------ table helpers
def jsonify_columns(df, cols=('gates', 'nonclifford')):
    """Store dict-valued columns as JSON strings (same format as the saved CSV files)."""
    import json
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda g: g if isinstance(g, str) else json.dumps({k: float(v) for k, v in g.items()}))
    return df


def identify_polynomial(values, npts=4):
    """Interpolate a polynomial in d through the first npts points (degree <= npts-1) and report
    whether it reproduces ALL given points exactly.  values: dict d -> integer.  Returns (expr, ok)."""
    import sympy
    x = sympy.Symbol('d')
    ds = sorted(values)
    if len(ds) <= npts:
        return None, False
    p = sympy.interpolate([(dd, int(round(values[dd]))) for dd in ds[:npts]], x)
    ok = all(sympy.nsimplify(p.subs(x, dd) - int(round(values[dd]))) == 0 for dd in ds)
    return sympy.expand(p), ok
