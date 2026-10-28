"""noisy_d4_lib: circuit-level noisy simulation of the Kobayashi-Zhu-Hsin D4 code switching protocol.

Non-Clifford simulation method: Clifford-stabilizer simulation with Pauli proxies
(Y. Takada, S. D. Bartlett, D. J. Williamson, arXiv:2609.16929, Algorithm 2 (Sec. VII D) and Sec. VIII).
For every Monte Carlo shot the noisy non-Clifford circuit is mapped to a Clifford circuit (the
"Pauli-proxy circuit") with the same measurement statistics:
  * every diagonal third-level gate D (CCZ of the D4 gadgets, CS / CS^dag / T^dag of U) is removed;
  * the X component e of the sampled Pauli errors and of applied X corrections, propagated forward
    through the Clifford gates of the true circuit, is tracked ("continuing errors"); where e meets
    D the diagonal Clifford gate C_e = D^dag X^e D X^e (a product of S, Z, CZ) is inserted
    (= the controlled Pauli dressing Lambda(D_U(E)) of Takada Fig. 14 / Eq. (165));
  * Z components commute with every diagonal gate and need no propagation.
All Clifford gates, sampled Pauli errors and inserted dressings are executed by stim's TableauSimulator.

Requires (as globals, embedded before this code): numpy as np, stim, common_lib, d4_lib.
"""
import itertools
import numpy as np

# ------------------------------------------------------------------ gate algebra
DIAG_THETA = {                     # phase polynomials (units of pi/4) of the diagonal 3rd-level gates
    'CCZ': lambda x: 4 * x[0] * x[1] * x[2],
    'NCCZ': lambda x: 4 * x[0] * x[1] * x[2],      # virtual noiseless CCZ (deferred switch-out correction)
    'CS': lambda x: 2 * x[0] * x[1],
    'CSDG': lambda x: 6 * x[0] * x[1],
    'T': lambda x: x[0],
    'TDG': lambda x: 7 * x[0],
}
NONCLIFF = set(DIAG_THETA)
_DRESS = {}


def diag_clifford_gates(delta, k):
    """delta: dict bits -> phase exponent mod 8 (even).  Returns gates ('S'|'Z'|'SDG', i), ('CZ', i, j)
    (local indices) equal to diag(omega^delta) up to a global phase; asserts the result is exact."""
    c0 = delta[(0,) * k]
    a, gates = [], []
    for i in range(k):
        e = tuple(int(j == i) for j in range(k))
        ai = (delta[e] - c0) % 8
        assert ai % 2 == 0, 'dressing is not Clifford'
        a.append(ai // 2)
        if ai // 2:
            gates.append(({1: 'S', 2: 'Z', 3: 'SDG'}[ai // 2], i))
    b = {}
    for i, j in itertools.combinations(range(k), 2):
        e = tuple(int(m in (i, j)) for m in range(k))
        v = (delta[e] - c0 - 2 * a[i] - 2 * a[j]) % 8
        assert v in (0, 4), 'dressing has a non-Clifford quadratic term'
        b[(i, j)] = v // 4
        if v:
            gates.append(('CZ', i, j))
    for x in itertools.product((0, 1), repeat=k):
        pred = c0 + sum(2 * a[i] * x[i] for i in range(k)) + sum(4 * b[ij] * x[ij[0]] * x[ij[1]] for ij in b)
        assert (pred - delta[x]) % 8 == 0, 'dressing has a cubic term'
    return gates


def dressing(name, e):
    """C_e = D^dag X^e D X^e for diagonal gate `name` and X pattern e (tuple of bits), local indices."""
    key = (name, e)
    if key not in _DRESS:
        th = DIAG_THETA[name]
        k = len(e)
        delta = {x: (th(tuple(xi ^ ei for xi, ei in zip(x, e))) - th(x)) % 8
                 for x in itertools.product((0, 1), repeat=k)}
        _DRESS[key] = diag_clifford_gates(delta, k)
    return _DRESS[key]


# ------------------------------------------------------------------ noise model
class NoiseModel:
    """Standard circuit-level Pauli noise (Takada et al. App. A 2) plus optional idling noise:
    after every k-qubit unitary a k-qubit depolarizing channel (probability p; p_ccz for CCZ, p_nc for
    the gates of U), preparation followed by a flip (p_reset), measurement preceded by a flip (p_meas),
    every idle qubit of a time step depolarized with p_idle."""

    def __init__(self, p, p_meas=None, p_reset=None, p_idle=None, p_ccz=None, p_nc=None):
        self.p = p
        self.p_meas = p if p_meas is None else p_meas
        self.p_reset = p if p_reset is None else p_reset
        self.p_idle = p if p_idle is None else p_idle
        self.p_ccz = p if p_ccz is None else p_ccz
        self.p_nc = p if p_nc is None else p_nc

    def locations(self, ops, idle):
        """Noise locations of one time step: list of (when, kind, qubits, prob), when in {'pre','post'}."""
        loc = []
        for op in ops:
            nm, qs = op[0], op[1:]
            if nm in ('NCCZ', 'MZ0'):
                continue
            if nm == 'FLIPX':
                loc.append(('pre', 'X', qs, self.p_meas))
            elif nm == 'MZ':
                loc.append(('pre', 'X', qs, self.p_meas))
            elif nm == 'MX':
                loc.append(('pre', 'Z', qs, self.p_meas))
            elif nm == 'RZ':
                loc.append(('post', 'X', qs, self.p_reset))
            elif nm == 'RX':
                loc.append(('post', 'Z', qs, self.p_reset))
            elif nm == 'CCZ':
                loc.append(('post', 'D3', qs, self.p_ccz))
            elif nm in ('CS', 'CSDG', 'T', 'TDG'):
                loc.append(('post', 'D%d' % len(qs), qs, self.p_nc))
            else:
                loc.append(('post', 'D%d' % len(qs), qs, self.p))
        for q in idle:
            loc.append(('post', 'D1', (q,), self.p_idle))
        return [l for l in loc if l[3] > 0]


PAULI_CHOICES = {k: [''.join(p) for p in itertools.product('IXYZ', repeat=k) if set(p) != {'I'}]
                 for k in (1, 2, 3)}


# ------------------------------------------------------------------ the noisy protocol circuit
class D4SwitchingCircuit:
    """Time-step level description of the noisy KZH protocol on the bounded square lattice.

    Blocks (in order):
      ('S0', r)  r = 1..R0   FSC |+_L> preparation: data |+>, rounds of all FSC checks
      ('S1', r)  r = 1..Rin  switch-in: g |0>, rounds of all D4 stabilizers ([KZH] step 2)
      ('U',)                 transversal U = W V ([KZH] eq. (8), step 3)
      ('SO',)                switch-out: Z^g readout ([KZH] step 4; deferred-correction form)
      ('S4', r)  r = 1..Rout FSC rounds after switching back ([KZH] step 4)
      ('FIN',)               one noiseless FSC round (fictitious error correction, Takada Sec. VI D)
    Qubits are re-indexed compactly: data (non-frozen r,g,b) first, then ancillas.
    sequential=True measures the gadgets of a commuting class one after another (validation only)."""

    def __init__(self, L, R0, Rin, Rout, sequential=False, n_pre_u=1):
        self.n_pre_u = n_pre_u
        P = D4Protocol(L)
        lat = P.lat
        self.P, self.lat, self.L = P, lat, L
        self.R0, self.Rin, self.Rout = R0, Rin, Rout
        data = sorted(P.data_qubits((R, G, B)))
        gadgets = {'D4': P.round_gadgets('D4'), 'FSC': P.round_gadgets('FSC')}
        ancs = sorted({g[0][1] for k in gadgets for g in gadgets[k]})
        c = {q: i for i, q in enumerate(data + ancs)}
        self.c = c
        self.nq, self.nd = len(data) + len(ancs), len(data)
        self.data_rb = sorted(c[q] for q in P.data_qubits((R, B)))
        self.data_g = sorted(c[q] for q in P.data_qubits((G,)))
        self.data_all = sorted(c[q] for q in data)
        conv = {'INIT_X': 'RX', 'INIT_Z': 'RZ', 'MEAS_X': 'MX', 'MEAS_Z': 'MZ', 'CNOT': 'CX'}
        cv = lambda g: (conv.get(g[0], g[0]),) + tuple(c[q] for q in g[1:])
        # label of every ancilla, per code
        self.lab_of = {}
        for code, cd in (('D4', P.d4), ('FSC', P.fsc)):
            labs = [l for l, _ in cd.xgens] + [l for l in cd.zlabels if not P._is_frozen_check(l)]
            for l in labs:
                self.lab_of[(code, c[P.anc[l]])] = l

        def round_moments(code, which='all'):
            gl = gadgets[code]
            if which != 'all':
                gl = [g for g in gl if g[0][0] == ('INIT_Z' if which == 'Z' else 'INIT_X')]
            classes = commuting_classes(gl)
            if sequential:
                classes = [[g] for cls in classes for g in cls]
            moms = []
            for cls in classes:
                for layer in class_layers(cls):
                    moms.append([cv(g) for g in layer])
            return moms
        # D4 rounds: Pauli Z checks first, then (after the real-time flux correction) the X-type
        # Clifford stabilizers, so that fluxes detected by the Z checks are removed before they meet CCZs
        self.round_moms = {'FSC': round_moments('FSC'), 'D4z': round_moments('D4', 'Z'),
                           'D4x': round_moments('D4', 'X')}
        self.anc = {code: sorted({c[g[0][1]] for g in gadgets[code]}) for code in gadgets}
        self.U_moms = [[cv(g) for g in layer] for layer in class_layers([P.U_gates()])]
        # ---- switch-out data: S^g generators, Gamma (g configuration -> kappa), CZ^{r,b} pairs
        self.gens_g = [(lab, op) for lab, op in P.d4.xgens if lab[2] == 'g']
        gpos = {q: i for i, q in enumerate(self.data_g)}
        Sg = np.zeros((len(self.gens_g), len(self.data_g)), dtype=np.uint8)
        for i, (lab, op) in enumerate(self.gens_g):
            for q in op.s:
                if q in c and c[q] in gpos:
                    Sg[i, gpos[c[q]]] = 1
        self.Sg = Sg
        self.Gamma = left_inverse_T(Sg)
        self.gpairs = [[(c[i], c[j]) for (i, j) in op.quad if i in c and j in c] for lab, op in self.gens_g]
        self.network = [('NCCZ', self.data_g[e], i, j)
                        for v in range(len(self.gens_g)) for e in np.nonzero(self.Gamma[v])[0]
                        for (i, j) in self.gpairs[v]]
        # Pauli-frame strings Z^r Z^b for the S^g eigenvalues (protocol step 5 of the noiseless check V7)
        self.frame_strings = []
        for lab, op in self.gens_g:
            x0, y0 = lab[1]
            qs = [lat.q(lat.eidx[('v', x0, y)], cc) for y in range(y0) for cc in (R, B)]
            self.frame_strings.append(sorted(c[q] for q in qs if q in c))
        # ---- Z checks of the FSC (rb data, frozen checks removed) -> used for the S1 X frame
        self.zchecks = {}
        for code, cd in (('D4', P.d4), ('FSC', P.fsc)):
            rows = []
            for lab, (l2, qs) in zip(cd.zlabels, lat.z_checks(cd.colours)):
                if P._is_frozen_check(lab):
                    continue
                rows.append((lab, sorted(c[q] for q in qs if q in c)))
            self.zchecks[code] = rows
        self.xgens = {}
        for code, cd in (('D4', P.d4), ('FSC', P.fsc)):
            self.xgens[code] = [(lab, sorted(c[q] for q in op.s if q in c),
                                 [(c[i], c[j]) for (i, j) in op.quad if i in c and j in c]) for lab, op in cd.xgens]
        self._logical()
        self.blocks = self._blocks()

    # ------------------------------------------------------------ logical Clifford stabilizer
    def _logical(self):
        """Ideal output = D_F |+_L>_FSC with F(a) = w(a) + 4 sum_v c'_v(a) f^g_v(a) (protocol phase,
        validated in V7/V7b/V8).  Its logical Clifford stabilizer is H = D_F X^t D_F^dag = X^t D_Delta,
        Delta(x) = F(x+t) - F(x) = c0 + sum 2 a_i x_i + sum 4 b_ij x_i x_j (mod 8)."""
        P, lat, c = self.P, self.lat, self.c
        F = {}

        def add(mono, coef):
            m = frozenset(mono)
            if any(q not in c for q in m):
                return                                   # frozen qubits are |0>
            m = frozenset(c[q] for q in m)
            F[m] = (F.get(m, 0) + coef) % 8
        for coef, mono in W_operator(lat).terms:
            add(mono, coef)
        # c'_v(a): path sums along a BFS tree from the L_rb vertices (linear in a)
        from collections import deque
        root = lrb_vertices(lat)
        path = {v: [] for v in root}
        dq = deque(root)
        while dq:
            u = dq.popleft()
            for e in lat.incident(u):
                p, q = lat.endpoints(e)
                w = q if p == u else p
                if w not in path:
                    path[w] = path[u] + [e]
                    dq.append(w)
        for v in P.gverts:
            lin = [lat.q(e, cc) for e in path[v] for cc in (R, B)]
            for (i, j) in lat.cz_pairs(v, G):
                for k in lin:
                    add({k, i, j}, 4)
        self.Fpoly = {m: v for m, v in F.items() if v}
        t = sorted(c[int(q)] for q in np.nonzero(P.fsc.t_log)[0])
        self.t_log = t
        tset = set(t)
        D = {}
        for m, coef in self.Fpoly.items():           # F(x+t): substitute x_i -> 1 - x_i for i in t
            fixed = [q for q in m if q not in tset]
            flip = [q for q in m if q in tset]
            for sub in itertools.product((0, 1), repeat=len(flip)):
                mono = frozenset(fixed + [q for q, s in zip(flip, sub) if s])
                sign = (-1) ** sum(sub)
                D[mono] = (D.get(mono, 0) + coef * sign) % 8
        for m, coef in self.Fpoly.items():
            D[m] = (D.get(m, 0) - coef) % 8
        D = {m: v for m, v in D.items() if v}
        self.Delta_c0 = D.pop(frozenset(), 0)
        self.Delta_a, self.Delta_b = {}, {}
        for m, v in D.items():
            if len(m) == 1:
                assert v % 2 == 0, 'logical stabilizer not Clifford (odd linear term)'
                self.Delta_a[next(iter(m))] = v // 2 % 4
            elif len(m) == 2:
                assert v % 4 == 0, 'logical stabilizer not Clifford (quadratic term)'
                self.Delta_b[tuple(sorted(m))] = v // 4 % 2
            else:
                raise AssertionError('logical stabilizer not Clifford (cubic term)')
        self.Delta_a = {q: a for q, a in self.Delta_a.items() if a}
        self.Delta_b = {ij: b for ij, b in self.Delta_b.items() if b}

    def logical_proxy(self, fx, sign=1):
        """stim.PauliString of the Pauli proxy of the dressed logical stabilizer X^e H X^e
        (e = X frame fx on the data): proxy(X^t D_Delta R_e) = X^t * prod_{a_i=2} Z_i * R_e, where
        R_e = D_Delta^dag X^e D_Delta X^e = phase * Z-string (Takada Secs. VI D, VII B)."""
        n = self.nq
        ps = stim.PauliString(n)
        for q in self.t_log:
            ps[q] = 'X'
        zs = stim.PauliString(n)
        for q, a in self.Delta_a.items():
            if a == 2:
                zs[q] = 'Z'
        R = stim.PauliString(n)
        ph = 1
        for q, a in self.Delta_a.items():
            if fx[q] and a % 2 == 1:
                ph *= 1j ** a
                R *= _single(n, q, 'Z')
            elif fx[q] and a == 2:
                ph *= -1
        for (i, j), b in self.Delta_b.items():
            if fx[i] and fx[j]:
                ph *= -1
                R *= _single(n, i, 'Z') * _single(n, j, 'Z')
            elif fx[i]:
                R *= _single(n, j, 'Z')
            elif fx[j]:
                R *= _single(n, i, 'Z')
        out = ps * zs * R
        return out * complex(ph * sign)

    # ------------------------------------------------------------ blocks
    def _blocks(self):
        B_ = []
        rb, g, alld = self.data_rb, self.data_g, self.data_all
        for r in range(1, self.R0 + 1):
            pre = [[('RX', q) for q in rb]] if r == 1 else []
            B_.append(dict(name=('S0', r), code='FSC', moms=pre + self.round_moms['FSC'], data=rb))
        for r in range(1, self.Rin + 1 + self.n_pre_u):
            pre = [[('RZ', q) for q in g]] if r == 1 else []
            B_.append(dict(name=('S1', r, 'z'), code='D4', moms=pre + self.round_moms['D4z'], data=alld))
            if r <= self.Rin:            # rounds Rin+1.. have only the Z checks (flux checks before U)
                B_.append(dict(name=('S1', r, 'x'), code='D4', moms=self.round_moms['D4x'], data=alld))
        B_.append(dict(name=('U',), code=None, moms=self.U_moms, data=alld))
        B_.append(dict(name=('SO',), code=None,
                       moms=[[('FLIPX', q) for q in g], list(self.network), [('MZ0', q) for q in g]], data=alld))
        for r in range(1, self.Rout + 1):
            B_.append(dict(name=('S4', r), code='FSC', moms=self.round_moms['FSC'], data=rb))
        B_.append(dict(name=('FIN',), code='FSC', moms=self.round_moms['FSC'], data=rb, noiseless=True))
        return B_


def _single(n, q, p):
    s = stim.PauliString(n)
    s[q] = p
    return s


def left_inverse_T(S):
    """Gamma (m x n) over F2 with Gamma @ S^T = I_m, for S (m x n) of full row rank."""
    m, n = S.shape
    A = (S.T % 2).astype(np.uint8)
    M = np.concatenate([A, np.eye(n, dtype=np.uint8)], axis=1)
    r = 0
    for col in range(m):
        nz = np.nonzero(M[r:, col])[0]
        assert len(nz), 'S^T is not of full column rank'
        p = r + nz[0]
        M[[r, p]] = M[[p, r]]
        for rr in range(n):
            if rr != r and M[rr, col]:
                M[rr] ^= M[r]
        r += 1
    Gm = M[:m, m:]
    assert not ((Gm.astype(int) @ A.astype(int)) % 2 - np.eye(m, dtype=int)).any()
    return Gm


# ------------------------------------------------------------------ exact reference (validation only)
class TrueSim:
    """Exact sparse state-vector simulation of the TRUE (non-Clifford) circuit, used to validate the
    Pauli-proxy simulation.  Basis states are uint64 bit masks: data qubits 0..nd-1, ancillas get a
    slot nd+k while they are alive (reset ... measurement)."""
    OMEGA = np.exp(1j * np.pi / 4)

    def __init__(self, nd, keys, amps):
        self.nd = nd
        self.keys = np.asarray(keys, dtype=np.uint64)
        self.amps = np.asarray(amps, dtype=complex)
        self.slot = {}
        self.free = []
        self.nslots = 0

    def bit(self, q):
        if q < self.nd:
            return q
        if q not in self.slot:
            raise KeyError(q)
        return self.slot[q]

    def _alloc(self, q):
        if q < self.nd:
            return q
        if q in self.slot:
            return self.slot[q]
        b = self.free.pop() if self.free else self.nd + self.nslots
        if b == self.nd + self.nslots:
            self.nslots += 1
        self.slot[q] = b
        assert b < 64
        return b

    def _bits(self, b):
        return ((self.keys >> np.uint64(b)) & np.uint64(1)).astype(np.int64)

    def _merge(self):
        u, inv = np.unique(self.keys, return_inverse=True)
        re = np.bincount(inv, weights=self.amps.real, minlength=len(u))
        im = np.bincount(inv, weights=self.amps.imag, minlength=len(u))
        a = re + 1j * im
        keep = np.abs(a) > 1e-12
        self.keys, self.amps = u[keep], a[keep]

    def X(self, q):
        self.keys = self.keys ^ np.uint64(1 << self.bit(q))

    def Z(self, q):
        self.amps = self.amps * (1 - 2 * self._bits(self.bit(q)))

    def Y(self, q):
        self.Z(q)
        self.X(q)
        self.amps = self.amps * 1j

    def H(self, q):
        b = self.bit(q)
        m = np.uint64(1 << b)
        v = self._bits(b)
        k0 = self.keys & ~m
        a = self.amps / np.sqrt(2)
        self.keys = np.concatenate([k0, k0 | m])
        self.amps = np.concatenate([a, a * (1 - 2 * v)])
        self._merge()

    def diag(self, qs, theta):
        bits = [self._bits(self.bit(q)) for q in qs]
        ph = np.array([theta(t) for t in zip(*bits)], dtype=float) if len(self.keys) else np.zeros(0)
        self.amps = self.amps * self.OMEGA ** ph

    def CX(self, c, t):
        vc = self._bits(self.bit(c)).astype(np.uint64)
        self.keys = self.keys ^ (vc << np.uint64(self.bit(t)))

    def measure_z(self, q, forced):
        b = self.bit(q)
        v = self._bits(b)
        p1 = float(np.sum(np.abs(self.amps[v == 1]) ** 2))
        prob = p1 if forced else 1 - p1
        keep = v == forced
        self.keys, self.amps = self.keys[keep], self.amps[keep]
        if prob > 1e-12:
            self.amps = self.amps / np.sqrt(prob)
        return prob

    def release(self, q, outcome):
        if q >= self.nd:
            if outcome:
                self.X(q)
            self.free.append(self.slot.pop(q))

    def reset(self, q, basis, rng):
        b = self._alloc(q)
        if q < self.nd:                      # data reset: measure and flip to |0>
            v = self._bits(b)
            p1 = float(np.sum(np.abs(self.amps[v == 1]) ** 2))
            o = int(rng.random() < p1)
            self.measure_z(q, o)
            if o:
                self.X(q)
        if basis == 'X':
            self.H(q)

    def run_moment(self, mom, errs, outcomes, rng):
        """Execute a time step with the given sampled errors; measurements forced to `outcomes`
        (list in the order of the moment's measurements).  Returns list of probabilities."""
        for when, ps, qs in errs:
            if when == 'pre':
                for pc, q in zip(ps, qs):
                    if pc != 'I':
                        getattr(self, pc)(q)
        probs = []
        k = 0
        for op in mom:
            nm, qs = op[0], op[1:]
            if nm in DIAG_THETA:
                self.diag(qs, DIAG_THETA[nm])
            elif nm == 'CX':
                self.CX(*qs)
            elif nm == 'CZ':
                self.diag(qs, lambda x: 4 * x[0] * x[1])
            elif nm in ('S', 'SDG'):
                self.diag(qs, (lambda x: 2 * x[0]) if nm == 'S' else (lambda x: 6 * x[0]))
            elif nm in ('X', 'Z', 'Y'):
                getattr(self, nm)(qs[0])
            elif nm in ('RX', 'RZ'):
                self.reset(qs[0], nm[1], rng)
            elif nm in ('MX', 'MZ', 'MZ0'):
                q = qs[0]
                if nm == 'MX':
                    self.H(q)
                o = outcomes[k]
                k += 1
                probs.append(self.measure_z(q, o))
                if q >= self.nd:
                    self.release(q, o)
            elif nm == 'FLIPX':
                pass
            else:
                raise ValueError(nm)
        for when, ps, qs in errs:
            if when == 'post':
                for pc, q in zip(ps, qs):
                    if pc != 'I':
                        getattr(self, pc)(q)
        return probs

    def expect_logical(self, circ):
        """<Psi| X^t D_Delta |Psi> on the data qubits (all ancillas released)."""
        assert not self.slot
        x = self.keys
        ph = np.full(len(x), circ.Delta_c0, dtype=float)
        bits = {}
        get = lambda q: bits.setdefault(q, ((x >> np.uint64(q)) & np.uint64(1)).astype(np.int64))
        for q, a in circ.Delta_a.items():
            ph += 2 * a * get(q)
        for (i, j), b in circ.Delta_b.items():
            ph += 4 * b * get(i) * get(j)
        Da = self.amps * self.OMEGA ** ph
        tm = np.uint64(sum(1 << q for q in circ.t_log))
        k2 = x ^ tm
        order = np.argsort(x)
        xs = x[order]
        pos = np.searchsorted(xs, k2)
        pos[pos >= len(xs)] = 0
        hit = xs[pos] == k2
        amp_t = np.zeros(len(x), dtype=complex)
        amp_t[hit] = self.amps[order][pos[hit]]
        return complex(np.sum(np.conj(amp_t) * Da))


# ------------------------------------------------------------------ compiled protocol (fast Monte Carlo)
_KIND = {'X': 0, 'Z': 1, 'D1': 2, 'D2': 3, 'D3': 4}


class CompiledMoment:
    __slots__ = ('ops', 'base', 'meas', 'cx_c', 'cx_t', 'resets', 'clears', 'nonc', 'loc_slice')


class CompiledProtocol:
    """Pre-compiled time steps of a D4SwitchingCircuit + flattened noise-location table."""

    def __init__(self, circ, noise):
        self.circ, self.noise = circ, noise
        locs = []
        self.blocks = []
        live = set()
        for blk in circ.blocks:
            moms = []
            for mom in blk['moms']:
                cm = CompiledMoment()
                cm.ops = mom
                lines, meas, cxc, cxt, rs, cl, nc = [], [], [], [], [], [], []
                for op in mom:
                    nm, qs = op[0], op[1:]
                    if nm in NONCLIFF:
                        nc.append((nm, qs))
                    elif nm == 'FLIPX':
                        pass
                    elif nm == 'CX':
                        lines.append('CX %d %d' % qs)
                        cxc.append(qs[0]); cxt.append(qs[1])
                    elif nm == 'CZ':
                        lines.append('CZ %d %d' % qs)
                    elif nm in ('RX', 'RZ'):
                        lines.append(('RX %d' if nm == 'RX' else 'R %d') % qs[0])
                        rs.append(qs[0])
                    elif nm in ('MX', 'MZ', 'MZ0'):
                        lines.append(('MX %d' if nm == 'MX' else 'M %d') % qs[0])
                        meas.append((qs[0], 'X' if nm == 'MX' else 'Z'))
                        if nm == 'MZ0':
                            cl.append(qs[0])
                    else:
                        lines.append('%s %d' % (_STIM1[nm], qs[0]))
                cm.base = stim.Circuit('\n'.join(lines)) if lines else None
                cm.meas = meas
                cm.cx_c, cm.cx_t = np.array(cxc, dtype=np.int64), np.array(cxt, dtype=np.int64)
                cm.resets, cm.clears = np.array(rs, dtype=np.int64), np.array(cl, dtype=np.int64)
                cm.nonc = nc
                touched = {q for op in mom for q in op[1:]}
                idle = [q for q in blk['data'] if q not in touched] + [q for q in live if q not in touched]
                ml = [] if blk.get('noiseless') else noise.locations(mom, idle)
                s0 = len(locs)
                locs += ml
                cm.loc_slice = (s0, len(locs))
                for op in mom:
                    if op[0] in ('RX', 'RZ') and op[1] >= circ.nd:
                        live.add(op[1])
                    if op[0] in ('MX', 'MZ'):
                        live.discard(op[1])
                moms.append(cm)
            self.blocks.append(dict(name=blk['name'], code=blk['code'], moms=moms))
        self.loc_when = np.array([0 if l[0] == 'pre' else 1 for l in locs], dtype=np.int8)
        self.loc_kind = np.array([_KIND[l[1]] for l in locs], dtype=np.int8)
        self.loc_p = np.array([l[3] for l in locs])
        self.loc_q = [l[2] for l in locs]
        self.n_loc = len(locs)


class FastShot:
    """One Monte Carlo shot of the Pauli-proxy circuit (same semantics as ProxyRun, compiled)."""
    _G1 = {'S': 's', 'SDG': 's_dag', 'Z': 'z'}

    def __init__(self, comp, rng, forced=None):
        self.comp, self.rng, self.circ = comp, rng, comp.circ
        self.sim = stim.TableauSimulator(seed=int(rng.integers(2 ** 62)))
        self.sim.set_num_qubits(comp.circ.nq)
        self.fx = np.zeros(comp.circ.nq, dtype=np.uint8)
        if forced is not None:                  # {location index: pauli string}
            self.errs = {i: (comp.loc_when[i], ps, comp.loc_q[i]) for i, ps in forced.items()}
            self.n_err = len(forced)
            self.n_dress = 0
            return
        hit = np.nonzero(rng.random(comp.n_loc) < comp.loc_p)[0]
        self.errs = {}
        for i in hit:
            k = comp.loc_kind[i]
            qs = comp.loc_q[i]
            if k == 0:
                ps = 'X'
            elif k == 1:
                ps = 'Z'
            else:
                ch = PAULI_CHOICES[len(qs)]
                ps = ch[rng.integers(len(ch))]
            self.errs[i] = (comp.loc_when[i], ps, qs)
        self.n_err = len(hit)
        self.n_dress = 0

    def _pauli(self, ps, qs):
        sim, fx = self.sim, self.fx
        for pc, q in zip(ps, qs):
            if pc == 'X':
                sim.x(q); fx[q] ^= 1
            elif pc == 'Y':
                sim.y(q); fx[q] ^= 1
            elif pc == 'Z':
                sim.z(q)

    def run_block(self, bi):
        """Run block bi; returns list of (qubit, basis, outcome) of its measurements."""
        blk = self.comp.blocks[bi]
        sim, fx = self.sim, self.fx
        nmeas = 0
        meas = []
        for cm in blk['moms']:
            a, b = cm.loc_slice
            errs = [self.errs[i] for i in range(a, b) if i in self.errs] if b > a and self.errs else []
            for w, ps, qs in errs:
                if w == 0:
                    self._pauli(ps, qs)
            for nm, qs in cm.nonc:
                e = tuple(int(fx[q]) for q in qs)
                if any(e):
                    for gt in dressing(nm, e):
                        if gt[0] == 'CZ':
                            sim.cz(qs[gt[1]], qs[gt[2]])
                        else:
                            getattr(sim, self._G1[gt[0]])(qs[gt[1]])
                        self.n_dress += 1
            if cm.base is not None:
                sim.do_circuit(cm.base)
            if len(cm.cx_c):
                fx[cm.cx_t] ^= fx[cm.cx_c]
            if len(cm.resets):
                fx[cm.resets] = 0
            if len(cm.clears):
                fx[cm.clears] = 0
            for w, ps, qs in errs:
                if w == 1:
                    self._pauli(ps, qs)
            nmeas += len(cm.meas)
            meas += cm.meas
        if not nmeas:
            return []
        rec = sim.current_measurement_record()[-nmeas:]
        return [(q, bb, int(o)) for (q, bb), o in zip(meas, rec)]

    def apply_x(self, qubits):
        for q in qubits:
            self.sim.x(q)
            self.fx[q] ^= 1

    def init_frame(self, zrows, qubits):
        """X frame at the start of the first non-Clifford block (S1).  S0 is a Clifford circuit, so stim's
        state is the true state E|ideal>, |ideal> = FSC|+_L> (all checks +1); the X part of E is any
        solution x of H_Z x = s (s = current Z-check signs): solutions differ by X stabilizers / logical X
        of |ideal>, which act trivially on it, and Algorithm 2 is exact for any such representation."""
        s = []
        for lab, qs in zrows:
            ps = stim.PauliString(self.circ.nq)
            for q in qs:
                ps[q] = 'Z'
            v = self.sim.peek_observable_expectation(ps)
            assert v in (1, -1), 'Z check not deterministic at the end of S0'
            s.append(0 if v == 1 else 1)
        H = np.zeros((len(zrows), len(qubits)), dtype=np.uint8)
        pos = {q: i for i, q in enumerate(qubits)}
        for r, (lab, qs) in enumerate(zrows):
            for q in qs:
                H[r, pos[q]] = 1
        x = f2_solve(H.T, np.array(s, dtype=np.uint8))      # x H^T = s  <=>  H x = s
        assert x is not None
        self.fx[:] = 0
        for i in np.nonzero(x)[0]:
            self.fx[qubits[i]] = 1
        return np.array(s)

    _G = {'CX': 'cx', 'CZ': 'cz', 'S': 's', 'SDG': 's_dag', 'Z': 'z', 'X': 'x', 'Y': 'y'}

    def run_block_lockstep(self, bi, ts, rng):
        """Validation: same as run_block, but op by op, with the exact simulation `ts` (TrueSim) of the
        TRUE non-Clifford circuit run in lockstep with the same errors.  Before every measurement stim's
        peek value (+1/-1 deterministic, 0 random) is compared with the exact probability of the outcome.
        Returns list of (qubit, basis, outcome, peek, exact probability)."""
        blk = self.comp.blocks[bi]
        sim, fx = self.sim, self.fx
        out = []
        for cm in blk['moms']:
            a, b = cm.loc_slice
            errs = [self.errs[i] for i in range(a, b) if i in self.errs]
            terrs = [('pre' if w == 0 else 'post', ps, qs) for w, ps, qs in errs]
            for w, ps, qs in errs:
                if w == 0:
                    self._pauli(ps, qs)
            for nm, qs in cm.nonc:
                e = tuple(int(fx[q]) for q in qs)
                if any(e):
                    for gt in dressing(nm, e):
                        getattr(sim, self._G[gt[0]])(*[qs[i] for i in gt[1:]])
                        self.n_dress += 1
            outs, peeks = [], []
            for op in cm.ops:
                nm, qs = op[0], op[1:]
                if nm in NONCLIFF or nm == 'FLIPX':
                    continue
                if nm in ('RX', 'RZ'):
                    (sim.reset_x if nm == 'RX' else sim.reset_z)(qs[0])
                elif nm in ('MX', 'MZ', 'MZ0'):
                    q = qs[0]
                    if nm == 'MX':
                        peeks.append(sim.peek_x(q)); sim.h(q); o = sim.measure(q); sim.h(q)
                    else:
                        peeks.append(sim.peek_z(q)); o = sim.measure(q)
                    outs.append(int(o))
                else:
                    getattr(sim, self._G[nm])(*qs)
            if len(cm.cx_c):
                fx[cm.cx_t] ^= fx[cm.cx_c]
            if len(cm.resets):
                fx[cm.resets] = 0
            if len(cm.clears):
                fx[cm.clears] = 0
            for w, ps, qs in errs:
                if w == 1:
                    self._pauli(ps, qs)
            probs = ts.run_moment(cm.ops, terrs, outs, rng)
            out += [(q, bb, o, pk, pr) for (q, bb), o, pk, pr in zip(cm.meas, outs, peeks, probs)]
        return out


# ------------------------------------------------------------------ decoders
def _wt(p):
    p = min(max(p, 1e-12), 0.5 - 1e-9)
    return float(np.log((1 - p) / p))


_XFRAC = {0: 1.0, 1: 0.0, 2: 2 / 3, 3: 8 / 15, 4: 32 / 63}       # prob. that a location puts X/Y on a qubit


class FluxDecoder:
    """Real-time ("just-in-time") decoding of the Pauli Z checks (non-Abelian fluxes = X errors).

    [KZH] step 2 pairs the fluxes "in real time ... whenever the observation time of the syndromes is
    comparable to the separation of the syndromes" with the decoder of [Dav] (not accessible here).
    Implementation used in this study (own construction, documented as such): after every round t the
    space-time matching graph of all rounds <= t (with an open time boundary after round t) is decoded
    by minimum-weight perfect matching (PyMatching); the space-like edges whose round is <= t - delay
    are committed and the corresponding X corrections are applied physically at once (feed-forward).
    Before the first non-Clifford block (end of S0) and before U (end of S1) everything is committed.
    The g readout of the switch-out is decoded together with the g fluxes of S1 and gives the
    corrected configuration used for kappa."""

    def __init__(self, comp, delay=0, gauge_w=0.05):
        circ = comp.circ
        self.comp, self.circ, self.delay, self.gauge_w = comp, circ, delay, gauge_w
        zr = {code: circ.zchecks[code] for code in ('D4', 'FSC')}
        g_rows = [(lab, qs) for lab, qs in zr['D4'] if lab[0] == 'plaq' and lab[2] == 'g']
        # flux rounds
        self.fr = []
        for bi, blk in enumerate(comp.blocks):
            nm = blk['name']
            if nm[0] in ('S0', 'S4', 'FIN') or (nm[0] == 'S1' and nm[2] == 'z'):
                self.fr.append(dict(block=bi, name=nm, rows=zr[blk['code']]))
            elif nm[0] == 'SO':
                self.fr.append(dict(block=bi, name=nm, rows=g_rows))
        off = 0
        for k, f in enumerate(self.fr):
            f['off'] = off
            f['idx'] = {lab: off + i for i, (lab, qs) in enumerate(f['rows'])}
            off += len(f['rows'])
        self.n_nodes = off
        # relation for the SO detectors: prod_{q in g-plaq} Z^g Z^r Z^b (pre-V) = sum of D4 rows
        lat, c = circ.lat, circ.c
        inv = {v: k for k, v in c.items()}
        d4rows = zr['D4']
        H = np.zeros((len(d4rows), circ.nq), dtype=np.uint8)
        for i, (lab, qs) in enumerate(d4rows):
            H[i, qs] = 1
        self.so_rel = {}
        for lab, qs in g_rows:
            v = np.zeros(circ.nq, dtype=np.uint8)
            for q in qs:
                e = inv[q] // 3
                for cc in (R, G, B):
                    raw = lat.q(e, cc)
                    if raw in c:
                        v[c[raw]] ^= 1
            sol = f2_solve(H, v)
            assert sol is not None, 'SO relation not in the D4 Z-check span'
            self.so_rel[lab] = [d4rows[i][0] for i in np.nonzero(sol)[0]]
        # rows containing each qubit, per code
        self.rows_of = {code: {} for code in ('D4', 'FSC')}
        for code in ('D4', 'FSC'):
            for lab, qs in zr[code]:
                for q in qs:
                    self.rows_of[code].setdefault(q, []).append(lab)
        self._error_rates()
        self._build()

    def _error_rates(self):
        """Phenomenological edge probabilities estimated from the circuit-level noise locations."""
        comp = self.comp
        self.p_data = []           # per flux round: dict qubit -> prob of X/Y since the previous flux round
        self.p_meas = []           # per flux round: dict label -> prob that the check outcome is flipped
        prev_block = -1
        for f in self.fr:
            blk = comp.blocks[f['block']]
            pd, pa = {}, {}
            for bj in range(prev_block + 1, f['block'] + 1):
                for cm in comp.blocks[bj]['moms']:
                    a, b = cm.loc_slice
                    for i in range(a, b):
                        fr_ = _XFRAC[comp.loc_kind[i]] * comp.loc_p[i]
                        for q in comp.loc_q[i]:
                            if q < self.circ.nd:
                                pd[q] = pd.get(q, 0) + fr_
                            elif bj == f['block']:
                                pa[q] = pa.get(q, 0) + fr_
            prev_block = f['block']
            pm = {}
            if blk['code']:
                for lab, qs in f['rows']:
                    a = self.circ.P.anc[lab]
                    pm[lab] = pa.get(self.circ.c[a], 0)
            self.p_data.append(pd)
            self.p_meas.append(pm)

    def _build(self):
        """Matching graphs M_t for t = 0..len(fr)-1 (rounds <= t, open boundary after t)."""
        self.mt = []
        nfr = len(self.fr)
        for t in range(nfr):
            m = pymatching.Matching()
            look = {}

            def add(u, v, w, info):
                key = (u, v) if v == -1 else (min(u, v), max(u, v))
                if key in look and look[key][0] <= w:
                    return
                look[key] = (w, info)
            for k in range(t + 1):
                f = self.fr[k]
                code = self.comp.blocks[f['block']]['code'] or 'D4'
                rows = f['rows']
                rowset = {lab for lab, _ in rows}
                # space edges: X on a data qubit before round k
                qrows = {}
                for lab, qs in rows:
                    for q in qs:
                        qrows.setdefault(q, []).append(f['idx'][lab])
                for q, nodes in qrows.items():
                    if k == 0:
                        w = self.gauge_w           # random first-round outcomes of S0 (gauge choice)
                    else:
                        w = _wt(self.p_data[k].get(q, 1e-6))
                    if len(nodes) == 2:
                        add(nodes[0], nodes[1], w, ('s', q, k))
                    elif len(nodes) == 1:
                        add(nodes[0], -1, w, ('s', q, k))
                # time edges to the previous occurrence of the same label
                if k > 0:
                    for lab, qs in rows:
                        prev = self._prev(k, lab)
                        if prev is not None:
                            kp, node_p = prev
                            w = _wt(self.p_meas[kp].get(lab, 1e-6))
                            add(node_p, f['idx'][lab], w, ('t', lab, kp))
            # switch-out round: a measurement error of a check of the last D4 Z round enters the reference
            # of the g-readout detectors (hyper-edge with its next detector); represent it by an edge between
            # the two nodes known at this time and fix the stored reference when it is committed ('m')
            if self.fr[t]['name'][0] == 'SO':
                f = self.fr[t]
                kp = t - 1
                for lab, qs in f['rows']:
                    for l2 in self.so_rel[lab]:
                        if l2 != lab and l2 in self.fr[kp]['idx']:
                            add(self.fr[kp]['idx'][l2], f['idx'][lab], _wt(self.p_meas[kp].get(l2, 1e-6)),
                                ('m', l2, kp))
            # open time boundary: nodes whose label is measured again after t
            for k in range(t + 1):
                f = self.fr[k]
                for lab, qs in f['rows']:
                    if self._next(k, lab) is not None and self._next(k, lab) > t and self._latest(k, lab, t):
                        add(f['idx'][lab], -1, _wt(self.p_meas[k].get(lab, 1e-6) + 1e-9), ('f', lab, k))
            for (u, v), (w, info) in look.items():
                if v == -1:
                    m.add_boundary_edge(u, weight=w)
                else:
                    m.add_edge(u, v, weight=w)
            nn = self.fr[t]['off'] + len(self.fr[t]['rows'])
            self.mt.append((m, {k_: v[1] for k_, v in look.items()}, nn))

    def _prev(self, k, lab):
        for kp in range(k - 1, -1, -1):
            if lab in self.fr[kp]['idx']:
                return kp, self.fr[kp]['idx'][lab]
        return None

    def _next(self, k, lab):
        for kn in range(k + 1, len(self.fr)):
            if lab in self.fr[kn]['idx']:
                return kn
        return None

    def _latest(self, k, lab, t):
        """True if round k holds the latest occurrence of lab among rounds <= t."""
        for kk in range(k + 1, t + 1):
            if lab in self.fr[kk]['idx']:
                return False
        return True

    # ---------------------------------------------------------------- per shot
    def start(self):
        self.last = {}
        self.active = set()
        self.k = -1

    def push_round(self, k, outcomes_by_label=None, readout=None):
        """Add the detection events of flux round k.  outcomes_by_label: dict label -> bit (Z checks);
        readout: dict g-qubit -> bit (switch-out)."""
        f = self.fr[k]
        self.k = k
        if f['name'][0] == 'SO':
            par = {}
            for lab, qs in f['rows']:
                v = 0
                for q in qs:
                    v ^= readout[q]
                ref = 0
                for l2 in self.so_rel[lab]:
                    ref ^= self.last.get(l2, 0)
                par[lab] = v
                if v ^ ref:
                    self.active ^= {f['idx'][lab]}
            self.readout = dict(readout)
            return
        for lab, qs in f['rows']:
            m = outcomes_by_label[lab]
            if m ^ self.last.get(lab, 0):
                self.active ^= {f['idx'][lab]}
            self.last[lab] = m

    def decode(self, flush=False, min_age=None):
        """Decode with all rounds <= current; returns list of committed space corrections (qubit, k).
        Commits space-like edges with round k <= t - delay (flush: k <= t - min_age, default all)."""
        t = self.k
        m, look, nn = self.mt[t]
        if not self.active:
            return []
        syn = np.zeros(m.num_detectors, dtype=np.uint8)
        for u in self.active:
            syn[u] = 1
        edges = m.decode_to_edges_array(syn)
        out = []
        for u, v in edges:
            u, v = int(u), int(v)
            key = (u, -1) if v == -1 else ((v, -1) if u == -1 else (min(u, v), max(u, v)))
            info = look[key]
            if info[0] == 'm' and flush:
                self.active ^= {key[0], key[1]}
                self.last[info[1]] = self.last.get(info[1], 0) ^ 1     # it was a measurement error
                continue
            if info[0] != 's':
                continue
            q, k = info[1], info[2]
            lim = (t - (min_age or 0)) if flush else (t - self.delay)
            if k > lim:
                continue
            out.append((q, k))
            if key[1] == -1:
                self.active ^= {key[0]}
            else:
                self.active ^= {key[0], key[1]}
        return out

    def register_correction(self, q, code):
        """A physical X correction on data qubit q: flip the stored reference of every Z check of the
        current code containing q (as if it had been applied before that check was measured)."""
        for lab in self.rows_of[code].get(q, []):
            if lab in self.last:
                self.last[lab] ^= 1


class ChargeDecoder:
    """Decoding of the X-type checks (Abelian charges = Z errors) after the protocol: minimum-weight
    matching on the detector error model of the Pauli-proxy circuit (stim + PyMatching).  The DEM
    describes the Pauli errors of the circuit; the extra randomness produced by non-Abelian dressings
    is not in the model (it is present in the simulated data).  Output: predicted flip of the logical
    observable (proxy of the logical Clifford stabilizer, including the Pauli-frame strings)."""

    def __init__(self, comp, dem_noise=None, aligned=True):
        circ = comp.circ
        self.comp, self.circ = comp, circ
        nz = dem_noise or comp.noise
        # measurement bookkeeping
        meas_lab = []          # per absolute measurement: (block_name, kind, label or qubit)
        blk_meas = []
        for bi, blk in enumerate(comp.blocks):
            ids = []
            for cm in blk['moms']:
                for q, b in cm.meas:
                    if blk['code']:
                        lab = circ.lab_of[(blk['code'], q)]
                        meas_lab.append((blk['name'], b, lab))
                    else:
                        meas_lab.append((blk['name'], 'R', q))
                    ids.append(len(meas_lab) - 1)
            blk_meas.append(ids)
        self.n_meas = len(meas_lab)
        self.blk_meas = blk_meas
        idx = {}
        for i, (bn, b, lab) in enumerate(meas_lab):
            idx[(bn, lab)] = i
        self.meas_idx = idx
        sup = {code: {lab: set(xs) for lab, xs, quad in circ.xgens[code]} for code in ('D4', 'FSC')}
        names = [blk['name'] for blk in comp.blocks]
        S0 = [n for n in names if n[0] == 'S0']
        S1 = [n for n in names if n[0] == 'S1' and n[2] == 'x']
        S4 = [n for n in names if n[0] in ('S4', 'FIN')]
        dets = []
        fscl = [lab for lab, _, _ in circ.xgens['FSC']]
        d4l = [lab for lab, _, _ in circ.xgens['D4']]
        for j, n in enumerate(S0):
            for lab in fscl:
                dets.append([idx[(n, lab)]] + ([idx[(S0[j - 1], lab)]] if j else []))

        def span_solve(target, basis_labels, code):
            M = np.zeros((len(basis_labels), circ.nq), dtype=np.uint8)
            for i, l in enumerate(basis_labels):
                M[i, list(sup[code][l])] = 1
            v = np.zeros(circ.nq, dtype=np.uint8)
            v[list(target)] = 1
            sol = f2_solve(M, v)
            return None if sol is None else [basis_labels[i] for i in np.nonzero(sol)[0]]
        for j, n in enumerate(S1):
            for lab in d4l:
                if j == 0:
                    if not S0:
                        continue
                    combo = span_solve(sup['D4'][lab], fscl, 'FSC')
                    if combo is None:
                        continue                       # S^g: random first outcome (gauging)
                    dets.append([idx[(n, lab)]] + [idx[(S0[-1], l)] for l in combo])
                else:
                    dets.append([idx[(n, lab)], idx[(S1[j - 1], lab)]])
        lat, c = circ.lat, circ.c
        inv = {v: k for k, v in c.items()}
        for j, n in enumerate(S4):
            first = {}
            for lab in fscl:
                if j == 0:
                    tgt = set(sup['FSC'][lab])
                    for q in list(sup['FSC'][lab]):
                        gq = lat.q(inv[q] // 3, G)
                        if gq in c:
                            tgt ^= {c[gq]}
                    combo = span_solve(tgt, d4l, 'D4')
                    assert combo is not None, 'S4 X check not determined by the last D4 round'
                    first[lab] = [idx[(n, lab)]] + [idx[(S1[-1], l)] for l in combo]
                else:
                    dets.append([idx[(n, lab)], idx[(S4[j - 1], lab)]])
            if j == 0:
                # detector basis across the switch-out: aligned=True uses, at a vertex with separate r and
                # b checks, D_r and D_r + D_b (S^g of the last D4 round cancels in D_r + D_b), so that a g
                # charge created during the last D4 round is a graph-like error, but an r charge then has
                # 4 symptoms; aligned=False (D_r, D_b) has the opposite property.  D4Experiment decodes
                # with both and keeps the lower-weight solution (no single basis makes all graph-like)
                for lab, d in first.items():
                    v, col = lab[1], lab[2]
                    if aligned and col == 'b' and ('X', v, 'r') in first:
                        dets.append(sorted(set(d) ^ set(first[('X', v, 'r')])))
                    else:
                        dets.append(d)
        self.dets = dets
        # logical observable: proxy logical (measured by MPP at the end of the DEM circuit) + frame terms
        tset = set(circ.t_log)
        self.cross = [len(set(fs) & tset) % 2 for fs in circ.frame_strings]
        self.obs_meas = [idx[(S1[-1], lab)] for (lab, op), cr in zip(circ.gens_g, self.cross) if cr]
        self._dem_circuit(nz)
        # sparse detector evaluation
        self.det_ptr = np.cumsum([0] + [len(d) for d in dets])
        self.det_idx = np.concatenate([np.array(d, dtype=np.int64) for d in dets])

    def _dem_circuit(self, nz):
        circ, comp = self.circ, self.comp
        lines = []
        for bi, blk in enumerate(comp.blocks):
            noiseless = blk['name'][0] == 'FIN'
            for cm in blk['moms']:
                pre, post = [], []
                touched = {q for op in cm.ops for q in op[1:]}
                a, b = cm.loc_slice
                for i in range(a, b):
                    k, p, qs = comp.loc_kind[i], comp.loc_p[i], comp.loc_q[i]
                    if k == 0:
                        ln = 'X_ERROR(%.12g) %d' % (p, qs[0])
                    elif k == 1:
                        ln = 'Z_ERROR(%.12g) %d' % (p, qs[0])
                    elif k == 2:
                        ln = 'DEPOLARIZE1(%.12g) %d' % (p, qs[0])
                    elif k == 3:
                        ln = 'DEPOLARIZE2(%.12g) %d %d' % (p, qs[0], qs[1])
                    else:
                        ln = 'DEPOLARIZE1(%.12g) %d %d %d' % (p * 48 / 63, *qs)
                    (pre if comp.loc_when[i] == 0 else post).append(ln)
                lines += pre
                if cm.base is not None:
                    lines.append(str(cm.base))
                lines += post
            lines.append('TICK')
        n = circ.nq
        lp = ['X%d' % q for q in circ.t_log if circ.Delta_a.get(q) != 2]
        lp += ['Y%d' % q for q in circ.t_log if circ.Delta_a.get(q) == 2]
        lp += ['Z%d' % q for q, a in circ.Delta_a.items() if a == 2 and q not in set(circ.t_log)]
        lines.append('MPP ' + '*'.join(lp))
        tot = self.n_meas + 1
        for d in self.dets:
            lines.append('DETECTOR ' + ' '.join('rec[%d]' % (i - tot) for i in d))
        lines.append('OBSERVABLE_INCLUDE(0) rec[-1] ' + ' '.join('rec[%d]' % (i - tot) for i in self.obs_meas))
        self.stim_circuit = stim.Circuit('\n'.join(lines))
        dem = self.stim_circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True,
                                                     approximate_disjoint_errors=True)
        self.matching = pymatching.Matching.from_detector_error_model(dem)
        # graph-like components of the DEM (for the flux-informed re-weighting)
        comp = {}
        for ins in dem.flattened():
            if ins.type != 'error':
                continue
            p = ins.args_copy()[0]
            cur_d, cur_o = [], 0
            parts = []
            for t in ins.targets_copy():
                if t.is_separator():
                    parts.append((tuple(sorted(cur_d)), cur_o)); cur_d, cur_o = [], 0
                elif t.is_relative_detector_id():
                    cur_d.append(t.val)
                elif t.is_logical_observable_id():
                    cur_o ^= 1
            parts.append((tuple(sorted(cur_d)), cur_o))
            for key in parts:
                if not key[0] or len(key[0]) > 2:
                    continue
                q0 = comp.get(key, 0.0)
                comp[key] = q0 * (1 - p) + p * (1 - q0)
        keys = list(comp)
        self.comp_keys = keys
        self.comp_index = {k[0]: i for i, k in enumerate(keys)}
        from scipy.sparse import csc_matrix
        rows, cols = [], []
        for j, (dd, o) in enumerate(keys):
            for dtt in dd:
                rows.append(dtt); cols.append(j)
        nd = self.stim_circuit.num_detectors
        self.H = csc_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(nd, len(keys)))
        self.O = csc_matrix((np.ones(sum(o for _, o in keys), dtype=np.uint8),
                             ([0] * sum(o for _, o in keys), [j for j, (dd, o) in enumerate(keys) if o])),
                            shape=(1, len(keys)))
        self.w0 = np.array([_wt(comp[k]) for k in keys])
        # detectors containing each measurement
        self.dets_of_meas = {}
        for j, d_ in enumerate(self.dets):
            for i in d_:
                self.dets_of_meas.setdefault(i, []).append(j)

    def detectors(self, outcomes):
        o = np.asarray(outcomes, dtype=np.uint8)[self.det_idx]
        return np.bitwise_xor.reduceat(o, self.det_ptr[:-1]) if len(o) else np.zeros(0, np.uint8)

    def flux_erasures(self, flux_events, w_erase=0.1):
        """Flux-informed re-weighting (our construction, following [KZH]'s description of cleaning up the
        Abelian charges after the fluxes have been paired): a flux on qubit q present during the X-type
        round ('S1', r, 'x') randomizes the outcomes of the D4 stabilizers whose CZ dressing contains q and
        leaves Z errors on the CZ partners with probability 1/2.  The corresponding time-like components
        and the space-like components of the partner Z errors get the weight w_erase."""
        circ = self.circ
        if not hasattr(self, '_cz_of'):
            self._cz_of = {}
            for lab, xs, quad in circ.xgens['D4']:
                for (i, j) in quad:
                    self._cz_of.setdefault(i, []).append((lab, j))
                    self._cz_of.setdefault(j, []).append((lab, i))
            self._stars = {}
            for lab, xs, quad in circ.xgens['D4']:
                for q in xs:
                    self._stars.setdefault(q, []).append(lab)
        er = set()
        for q, r in flux_events:
            blk = ('S1', r, 'x')
            nxt = ('S1', r + 1, 'x')
            for lab, partner in self._cz_of.get(q, []):
                m = self.meas_idx.get((blk, lab))
                if m is None:
                    continue
                dd = tuple(sorted(self.dets_of_meas.get(m, [])))
                if 1 <= len(dd) <= 2 and dd in self.comp_index:
                    er.add(self.comp_index[dd])
                # space-like component: Z on the partner flips the two stars containing it
                st = self._stars.get(partner, [])
                for rr in (blk, nxt):
                    ds = []
                    for l2 in st:
                        m2 = self.meas_idx.get((rr, l2))
                        if m2 is not None:
                            ds += [j for j in self.dets_of_meas.get(m2, []) if self.dets[j][0] == m2]
                    dd = tuple(sorted(ds))
                    if 1 <= len(dd) <= 2 and dd in self.comp_index:
                        er.add(self.comp_index[dd])
        return er

    def predict(self, outcomes, erasures=None, w_erase=0.1):
        """Returns (predicted flip of the logical observable, weight of the matching)."""
        det = self.detectors(outcomes)
        if not erasures:
            pred, w = self.matching.decode(det, return_weight=True)
            return int(pred[0]), float(w)
        w = self.w0.copy()
        w[list(erasures)] = w_erase
        m = pymatching.Matching.from_check_matrix(self.H, weights=w, faults_matrix=self.O)
        pred, wt = m.decode(det, return_weight=True)
        return int(pred[0]), float(wt)


class D4Experiment:
    """Noisy KZH protocol + decoders.  run(shots) returns per-shot logical infidelity samples
    (0, 1/2 or 1; Takada Sec. VI D with the infidelity (1 - <H>)/2 of the output) and diagnostics."""

    def __init__(self, L, R0, Rin, Rout, noise, delay=0, pre_u_age=0, n_pre_u=1, flux_informed=True,
                 quiet_window=False):
        self.pre_u_age = pre_u_age
        self.flux_informed = flux_informed
        self.circ = D4SwitchingCircuit(L, R0, Rin, Rout, n_pre_u=n_pre_u)
        self.comp = CompiledProtocol(self.circ, noise)
        self.noise = noise
        self.flux = FluxDecoder(self.comp, delay=delay)
        self.charges = [ChargeDecoder(self.comp, aligned=a) for a in (True, False)]
        # diagnostic (not a physical model): no noise in the switching window = the last D4 round, the Z
        # rounds before U, U, the switch-out and the first FSC round, where the single faults that our
        # decoders fail to correct are located (validation N5).  Gives an optimistic bound.  The decoders keep
        # the priors of the full noise model (they are built before the window is silenced).
        self.quiet_window = quiet_window
        if quiet_window:
            win = {('S1', Rin, 'z'), ('S1', Rin, 'x'), ('U',), ('SO',), ('S4', 1)}
            win |= {('S1', r, 'z') for r in range(Rin + 1, Rin + 1 + n_pre_u)}
            for blk in self.comp.blocks:
                if blk['name'] in win:
                    a, b = blk['moms'][0].loc_slice[0], blk['moms'][-1].loc_slice[1]
                    self.comp.loc_p[a:b] = 0.0
        self.gset = set(self.circ.data_g)

    def shot(self, rng, decode=True, forced=None):
        circ, comp, fl = self.circ, self.comp, self.flux
        sh = FastShot(comp, rng, forced)
        fl.start()
        outs = []
        kfr = 0
        m_g_last = None
        flux_events = []
        stats = dict(n_err=sh.n_err, n_corr=0)
        names = [b['name'] for b in comp.blocks]
        last_S0 = max([i for i, n in enumerate(names) if n[0] == 'S0'], default=-1)
        last_S1 = max(i for i, n in enumerate(names) if n[0] == 'S1')          # Z-only round before U
        last_S1x = max(i for i, n in enumerate(names) if n[0] == 'S1' and n[2] == 'x')
        for bi, blk in enumerate(comp.blocks):
            nm = blk['name']
            if nm == ('S1', 1, 'z'):
                sh.init_frame(circ.zchecks['FSC'], circ.data_rb)
            res = sh.run_block(bi)
            outs += [o for (_, _, o) in res]
            if bi == last_S1x:
                gl = {circ.lab_of[('D4', q)]: o for (q, b, o) in res if b == 'X'}
                m_g_last = [gl[lab] for lab, _ in circ.gens_g]
            if nm[0] in ('S0', 'S4', 'FIN') or (nm[0] == 'S1' and nm[2] == 'z'):
                code = blk['code']
                zl = {circ.lab_of[(code, q)]: o for (q, b, o) in res if b == 'Z'}
                fl.push_round(kfr, zl)
                kfr += 1
                if decode:
                    flush = bi in (last_S0, last_S1) or nm[0] == 'FIN'
                    # before U: corrections supported only by the newest round stay undecided (decoded
                    # after the switch-out with the S4 / g-readout data)
                    cc = fl.decode(flush=flush, min_age=self.pre_u_age if bi == last_S1 else 0)
                    corr = [q for q, k in cc]
                    if nm[0] == 'S1':
                        # the flux was present during the X-type round preceding this Z round
                        flux_events += [(q, nm[1] - 1) for q in corr if nm[1] >= 2]
                    sh.apply_x(corr)
                    for q in corr:
                        fl.register_correction(q, code)
                    stats['n_corr'] += len(corr)
            elif nm[0] == 'SO':
                readout = {q: o for (q, b, o) in res}
                fl.push_round(kfr, readout=readout)
                kfr += 1
                a_dec = dict(readout)
                if decode:
                    phys = []
                    for q, k in fl.decode(flush=True):
                        if q in self.gset:
                            a_dec[q] ^= 1
                        else:
                            phys.append(q)
                    sh.apply_x(phys)
                    for q in phys:
                        fl.register_correction(q, 'FSC')
                    stats['n_corr'] += len(phys)
                self._switch_out_fix(sh, readout, a_dec, rng, noisy=forced is None and not self.quiet_window)
        # Pauli frame of the S^g eigenvalues (last S1 round), then the logical Clifford stabilizer
        for v, m in enumerate(m_g_last):
            if m:
                for q in circ.frame_strings[v]:
                    sh.sim.z(q)
        val = sh.sim.peek_observable_expectation(circ.logical_proxy(sh.fx))
        if decode:
            sols = [ch.predict(outs, ch.flux_erasures(flux_events) if self.flux_informed else None)
                    for ch in self.charges]
            flip = min(sols, key=lambda s: s[1])[0]
        else:
            flip = 0
        val = val * (-1) ** flip
        stats['n_dress'] = sh.n_dress
        return (1 - val) / 2, stats

    def _switch_out_fix(self, sh, readout, a_dec, rng, noisy=True):
        """kappa = Gamma(a): the virtual network applied CZ(Gamma(a_raw)); the protocol applies
        CZ(Gamma(a_dec)) physically -> add the difference (Clifford) and the noise of the physical CZs."""
        circ = self.circ
        ar = np.array([readout[q] for q in circ.data_g], dtype=np.uint8)
        ad = np.array([a_dec[q] for q in circ.data_g], dtype=np.uint8)
        kr = (circ.Gamma.astype(int) @ ar) % 2
        kd = (circ.Gamma.astype(int) @ ad) % 2
        p = self.noise.p
        for v in range(len(circ.gens_g)):
            if kr[v] ^ kd[v]:
                for (i, j) in circ.gpairs[v]:
                    sh.sim.cz(i, j)
            if kd[v] and p > 0 and noisy:
                for (i, j) in circ.gpairs[v]:
                    if rng.random() < p:
                        ps = PAULI_CHOICES[2][rng.integers(15)]
                        sh._pauli(ps, (i, j))

    def run(self, shots, rng, decode=True):
        inf = np.zeros(shots)
        st = []
        for s in range(shots):
            inf[s], d = self.shot(rng, decode)
            st.append(d)
        return inf, st


# ------------------------------------------------------------------ applicability of Takada et al.
def takada_conditions(L):
    """Algebraic checks of the sufficient conditions of Takada et al. (Definition 2 and Sec. VIII) for
    the D4 code switching protocol on the L x L lattice.  Returns dict name -> (passed, detail)."""
    P = D4Protocol(L)
    lat, d4, fsc = P.lat, P.d4, P.fsc
    res = {}
    # (a) admissible factorisation: every X-type Clifford stabilizer = X factors x CZ factors with disjoint
    #     supports (transversal, so every effectively measured operator stays Hermitian, Eq. (131))
    bad = 0
    for lab, op in d4.xgens:
        qs = list(op.s) + [q for pr in op.quad for q in pr]
        bad += len(qs) != len(set(qs))
    res['factors transversal (X and CZ factors on disjoint qubits)'] = (bad == 0, '%d generators' % len(d4.xgens))
    # (b) Pauli dressings: X/Y errors on the Clifford factors (CZ; S, CZ in the logical stabilizer; S,
    #     CZ dressings) give weight-1 Z dressings; all dressings commute (Eq. (91))
    I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Zm = np.diag([1, -1])
    CZm = np.diag([1, 1, 1, -1]); Sm = np.diag([1, 1j])
    ok = True
    for E in (X, Y):
        D = np.kron(E, I2) @ CZm @ np.kron(E, I2).conj().T @ CZm.conj().T       # [E_1, CZ]
        ok &= np.allclose(np.abs(D), np.abs(np.kron(I2, Zm)))
        D1 = E @ Sm @ E.conj().T @ Sm.conj().T
        ok &= np.allclose(np.abs(D1), np.abs(Zm))
    res['dressings of CZ / S by X, Y are single-qubit Z (dressing axis Z on every qubit)'] = (bool(ok), '')
    # (c) effective commutation: group commutators of measured generators are Z-type Paulis in the
    #     persistent Z-stabilizer span (D4 Z checks), Eq. (58) / Lemma 1
    HZr, HZp = f2_rref(d4.HZ)
    nbad, npair = 0, 0
    for (l1, a), (l2, b) in itertools.combinations(d4.xgens, 2):
        lin, const = commutator_affine(a, b)
        npair += 1
        if lin and not f2_in_rowspace(HZr, HZp, lat.vec(lin)):
            nbad += 1
    res['D4 generators: commutators in the Z-stabilizer span'] = (nbad == 0, '%d pairs' % npair)
    # switch-in: S^g measured on FSC|+_L> (x) |0>_g: commutators with the FSC generators and the logical X
    # must lie in the span of the FSC Z checks and the single-qubit Z^g (current Z stabilizers)
    # The state before switch-in, FSC|+_L> (x) |0>_g, is stabilized by the FSC Z checks, the single-qubit
    # Z^g, and by the D4-dressed X-type operators S^r_v, S^b_v (their CZ^{g,.} dressing acts trivially
    # on |0>_g) and a dressed logical X = X^t D (D diagonal Clifford, also trivial on |0>_g).  Takada
    # Sec. VIII B: the generators may be chosen such that the new measurements (S^g) effectively commute
    # or anticommute with them.  Check this choice:
    zin = [rows for lab, rows in lat.z_checks((R, B))] + [{lat.q(e, G)} for e in range(lat.E)]
    Hin = np.array([lat.vec(r) for r in zin], dtype=np.uint8)
    Hir, Hip = f2_rref(Hin)
    gens_in = [op for lab, op in d4.xgens if lab[2] != 'g']
    nbad = 0
    for lab, op in d4.xgens:
        for g in gens_in:
            lin, const = commutator_affine(op, g)
            if lin and not f2_in_rowspace(Hir, Hip, lat.vec(lin)):
                nbad += 1
    # dressed logical X: find D = prod CZ^{c_k} over the CZ pairs of the D4 generators with
    # [X^t D, S] in the D4 Z-stabilizer span for every X-type generator S (linear system over F2)
    t = set(int(q) for q in np.nonzero(fsc.t_log)[0])
    pairs = sorted({pr for _, op in d4.xgens for pr in op.quad})
    rows_eq = []                          # for each generator: commutator linear part as function of c
    HZr2, HZp2 = f2_rref(d4.HZ)
    # basis of the complement of the Z span to express "lin in span" as linear equations
    Npre = f2_nullspace(d4.HZ)            # vectors orthogonal to the span: lin in span <=> N.lin = 0
    A_cols, b_vec = [], None
    base_list = []
    for lab, op in d4.xgens:
        lin0, _ = commutator_affine(PSOp(s=t), op)
        base_list.append(lat.vec(lin0) if lin0 else np.zeros(lat.n, dtype=np.uint8))
    for pr in pairs:
        col = []
        for lab, op in d4.xgens:
            lin, _ = commutator_affine(PSOp(quad=[pr]), op)
            col.append(lat.vec(lin) if lin else np.zeros(lat.n, dtype=np.uint8))
        A_cols.append(col)
    # equations: N @ (base_v + sum_k c_k col_k,v) = 0 for all v
    eqA = np.array([[int((Npre.astype(int) @ A_cols[k][v].astype(int) % 2)[i]) for k in range(len(pairs))]
                    for v in range(len(d4.xgens)) for i in range(Npre.shape[0])], dtype=np.uint8)
    eqb = np.array([int((Npre.astype(int) @ base_list[v].astype(int) % 2)[i])
                    for v in range(len(d4.xgens)) for i in range(Npre.shape[0])], dtype=np.uint8)
    sol = f2_solve(eqA.T, eqb) if eqA.size else np.zeros(0, dtype=np.uint8)
    logical_ok = sol is not None
    if logical_ok:
        D = [pr for pr, cbit in zip(pairs, sol) if cbit]
        Xbar = PSOp(s=t, quad=D)
        # dressing trivial on |0>_g: every CZ of D touches a g qubit
        triv = all(any(q % 3 == G for q in pr) for pr in D)
        for lab, op in d4.xgens:
            lin, const = commutator_affine(Xbar, op)
            if lin and not f2_in_rowspace(Hir, Hip, lat.vec(lin)):
                nbad += 1
        logical_ok = triv
    res['switch-in: S^g vs dressed generators of FSC|+_L>|0>_g (current Z span), dressed logical X exists'] = (
        nbad == 0 and logical_ok, 'logical dressing with %s CZ pairs' % (len(D) if logical_ok else '-'))
    # (d) primary-support exactness (Definition 1): X supports of the X-type generators (and the logical
    #     X) are linearly independent, so no product of them is a non-Pauli operator without X part
    Ssup = np.vstack([d4.S, d4.t_log[None, :]])
    rk = f2_rank(Ssup[:, d4.cols])
    res['primary-support exactness (independent X supports)'] = (rk == Ssup.shape[0], 'rank %d / %d' % (rk, Ssup.shape[0]))
    # (e) the transversal U and the deferred switch-out network consist of CNOTs and diagonal third-level
    #     gates only (Sec. VIII A, Eq. (178)): Pauli / diagonal-Clifford errors stay of that form
    names = {g[0] for g in P.U_gates()}
    res['U = CNOT + diagonal 3rd-level gates (Sec. VIII A)'] = (names <= {'CNOT', 'CS', 'CSDG', 'TDG'}, str(sorted(names)))
    # (f) logical Clifford stabilizer of the output: Clifford (no cubic / odd terms) and effectively
    #     commuting with every FSC X check (dressing = Z string in the FSC Z span with sign +-1)
    circ = D4SwitchingCircuit(L, 1, 1, 1)
    Hf = np.array([[1 if q in qs else 0 for q in range(circ.nq)] for lab, qs in circ.zchecks['FSC']], dtype=np.uint8)
    Hfr, Hfp = f2_rref(Hf)
    nbad = 0
    for lab, xs, quad in circ.xgens['FSC']:
        fx = np.zeros(circ.nq, dtype=np.uint8)
        fx[xs] = 1
        R_ = circ.logical_proxy(fx) * circ.logical_proxy(np.zeros(circ.nq, dtype=np.uint8))
        zpart = np.array([1 if R_[q] in (2, 3) else 0 for q in range(circ.nq)], dtype=np.uint8)
        xpart = np.array([1 if R_[q] in (1, 2) else 0 for q in range(circ.nq)], dtype=np.uint8)
        if xpart.any() or (zpart.any() and not f2_in_rowspace(Hfr, Hfp, zpart)) or R_.sign.imag != 0:
            nbad += 1
    res['output logical stabilizer: Clifford, effectively commutes with FSC X checks'] = (nbad == 0, '')
    return res


def lockstep_validation(L, Rin, Rout, noise, rng, sequential=False, corr_prob=0.05):
    """Run the Pauli-proxy shot (FastShot) and the exact simulation of the TRUE circuit (TrueSim) in
    lockstep from the start of S1 (initial state E|+_L>_FSC|0>_g with a random Pauli E on the FSC data),
    with the same sampled circuit-level errors and random feed-forward X corrections.  Returns dict with
    the number of measurements, the number of mismatches (deterministic vs random, probability of the
    outcome), and the final <H> from both simulations."""
    circ = D4SwitchingCircuit(L, 0, Rin, Rout, sequential=sequential)
    comp = CompiledProtocol(circ, noise)
    P, c, fsc = circ.P, circ.c, circ.P.fsc
    zero = coset_state(fsc, set())
    one = coset_state(fsc, set(int(q) for q in np.nonzero(fsc.t_log)[0]))
    amps = {}
    for st in (zero, one):
        for k, v in st.a.items():
            amps[k] = amps.get(k, 0) + v / np.sqrt(2)
    keys = [sum(1 << c[q] for q in k) for k in amps]
    ts = TrueSim(circ.nd, keys, list(amps.values()))
    sh = FastShot(comp, rng)
    n = circ.nq
    stabs = []
    for lab, xs, quad in circ.xgens['FSC']:
        ps = stim.PauliString(n)
        for q in xs:
            ps[q] = 'X'
        stabs.append(ps)
    for lab, qs in circ.zchecks['FSC']:
        ps = stim.PauliString(n)
        for q in qs:
            ps[q] = 'Z'
        stabs.append(ps)
    ps = stim.PauliString(n)
    for q in circ.t_log:
        ps[q] = 'X'
    stabs.append(ps)
    for q in range(n):
        if q not in circ.data_rb:
            ps = stim.PauliString(n)
            ps[q] = 'Z'
            stabs.append(ps)
    sh.sim.set_state_from_stabilizers(stabs, allow_redundant=True)
    for q in rng.choice(circ.data_rb, 2, replace=False):
        pc = 'XYZ'[rng.integers(3)]
        getattr(ts, pc)(int(q)); getattr(sh.sim, pc.lower())(int(q))
    sh.init_frame(circ.zchecks['FSC'], circ.data_rb)
    nmeas = nbad = nrand = 0
    m_g = None
    for bi, blk in enumerate(comp.blocks):
        res = sh.run_block_lockstep(bi, ts, rng)
        for q, bb, o, pk, pr in res:
            nmeas += 1
            if pk == 0:
                nrand += 1
                nbad += abs(pr - 0.5) > 1e-9
            else:
                nbad += (o != (0 if pk == 1 else 1)) or abs(pr - 1) > 1e-9
        nm = blk['name']
        if nm[0] == 'S1' and nm[2] == 'x':
            gl = {circ.lab_of[('D4', q)]: o for q, bb, o, pk, pr in res if bb == 'X'}
            m_g = [gl[lab] for lab, _ in circ.gens_g]
        if corr_prob and nm[0] in ('S1', 'S4') and rng.random() < corr_prob * len(blk['moms']):
            q = int(rng.choice(circ.data_rb if nm[0] == 'S4' else circ.data_all))
            sh.apply_x([q]); ts.X(q)
    for v, m in enumerate(m_g):
        if m:
            for q in circ.frame_strings[v]:
                sh.sim.z(q); ts.Z(q)
    true = ts.expect_logical(circ)
    proxy = sh.sim.peek_observable_expectation(circ.logical_proxy(sh.fx))
    return dict(n_meas=nmeas, n_random=nrand, n_mismatch=int(nbad), H_true=true, H_proxy=proxy,
                n_dress=sh.n_dress, agree=bool(nbad == 0 and abs(true - proxy) < 1e-9))
