"""d4_lib: D4 Clifford stabilizer code (Kobayashi-Zhu-Hsin) on the bounded square lattice, transversal U, code switching, resource model."""
import itertools, cmath
import numpy as np
from collections import Counter

R, G, B = 0, 1, 2
CNAME = "rgb"

# ------------------------------------------------- phase-shift operators
class PSOp:
    """Operator  O |x> = (-1)^{f(x)} |x + s>  on n qubits (computational basis x in F2^n).

    f(x) = const + sum_{i in lin} x_i + sum_{(i,j) in quad} x_i x_j   (mod 2), i<j.
    X-type Clifford stabilizers of the D4 code, Pauli X/Z strings and products
    thereof all have this form.  Composition rule (A o B)|x> = A B |x>:
        s = s_A + s_B,   f(x) = f_B(x) + f_A(x + s_B).
    """
    __slots__ = ("s", "lin", "quad", "const")

    def __init__(self, s=frozenset(), lin=frozenset(), quad=frozenset(), const=0):
        self.s = frozenset(s)
        self.lin = frozenset(lin)
        self.quad = frozenset(tuple(sorted(p)) for p in quad)
        self.const = const & 1

    def shifted_phase(self, t):
        """Return (lin, quad, const) of x -> f(x + t)."""
        lin = set(self.lin)
        const = self.const
        for i in self.lin:
            if i in t:
                const ^= 1
        for (i, j) in self.quad:
            ti, tj = i in t, j in t
            if ti and tj:
                const ^= 1
            if tj:
                lin ^= {i}
            if ti:
                lin ^= {j}
        return frozenset(lin), self.quad, const

    def __matmul__(self, other):
        lA, qA, cA = self.shifted_phase(other.s)
        return PSOp(self.s ^ other.s, lA ^ other.lin, qA ^ other.quad, cA ^ other.const)

    def phase_at(self, x):
        """f(x) for x given as a set/frozenset of 1-positions."""
        v = self.const
        for i in self.lin:
            if i in x:
                v ^= 1
        for (i, j) in self.quad:
            if i in x and j in x:
                v ^= 1
        return v

    def apply(self, x):
        """O|x> = sign |x'> ; returns (x', sign_bit)."""
        return frozenset(x) ^ self.s, self.phase_at(x)

    def support(self):
        sup = set(self.s) | set(self.lin)
        for p in self.quad:
            sup |= set(p)
        return sup


def commutator_affine(A, B):
    """A B = (-1)^{g(x)} B A on |x>.  Returns g as (lin, const) -- quadratic terms cancel."""
    AB = A @ B
    BA = B @ A
    assert AB.s == BA.s
    q = AB.quad ^ BA.quad
    assert not q, "commutator has quadratic part (should not happen)"
    return AB.lin ^ BA.lin, AB.const ^ BA.const


# ------------------------------------------------------------- the lattice
class D4Lattice:
    def __init__(self, L):
        assert L >= 1
        self.L = L
        self.edges = []          # (kind, x, y), kind 'h' or 'v'
        self.eidx = {}
        for y in range(L + 1):
            for x in range(L):
                self._add(('h', x, y))
        for x in range(L + 1):
            for y in range(L):
                self._add(('v', x, y))
        self.E = len(self.edges)
        self.n = 3 * self.E
        self.vertices = [(x, y) for y in range(L + 1) for x in range(L + 1)]
        self.plaquettes = [(x, y) for y in range(L) for x in range(L)]

    def _add(self, e):
        self.eidx[e] = len(self.edges)
        self.edges.append(e)

    def q(self, e, c):
        if not isinstance(e, int):
            e = self.eidx[e]
        return 3 * e + c

    # --- geometry --------------------------------------------------------
    def endpoints(self, e):
        k, x, y = self.edges[e] if isinstance(e, int) else e
        return ((x, y), (x + 1, y)) if k == 'h' else ((x, y), (x, y + 1))

    def incident(self, v):
        x, y = v
        out = []
        for e in [('h', x, y), ('h', x - 1, y), ('v', x, y), ('v', x, y - 1)]:
            if e in self.eidx:
                out.append(self.eidx[e])
        return out

    def plaq_edges(self, p):
        """(e01, e13, e02, e23) of plaquette p=(x,y)."""
        x, y = p
        ei = self.eidx
        return ei[('h', x, y)], ei[('v', x + 1, y)], ei[('v', x, y)], ei[('h', x, y + 1)]

    def plaq_boundary(self, p):
        return list(self.plaq_edges(p))

    def has_plaq(self, x, y):
        return 0 <= x < self.L and 0 <= y < self.L

    def boundary_of_edge(self, e):
        k, x, y = self.edges[e]
        L = self.L
        if k == 'v' and x == 0:
            return 'Lr'
        if k == 'h' and y == L:
            return 'Lb'
        if k == 'h' and y == 0:
            return 'Lrb'
        if k == 'v' and x == L:
            return 'Lrb'
        return None

    def boundary_edges(self, name):
        return [e for e in range(self.E) if self.boundary_of_edge(e) == name]

    # --- Z-type checks (rows of H_Z) --------------------------------------
    def z_checks(self, colours=(R, G, B)):
        """List of (label, set_of_qubits) Z-type stabilizer generators.
        Plaquettes for every colour + single-edge boundary conditions (Fig. 4 / eq. (7))."""
        rows = []
        for p in self.plaquettes:
            for c in colours:
                rows.append((('plaq', p, CNAME[c]), {self.q(e, c) for e in self.plaq_boundary(p)}))
        for e in range(self.E):
            bd = self.boundary_of_edge(e)
            if bd == 'Lr' and R in colours:
                rows.append((('bdZ', e, 'r'), {self.q(e, R)}))
            elif bd == 'Lb' and B in colours:
                rows.append((('bdZ', e, 'b'), {self.q(e, B)}))
            elif bd == 'Lrb':
                if R in colours and B in colours:
                    rows.append((('bdZ', e, 'rb'), {self.q(e, R), self.q(e, B)}))
                if G in colours:
                    rows.append((('bdZ', e, 'g'), {self.q(e, G)}))
        return rows

    # --- X-type Clifford stabilizers ----------------------------------------
    def cz_pairs(self, v, c):
        """CZ pairs (qubit, qubit) of the dressing of S^c_{X;v} (SM eq. for S_X, cup products
        evaluated with the hypercube formula; plaquettes outside the lattice are dropped)."""
        x, y = v
        pairs = []
        if c == R:      # int v u e' u e'' : v = vertex 0 of p(x,y);   CZ^{g,b}
            if self.has_plaq(x, y):
                e01, e13, e02, e23 = self.plaq_edges((x, y))
                pairs += [(self.q(e01, G), self.q(e13, B)), (self.q(e02, G), self.q(e23, B))]
        elif c == G:    # int e'' u v u e' : v = vertex 1 of p(x-1,y), vertex 2 of p(x,y-1); CZ^{r,b}
            if self.has_plaq(x - 1, y):
                e01, e13, e02, e23 = self.plaq_edges((x - 1, y))
                pairs.append((self.q(e01, R), self.q(e13, B)))
            if self.has_plaq(x, y - 1):
                e01, e13, e02, e23 = self.plaq_edges((x, y - 1))
                pairs.append((self.q(e02, R), self.q(e23, B)))
        else:           # int e' u e'' u v : v = vertex 3 of p(x-1,y-1); CZ^{r,g}
            if self.has_plaq(x - 1, y - 1):
                e01, e13, e02, e23 = self.plaq_edges((x - 1, y - 1))
                pairs += [(self.q(e01, R), self.q(e13, G)), (self.q(e02, R), self.q(e23, G))]
        return pairs

    def SX(self, v, c, dressed=True):
        s = {self.q(e, c) for e in self.incident(v)}
        quad = self.cz_pairs(v, c) if dressed else []
        # CZ applied first, then X (S = (prod X)(prod CZ)); X and CZ act on disjoint colours.
        return PSOp(s=s, quad=quad)

    def vertex_x_types(self, v, colours=(R, G, B)):
        """Allowed colour combinations at vertex v: the F2 span of colour sets whose
        X-part commutes with the boundary Z conditions.  Returns a list of colour tuples
        (a basis), derived automatically from the boundary Z checks."""
        bdrows = [rows for (lab, rows) in self.z_checks(colours) if lab[0] == 'bdZ']
        inc = self.incident(v)
        cand = []
        for k in range(1, len(colours) + 1):
            for combo in itertools.combinations(colours, k):
                s = {self.q(e, c) for e in inc for c in combo}
                if all(len(s & rw) % 2 == 0 for rw in bdrows):
                    cand.append(combo)
        # reduce to an F2 basis over colour space, preferring single colours
        basis, vecs = [], []
        for combo in sorted(cand, key=len):
            vec = np.zeros(3, dtype=np.uint8)
            vec[list(combo)] = 1
            if f2_rank(np.array(vecs + [vec])) > len(vecs):
                vecs.append(vec)
                basis.append(combo)
        return basis

    def x_generators(self, colours=(R, G, B), dressed=True):
        """List of (label, PSOp).  At a vertex where only a product of colours is allowed
        (L_rb), the generator is the ordered product S^r S^b (Fig. 4 shows S^r S^b there)."""
        gens = []
        for v in self.vertices:
            for combo in self.vertex_x_types(v, colours):
                op = PSOp()
                for c in combo:
                    op = op @ self.SX(v, c, dressed)
                gens.append((('X', v, ''.join(CNAME[c] for c in combo)), op))
        return gens

    def vec(self, qubits):
        v = np.zeros(self.n, dtype=np.uint8)
        v[list(qubits)] = 1
        return v


# ---------------------------------------------------------- code data
class CodeData:
    """Linear-algebra data of a (D4 or folded-surface) code on a lattice."""

    def __init__(self, lat, colours=(R, G, B), dressed=True):
        self.lat = lat
        self.colours = tuple(colours)
        self.zlabels, zr = zip(*lat.z_checks(colours))
        self.HZ = np.array([lat.vec(r) for r in zr], dtype=np.uint8)
        self.xgens = lat.x_generators(colours, dressed)
        self.S = np.array([lat.vec(op.s) for _, op in self.xgens], dtype=np.uint8)
        self.cols = sorted({lat.q(e, c) for e in range(lat.E) for c in colours})
        self.HZr, self.HZpiv = f2_rref(self.HZ)
        self.Sr, self.Spiv = f2_rref(self.S)
        # logical representatives
        CZ = f2_nullspace(self.HZ[:, self.cols])
        self.t_log = None
        for row in CZ:
            t = np.zeros(lat.n, dtype=np.uint8)
            t[self.cols] = row
            if not f2_in_rowspace(self.Sr, self.Spiv, t):
                self.t_log = t
                break
        KS = f2_nullspace(self.S[:, self.cols])
        self.z_log = None
        for row in KS:
            u = np.zeros(lat.n, dtype=np.uint8)
            u[self.cols] = row
            if not f2_in_rowspace(self.HZr, self.HZpiv, u):
                self.z_log = u
                break
        self.k = len(self.cols) - f2_rank(self.HZ[:, self.cols]) - f2_rank(self.S[:, self.cols])


def distances(cd):
    """(d_Z, d_X_shift) : min weight of a logical Z-type Pauli, and min weight of an
    X-shift lying in C_Z but outside the stabilizer shift span."""
    dz, uz = min_weight_odd(cd.S, cd.t_log, cd.cols)
    dx, tx = min_weight_odd(cd.HZ, cd.z_log, cd.cols)
    return dz, dx, uz, tx


# ------------------------------------------- transversal automorphism U = W V
class DiagPhase:
    """Diagonal unitary |x> -> omega^{w(x)} |x>, omega = exp(i pi/4), w integer mod 8.
    w(x) = sum_k c_k * prod_{i in mono_k} x_i  (monomials of degree <= 2)."""

    def __init__(self, terms=()):
        self.terms = list(terms)   # (coef mod 8, tuple_of_qubits)

    def value(self, x):
        v = 0
        for c, mono in self.terms:
            if all(i in x for i in mono):
                v += c
        return v % 8

    def support(self):
        s = set()
        for _, m in self.terms:
            s |= set(m)
        return s

    def terms_touching(self, qs):
        return DiagPhase([(c, m) for c, m in self.terms if set(m) & qs])


def W_operator(lat, cs_signs=(+1, -1), boundary_T=-1):
    """W of eq. (8):  prod_p CS^{r,b}_{e01,e13} (CS^{r,b}_{e02,e23})^dag  x  prod_{e in bdry_rb} (T^r_e)^dag.
    CS = diag(1,1,1,i) -> omega^{2 x y};  T^dag -> omega^{-x}."""
    terms = []
    for p in lat.plaquettes:
        e01, e13, e02, e23 = lat.plaq_edges(p)
        terms.append(((2 * cs_signs[0]) % 8, (lat.q(e01, R), lat.q(e13, B))))
        terms.append(((2 * cs_signs[1]) % 8, (lat.q(e02, R), lat.q(e23, B))))
    if boundary_T:
        for e in lat.boundary_edges('Lrb'):
            terms.append((boundary_T % 8, (lat.q(e, R),)))
    return DiagPhase(terms)


def V_map(lat, x):
    """V = prod_e CNOT^{(r,g)}_e CNOT^{(b,g)}_e :  x_g <- x_g + x_r + x_b  (set representation)."""
    x = set(x)
    out = set(x)
    for e in range(lat.E):
        qr, qg, qb = lat.q(e, R), lat.q(e, G), lat.q(e, B)
        if (qr in x) ^ (qb in x):
            out ^= {qg}
    return frozenset(out)


def V_support_map(lat, qs):
    """Qubits whose values determine (V x) restricted to qs."""
    out = set()
    for q in qs:
        e, c = divmod(q, 3)
        out.add(q)
        if c == G:
            out |= {3 * e + R, 3 * e + B}
    return out


def CZ_basis(cd):
    if not hasattr(cd, '_CZbasis'):
        nb = f2_nullspace(cd.HZ[:, cd.cols])
        full = np.zeros((nb.shape[0], cd.lat.n), dtype=np.uint8)
        full[:, cd.cols] = nb
        cd._CZbasis = full
    return cd._CZbasis


def project_CZ(cd, Q, functionals=()):
    """Exact image of C_Z under x -> (x|_Q, phi_1(x), ..., phi_k(x)) for F2-linear functionals
    phi_j (callables on a set of qubits, evaluated on basis vectors).  Returns a list of
    configurations (frozenset of Q-qubits) or, if functionals are given, (config, bits)."""
    Q = sorted(Q)
    Bs = CZ_basis(cd)
    P = Bs[:, Q]
    if functionals:
        extra = np.array([[phi(frozenset(np.nonzero(b)[0])) for phi in functionals] for b in Bs],
                         dtype=np.uint8).reshape(len(Bs), len(functionals))
        P = np.concatenate([P, extra], axis=1)
    Pr, piv = f2_rref(P)
    k = Pr.shape[0]
    assert k <= 24, f"local projection too large ({k})"
    out = []
    nq = len(Q)
    for bits in itertools.product((0, 1), repeat=k):
        v = np.zeros(P.shape[1], dtype=np.uint8)
        for b, row in zip(bits, Pr):
            if b:
                v ^= row
        cfg = frozenset(Q[i] for i in np.nonzero(v[:nq])[0])
        out.append((cfg, tuple(int(z) for z in v[nq:])) if functionals else cfg)
    return out


def check_conjugation(cd, Wop, gen_op, target_op):
    """Exhaustively check  U S U^dag == S'  on C_Z, where U = W V.
    Returns (ok, n_local_configs)."""
    lat = cd.lat
    As = V_map(lat, gen_op.s)
    if As != target_op.s:
        return False, 0
    # qubits on which the phase difference can depend
    Wt = Wop.terms_touching(set(As))
    Q = set(Wt.support()) | V_support_map(lat, gen_op.support()) | target_op.support() | set(As)
    configs = project_CZ(cd, Q)
    for x in configs:
        # omega^{ w(x+As) - w(x) + 4 f(A x) - 4 f'(x) }   (only W-terms touching As contribute)
        d = Wt.value(x ^ As) - Wt.value(x) + 4 * gen_op.phase_at(V_map(lat, x)) - 4 * target_op.phase_at(x)
        if d % 8:
            return False, len(configs)
    return True, len(configs)


def stabilizer_path_phase(cd, x, target):
    """Given x and target = x + (sum of generator shifts), apply generators to |x> and
    return the accumulated sign bit, i.e. psi(target) = (-1)^bit psi(x) for code states."""
    shifts = cd.S
    delta = cd.lat.vec(set(x) ^ set(target))
    coeff = f2_solve(shifts, delta)
    if coeff is None:
        return None
    bit = 0
    cur = frozenset(x)
    for i in np.nonzero(coeff)[0]:
        op = cd.xgens[i][1]
        cur, b = op.apply(cur)
        bit ^= b
    assert cur == frozenset(target)
    return bit


def logical_phases(cd, Wop):
    """Eigenphases (in units of pi/4) of U on |0_L>, |1_L>, assuming U preserves the code space
    (checked separately).  Uses x0 = 0 and x1 = t_log."""
    lat = cd.lat
    res = []
    for x in [frozenset(), frozenset(np.nonzero(cd.t_log)[0])]:
        Ax = V_map(lat, x)
        bit = stabilizer_path_phase(cd, x, Ax)
        assert bit is not None, "V does not map the coset to itself"
        res.append((Wop.value(Ax) + 4 * bit) % 8)
    return res


# ------------------------------------------- per-configuration protocol phase
def lrb_vertices(lat):
    vs = set()
    for e in lat.boundary_edges('Lrb'):
        vs |= set(lat.endpoints(e))
    return vs


def cprime(lat, a):
    """0-cochain c' with  delta c' = a_r + a_b  and c' = 0 on L_rb vertices (BFS).
    a: set of qubits (configuration of r,b qubits).  Returns dict vertex -> bit.
    Raises if a_r + a_b is not exact relative to L_rb."""
    from collections import deque
    val = {v: 0 for v in lrb_vertices(lat)}
    dq = deque(val)
    while dq:
        u = dq.popleft()
        for e in lat.incident(u):
            p, q = lat.endpoints(e)
            w = q if p == u else p
            bit = val[u] ^ (lat.q(e, R) in a) ^ (lat.q(e, B) in a)
            if w in val:
                if val[w] != bit:
                    raise ValueError("a_r + a_b not exact")
            else:
                val[w] = bit
                dq.append(w)
    return val


def g_phase_poly(lat, v):
    """CZ^{r,b} pairs of S^g_{X;v}  (f^g_v)."""
    return lat.cz_pairs(v, G)


def protocol_phase(lat, Wop, a, gverts):
    """Phase exponent (mod 8, units of pi/4) acquired by FSC configuration a through
    switch-in (S^g measurement) -> U -> Z^g measurement -> S^g correction -> Pauli frame,
    i.e.  F(a) = w(a) + 4 * sum_v c'_v(a) f^g_v(a)."""
    cp = cprime(lat, a)
    phi = 0
    for v in gverts:
        if cp[v]:
            for (i, j) in g_phase_poly(lat, v):
                if i in a and j in a:
                    phi ^= 1
    return (Wop.value(a) + 4 * phi) % 8


def check_protocol_invariance(lat, fsc, Wop, gverts, gen_op):
    """Exhaustive local proof that F(a + t) = F(a) (mod 8) for all a in C_FSC, t = shift of gen_op."""
    t = frozenset(gen_op.s)
    # c'(t) (linear in a)
    ct = cprime(lat, t)
    N2 = {v for v in gverts if ct[v]}
    N1 = {v for v in gverts if set(q for pr in g_phase_poly(lat, v) for q in pr) & t}
    Wt = Wop.terms_touching(set(t))
    Q = set(Wt.support()) | set(t)
    for v in N1 | N2:
        Q |= {q for pr in g_phase_poly(lat, v) for q in pr}
    # paths from a reference vertex w0 to each w in N1 (BFS on the lattice graph)
    w0 = min(N1 | N2 | {lat.endpoints(next(iter(t)) // 3)[0]})
    paths = {}
    from collections import deque
    prev = {w0: None}
    dq = deque([w0])
    while dq:
        u = dq.popleft()
        for e in lat.incident(u):
            p, q = lat.endpoints(e)
            w = q if p == u else p
            if w not in prev:
                prev[w] = (u, e)
                dq.append(w)
    for w in N1:
        es = []
        cur = w
        while prev[cur] is not None:
            u, e = prev[cur]
            es.append(e)
            cur = u
        paths[w] = es
        Q |= {lat.q(e, c) for e in es for c in (R, B)}
    beta_fn = lambda b: cprime(lat, b)[w0]          # c'_{w0}(a) is F2-linear in a
    configs = project_CZ(fsc, Q, functionals=[beta_fn])
    for a, (beta,) in configs:
        if True:
            cpa = {}
            for w in N1:
                bit = beta
                for e in paths[w]:
                    bit ^= (lat.q(e, R) in a) ^ (lat.q(e, B) in a)
                cpa[w] = bit
            at = a ^ t
            d = Wt.value(at) - Wt.value(a)
            phi = 0
            for w in N1:
                if cpa[w]:
                    for (i, j) in g_phase_poly(lat, w):
                        phi ^= ((i in at) and (j in at)) ^ ((i in a) and (j in a))
            for w in N2:
                for (i, j) in g_phase_poly(lat, w):
                    phi ^= ((i in at) and (j in at))
            if (d + 4 * phi) % 8:
                return False, len(configs)
    return True, len(configs)


OMEGA = cmath.exp(1j * cmath.pi / 4)


class SparseState:
    def __init__(self, amps):
        self.a = {k: v for k, v in amps.items() if abs(v) > 1e-14}

    def norm2(self):
        return sum(abs(v) ** 2 for v in self.a.values())

    def normalize(self):
        n = self.norm2() ** 0.5
        self.a = {k: v / n for k, v in self.a.items()}
        return self

    def apply_psop(self, op):
        out = {}
        for x, v in self.a.items():
            y, b = op.apply(x)
            out[y] = out.get(y, 0) + (-v if b else v)
        return SparseState(out)

    def apply_diag(self, D):
        return SparseState({x: v * OMEGA ** D.value(x) for x, v in self.a.items()})

    def apply_perm(self, f):
        out = {}
        for x, v in self.a.items():
            y = f(x)
            out[y] = out.get(y, 0) + v
        return SparseState(out)

    def project(self, op, sign):
        """(1 + sign*O)/2 |psi>  (unnormalised)."""
        Opsi = self.apply_psop(op).a
        out = {}
        for k in set(self.a) | set(Opsi):
            out[k] = 0.5 * (self.a.get(k, 0) + sign * Opsi.get(k, 0))
        return SparseState(out)

    def expect(self, op):
        Opsi = self.apply_psop(op).a
        return sum(np.conj(self.a.get(k, 0)) * v for k, v in Opsi.items())

    def measure_op(self, op, rng, forced=None):
        """Projective measurement of a Hermitian involution O; returns (outcome bit, state)."""
        p0 = self.project(op, +1)
        n0 = p0.norm2()
        if forced is None:
            bit = 0 if rng.random() < n0 else 1
        else:
            bit = forced
        st = p0 if bit == 0 else self.project(op, -1)
        prob = n0 if bit == 0 else 1 - n0
        return bit, st.normalize() if prob > 1e-12 else st, prob

    def measure_z(self, q, rng, forced=None):
        p1 = sum(abs(v) ** 2 for x, v in self.a.items() if q in x)
        bit = (1 if rng.random() < p1 else 0) if forced is None else forced
        st = SparseState({x: v for x, v in self.a.items() if (q in x) == bool(bit)})
        prob = p1 if bit else 1 - p1
        return bit, st.normalize() if prob > 1e-12 else st, prob

    def overlap(self, other):
        return sum(np.conj(v) * other.a.get(k, 0) for k, v in self.a.items())


def coset_state(cd, rep, phases=True):
    """Code state supported on rep + span(shifts):  psi(rep)=1, psi(x+s_v)=(-1)^{f_v(x)} psi(x)."""
    amps = {frozenset(rep): 1.0}
    # breadth-first application of generators
    frontier = [frozenset(rep)]
    while frontier:
        new = []
        for x in frontier:
            for _, op in cd.xgens:
                y, b = op.apply(x)
                if y not in amps:
                    amps[y] = amps[x] * (-1 if b else 1)
                    new.append(y)
                else:
                    if phases:
                        assert abs(amps[y] - amps[x] * (-1 if b else 1)) < 1e-9, "inconsistent code state"
        frontier = new
    return SparseState(amps).normalize()




def frozen_qubits(lat):
    """Qubits fixed to |0> by single-qubit boundary Z stabilizers (L_r: r, L_b: b, L_rb: g)."""
    fz = set()
    for lab, rows in lat.z_checks():
        if lab[0] == 'bdZ' and len(rows) == 1:
            fz |= rows
    return fz


class D4Protocol:
    def __init__(self, L, prune_frozen=True):
        self.L = L
        self.lat = lat = D4Lattice(L)
        self.frozen = frozen_qubits(lat) if prune_frozen else set()
        self.prune = prune_frozen
        self.d4 = CodeData(lat)
        self.fsc = CodeData(lat, (R, B), dressed=False)
        # ancilla allocation: one ancilla per stabilizer generator label (D4 superset)
        self.anc = {}
        nxt = lat.n
        labels = [lab for lab, _ in self.d4.xgens] + list(self.d4.zlabels)
        labels += [lab for lab, _ in self.fsc.xgens] + list(self.fsc.zlabels)
        for lab in labels:
            key = self._anc_key(lab)
            if key not in self.anc and not self._is_frozen_check(lab):
                self.anc[key] = nxt
                nxt += 1
        self.gverts = [lab[1] for lab, op in self.d4.xgens if lab[2] == 'g']

    @staticmethod
    def _anc_key(lab):
        return lab

    def _is_frozen_check(self, lab):
        return lab[0] == 'bdZ' and lab[2] in ('r', 'b', 'g') and self.prune

    def data_qubits(self, colours):
        return {self.lat.q(e, c) for e in range(self.lat.E) for c in colours} - self.frozen

    # ---------------------------------------------------------- gadgets
    def _keep_pair(self, i, j):
        return i not in self.frozen and j not in self.frozen

    def x_gadget(self, lab, dressed):
        _, v, cols = lab
        parts = []
        for ch in cols:
            c = CNAME.index(ch)
            xq = {self.lat.q(e, c) for e in self.lat.incident(v)} - self.frozen
            pairs = [p for p in self.lat.cz_pairs(v, c) if self._keep_pair(*p)] if dressed else []
            parts.append((xq, pairs))
        return clifford_x_gadget(self.anc[lab], parts)

    def z_gadget(self, lab, rows):
        return z_check_gadget(self.anc[lab], set(rows) - self.frozen)

    def round_gadgets(self, code, which='all'):
        """Gadgets of one stabilizer round.  code: 'D4' or 'FSC'.
        which: 'all' | 'X' | 'Z' | 'g' (D4 S^g only)."""
        cd = self.d4 if code == 'D4' else self.fsc
        dressed = code == 'D4'
        gads = []
        if which in ('all', 'X', 'g'):
            for lab, op in cd.xgens:
                if which == 'g' and lab[2] != 'g':
                    continue
                gads.append(self.x_gadget(lab, dressed))
        if which in ('all', 'Z'):
            for lab, (l2, rows) in zip(cd.zlabels, self.lat.z_checks(cd.colours)):
                if self._is_frozen_check(lab):
                    continue
                gads.append(self.z_gadget(lab, rows))
        return gads

    # ---------------------------------------------------------- transversal U
    def U_gates(self):
        lat, fz = self.lat, self.frozen
        gates = []
        for e in range(lat.E):
            qr, qg, qb = lat.q(e, R), lat.q(e, G), lat.q(e, B)
            for ctrl in (qr, qb):
                if ctrl in fz:
                    continue
                if qg in fz:
                    continue   # L_rb: CNOT(r,g) CNOT(b,g) = identity on Z^r Z^b = +1 (g removed)
                gates.append(('CNOT', ctrl, qg))
        for p in lat.plaquettes:
            e01, e13, e02, e23 = lat.plaq_edges(p)
            for (a, b, name) in [(lat.q(e01, R), lat.q(e13, B), 'CS'), (lat.q(e02, R), lat.q(e23, B), 'CSDG')]:
                if a in fz or b in fz:
                    continue
                gates.append((name, a, b))
        for e in lat.boundary_edges('Lrb'):
            gates.append(('TDG', lat.q(e, R)))
        return gates

    # ---------------------------------------------------------- switch-out
    def switch_out_gates(self, worst_case=True):
        lat = self.lat
        gq = sorted(self.data_qubits((G,)))
        gates = [('MEAS_Z', q) for q in gq]
        cz = []
        for v in self.gverts:
            for p in lat.cz_pairs(v, G):
                if self._keep_pair(*p):
                    cz.append(('CZ',) + tuple(p))
        return gates, cz      # cz: worst-case correction (all vertices in kappa)

    def qubit_sets(self):
        fsc_data = self.data_qubits((R, B))
        d4_data = self.data_qubits((R, G, B))
        fsc_anc = {self.anc[l] for l, _ in self.fsc.xgens} | {self.anc[l] for l in self.fsc.zlabels
                                                              if not self._is_frozen_check(l)}
        d4_anc = {self.anc[l] for l, _ in self.d4.xgens} | {self.anc[l] for l in self.d4.zlabels
                                                            if not self._is_frozen_check(l)}
        return dict(fsc_data=fsc_data, d4_data=d4_data, fsc_anc=fsc_anc, d4_anc=d4_anc)

# ----------------------------------------------------------------- Kobayashi (D4)
class D4CostData:
    """All d-dependent counts of the D4 protocol, computed once per L from explicit circuits."""

    def __init__(self, L, dur=None, decomposed=False):
        dur = dur or DUR_UNIT
        P = D4Protocol(L)
        self.L = L
        qs = P.qubit_sets()
        self.n = {k: len(v) for k, v in qs.items()}
        self.n_total = len(qs['d4_data'] | qs['d4_anc'])
        dec = decompose if decomposed else (lambda g: g)
        gad = {
            'D4': [dec(g) for g in P.round_gadgets('D4')],
            'D4g': [dec(g) for g in P.round_gadgets('D4', 'g')],
            'FSC': P.round_gadgets('FSC'),
        }
        self.round = {}
        for k, g in gad.items():
            ub = round_schedule_classes(g, dur)
            lb = round_lower_bound(g, dur)
            self.round[k] = dict(time_lb=lb['time_lb'], time_ub=ub['time_ub'], depth_lb=lb['depth_lb'],
                                 depth_ub=ub['depth_ub'], classes=ub['classes'],
                                 gates=Counter(gate_counts([x for gg in g for x in gg])),
                                 n_anc=len(g))
        self.n_anc_g = len(gad['D4g'])
        U = dec(P.U_gates())
        Ul = class_layers([U])              # all U gates mutually commute (validated)
        self.U = dict(gates=Counter(gate_counts(U)), time=circuit_time(Ul, dur), depth=len(Ul))
        meas, cz = P.switch_out_gates()
        czl = class_layers([cz]) if cz else []
        ml = [meas]
        self.out = dict(gates_worst=Counter(gate_counts(meas + cz)),
                        gates_expected=Counter({**gate_counts(meas), 'CZ': len(cz) / 2}),
                        time=circuit_time(ml + czl, dur), depth=len(ml) + len(czl))
        self.x_corr = dict(time=dur['X'], gates_worst=Counter({'X': self.n['fsc_data']}))
        self.init = Counter({'INIT_X': self.n['fsc_data'], 'INIT_Z': len(qs['d4_data']) - self.n['fsc_data']})


def d4_stages(cd, model, params=DEFAULT_PARAMS, comparison='A', data_only=False):
    d = cd.L
    n = cd.n
    fsc_q = n['fsc_data'] + (0 if data_only else n['fsc_anc'])
    d4_q = n['d4_data'] + (0 if data_only else n['d4_anc'])
    rows = []

    def add(stage, qubits, rounds, rkey, extra_time, gates, source, note=''):
        rt = cd.round[rkey] if rkey else dict(time_lb=0, time_ub=0)
        rows.append(dict(protocol='Kobayashi_D4', comparison=comparison, model=model, d=d, stage=stage,
                         qubits=qubits, rounds=rounds,
                         time_lb=rounds * rt['time_lb'] + extra_time,
                         time_ub=rounds * rt['time_ub'] + extra_time,
                         gates=dict(gates), source=source, note=note))

    r0 = rounds_for(model, params['kappa0'], d)
    g0 = Counter(cd.init)
    for k, v in cd.round['FSC']['gates'].items():
        g0[k] += v * r0
    g0['X'] += cd.x_corr['gates_worst']['X']
    add('S0 FSC |+_L> preparation', fsc_q, r0, 'FSC', cd.x_corr['time'], g0,
        SOURCE_ASSUME if model == 'FT_ref' else SOURCE_COUNT,
        'kappa0*d rounds (not specified in Kobayashi); X-correction worst case counted')
    if model == 'NL_min':
        r1, key = 1, 'D4g'
        q1 = n['d4_data'] + (0 if data_only else cd.n_anc_g)
        note1 = 'noiseless minimum: one round of S^g_X measurements only'
    else:
        r1, key = rounds_for(model, params['kappa_in'], d), 'D4'
        q1 = d4_q
        note1 = 'O(d) rounds of all D4 stabilizers (Kobayashi step 2); kappa_in*d assumed'
    g1 = Counter()
    for k, v in cd.round[key]['gates'].items():
        g1[k] += v * r1
    add('S1 switch-in (gauging) to D4', q1, r1, key, 0, g1,
        SOURCE_ASSUME if model == 'FT_ref' else SOURCE_COUNT, note1)
    add('S2 transversal U = W V (logical T^dag)', n['d4_data'], 0, None, cd.U['time'], cd.U['gates'],
        SOURCE_COUNT, 'constant depth; gates from eq. (8)')
    add('S3 switch-out: Z^g meas + S^g correction', n['d4_data'], 0, None, cd.out['time'],
        cd.out['gates_worst'], SOURCE_COUNT, 'CZ correction counted worst case (expected = half)')
    r4 = 0 if model == 'NL_min' else rounds_for(model, params['kappa_out'], d)
    g4 = Counter()
    for k, v in cd.round['FSC']['gates'].items():
        g4[k] += v * r4
    add('S4 FSC error-correction rounds', fsc_q, r4, 'FSC', 0, g4,
        SOURCE_ASSUME if model == 'FT_ref' else SOURCE_COUNT,
        'O(d) rounds (Kobayashi step 4); kappa_out*d assumed; 0 in the noiseless minimum')
    add('S5 logical X frame: T^dag|+> -> T|+>', 0, 0, None, 0, Counter(), SOURCE_COUNT,
        'Pauli frame update, no physical cost')
    # NOTE (comparison B): the FSC -> standard rotated-patch conversion is UNRESOLVED.  It is deliberately
    # NOT added as a zero-cost stage; it is reported separately by uncounted_costs().
    return rows



def d4_conversion_illustrative(cd, model, params=DEFAULT_PARAMS, data_only=False):
    """ILLUSTRATIVE ONLY: conversion modelled as one lattice-surgery time step (tau) during which the
    FSC plus 2 tiles (target patch + ancilla strip) are active.  Not taken from literature."""
    d = cd.L
    r = rounds_for(model, params['tau_rounds'], d)
    q = cd.n['fsc_data'] + (0 if data_only else cd.n['fsc_anc']) + 2 * (d * d if data_only else 2 * d * d)
    return dict(protocol='Kobayashi_D4', comparison='B_illustrative', model=model, d=d,
                stage='S6 conversion (ILLUSTRATIVE: 1 tau lattice surgery)', qubits=q, rounds=r,
                time_lb=None, time_ub=None, gates={}, source=SOURCE_UNRES,
                note='illustrative assumption, not a validated protocol')




# ---------------------------------------------------------- consistency checks of a code
def code_consistency(cd):
    """Exact F2 checks of a (possibly non-commuting) Clifford stabilizer code on its Z-subspace C_Z:
      shift_in_CZ : every X-type generator preserves C_Z (its X part commutes with all Z checks)
      noncommuting_pairs : generator pairs that do NOT commute on C_Z
      bad_squares : generators with S^2 != 1 on C_Z
      k : number of logical qubits = dim C_Z - rank(shifts)   (generators have independent shifts)"""
    lat = cd.lat
    HZ, S = cd.HZ, cd.S
    res = {}
    res['shift_in_CZ'] = not ((HZ.astype(int) @ S.T.astype(int)) % 2).any()
    supp = [op.support() for _, op in cd.xgens]
    bad = []
    for i in range(len(cd.xgens)):
        for j in range(i + 1, len(cd.xgens)):
            if not (supp[i] & supp[j]):
                continue
            lin, const = commutator_affine(cd.xgens[i][1], cd.xgens[j][1])
            if const or not f2_in_rowspace(cd.HZr, cd.HZpiv, lat.vec(lin)):
                bad.append((cd.xgens[i][0], cd.xgens[j][0]))
    res['noncommuting_pairs'] = bad
    sq = []
    for lab, op in cd.xgens:
        s2 = op @ op
        if s2.s or s2.quad or s2.const or not f2_in_rowspace(cd.HZr, cd.HZpiv, lat.vec(s2.lin)):
            sq.append(lab)
    res['bad_squares'] = sq
    res['n_qubits'] = len(cd.cols)
    res['dim_CZ'] = len(cd.cols) - f2_rank(HZ[:, cd.cols])
    res['rank_shifts'] = f2_rank(S[:, cd.cols])
    res['n_xgens'] = len(cd.xgens)
    res['k'] = res['dim_CZ'] - res['rank_shifts']
    res['shifts_independent'] = res['rank_shifts'] == res['n_xgens']
    return res


def U_target(lat, lab):
    """Image of generator `lab` under U predicted by SM eq. (permutingstabilizers):
    S^r -> S^r S^g, S^g -> S^g, S^b -> S^g S^b, and (boundary L_rb) S^r S^b -> S^r S^b."""
    _, v, cols = lab
    if cols == 'rb':
        return lat.SX(v, R) @ lat.SX(v, B)
    op = PSOp()
    for ch in cols:
        c = CNAME.index(ch)
        for cc in ([c] if c == G else [c, G]):
            op = op @ lat.SX(v, cc)
    return op


# ---------------------------------------------------------- Knill-Laflamme distance lower bound
def _milp_binary(c_obj, A_eq_mod2, rhs_mod2, extra_ub=None, fixed_zero=()):
    """min c.x over binary x subject to A x = rhs (mod 2) (rows of A_eq_mod2), and optional
    linear constraints extra_ub = [(row, ub), ...] (row.x <= ub).  Returns (value, x) or (None, None)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    A = np.array(A_eq_mod2, dtype=float)
    m, n = A.shape
    nv = n + m
    c = np.concatenate([np.asarray(c_obj, float), np.zeros(m)])
    cons = [LinearConstraint(np.concatenate([A, -2 * np.eye(m)], axis=1), rhs_mod2, rhs_mod2)]
    for row, ub in (extra_ub or []):
        cons.append(LinearConstraint(np.concatenate([np.asarray(row, float), np.zeros(m)])[None, :], -np.inf, ub))
    ub_ = np.concatenate([np.ones(n), np.full(m, np.inf)])
    for i in fixed_zero:
        ub_[i] = 0
    res = milp(c, constraints=cons, integrality=np.ones(nv), bounds=Bounds(np.zeros(nv), ub_))
    if res.status != 0:
        return None, None
    return round(res.fun), np.round(res.x[:n]).astype(np.uint8)


def low_weight_stabilizer_shifts(cd, wmax, frozen=()):
    """Exact enumeration (MILP with no-good cuts) of all t in span(S) with 1 <= |t| <= wmax,
    where span(S) = C_Z  intersect  z_log^perp  (k = 1).  Frozen qubits are excluded
    (an X on a frozen qubit violates its single-qubit Z check, so t would leave C_Z)."""
    cols = [c for c in cd.cols if c not in set(frozen)]
    H = np.vstack([cd.HZ[:, cols], cd.z_log[cols][None, :]])
    H = H[np.any(H, axis=1)]
    rhs = np.zeros(len(H))
    n = len(cols)
    cuts = [(np.ones(n), wmax), (-np.ones(n), -1)]
    found = []
    while True:
        val, x = _milp_binary(np.ones(n), H, rhs, extra_ub=cuts)
        if val is None or val > wmax:
            break
        sup = np.nonzero(x)[0]
        t = np.zeros(cd.lat.n, dtype=np.uint8)
        t[[cols[i] for i in sup]] = 1
        found.append(t)
        row = -np.ones(n)
        row[sup] = 1
        cuts.append((row, len(sup) - 1))                  # exclude exactly this vector
    return found


def kl_distance_lower_bound(cd, wmax, frozen=()):
    r"""Rigorous lower-bound argument for the Knill-Laflamme distance of the (non-Pauli) code cd.

    For a Pauli E = X^t Z^u (up to phase):
      * t not in C_Z           -> E maps the code space to an orthogonal space: detectable;
      * t in the logical coset -> |t| >= X-shift distance (checked separately);
      * t in span(S)           -> X^t = S_c F_c with S_c a stabilizer and F_c = (-1)^{f_c} diagonal,
        so P E P = P F_c Z^u P.  Expanding (-1)^{f_c} in Z-type Paulis Z^v with v supported on
        Q_f = supp(f_c), a violation of KL requires some v + u to be a logical Z operator l.
        Hence |E| >= |t| + |u \ t| >= |t| + min_l |l \ (Q_f u t)|  =: LB_t.
    Returns (min over enumerated t of LB_t, details).  t = 0 is covered by d_Z."""
    lat = cd.lat
    ts = low_weight_stabilizer_shifts(cd, wmax, frozen)
    cols = [c for c in cd.cols if c not in set(frozen)]
    Srows = cd.S[:, cols]
    Srows = Srows[np.any(Srows, axis=1)]
    best, details = None, []
    for t in ts:
        coef = f2_solve(cd.S, t)
        op = PSOp()
        for i in np.nonzero(coef)[0]:
            op = op @ cd.xgens[i][1]
        assert op.s == frozenset(np.nonzero(t)[0])
        Qf = {q for pr in op.quad for q in pr} | set(op.lin) | set(np.nonzero(t)[0])
        cost = np.array([0.0 if c in Qf else 1.0 for c in cols])
        A = np.vstack([Srows, cd.t_log[cols][None, :]])
        rhs = np.concatenate([np.zeros(len(Srows)), [1.0]])
        val, _ = _milp_binary(cost, A, rhs)
        lb = int(t.sum()) + val
        details.append(dict(weight_t=int(t.sum()), Qf_size=len(Qf), LB=lb))
        best = lb if best is None else min(best, lb)
    return best, ts, details


# ---------------------------------------------------------- full resource tables (used by main + comparison)
def d4_resource_tables(D_LIST, dur=None, params=None, verbose=True):
    """Per-stage and summary tables of the Kobayashi protocol for every d in D_LIST, gate set
    ('primitive' | 'clifford_t'), counting variant (data_only), time model and comparison A/B.
    Returns (stage_df, sum_df, blocks)."""
    import pandas as pd, time
    dur = dur or DUR_UNIT
    params = params or DEFAULT_PARAMS
    records, summaries, blocks = [], [], {}
    for d in D_LIST:
        t0 = time.time()
        for gs in ['primitive', 'clifford_t']:
            cd = D4CostData(d, dur, decomposed=(gs == 'clifford_t'))
            for data_only in [False, True]:
                for model in ['FT_ref', 'NL_min']:
                    for comp in ['A', 'B']:
                        st = d4_stages(cd, model, params, comp, data_only=data_only)
                        for r in st:
                            r.update(gateset=gs, data_only=data_only)
                        records += st
                        s = summarize(st)
                        s.update(protocol='Kobayashi_D4', d=d, model=model, comparison=comp, gateset=gs,
                                 data_only=data_only)
                        summaries.append(s)
            blocks[f'{d}|{gs}'] = dict(d=d, gateset=gs, n=cd.n, n_total=cd.n_total, n_anc_g=cd.n_anc_g,
                                       round={k: {kk: (dict(vv) if isinstance(vv, Counter) else vv) for kk, vv in v.items()}
                                              for k, v in cd.round.items()},
                                       U_time=cd.U['time'], U_gates=dict(cd.U['gates']), out_time=cd.out['time'],
                                       out_gates=dict(cd.out['gates_worst']), xcorr_time=cd.x_corr['time'])
        if verbose:
            print(f'Kobayashi d={d}: {time.time() - t0:.1f}s')
    return jsonify_columns(pd.DataFrame(records)), jsonify_columns(pd.DataFrame(summaries)), blocks
