"""Litinski 'A Game of Surface Codes' Sec. 3: 15-to-1 distillation.
Logical-level circuits (statevector) + tile-level layout model + physical (patch-level) model."""
import itertools
import numpy as np

# ----------------------------------------------------------------- matrices (transcribed)
# Eq. (1) of Litinski: rows 1-4 = X stabilizers, row 5 = logical X, columns = 15 qubits
M15 = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1]], dtype=np.uint8)
# Eq. (2): triorthogonal matrix G ; Eq. (4): its row-echelon form G~
G_TRI = np.array([[1] * 16,
                  [0] * 8 + [1] * 8,
                  [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                  [0, 1] * 8], dtype=np.uint8)
G_TILDE = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1]], dtype=np.uint8)


def triorthogonal(G):
    G = G.astype(int)
    a = all(G[i].sum() % 8 == 0 for i in range(len(G)))
    b = all((G[i] * G[j]).sum() % 4 == 0 for i, j in itertools.combinations(range(len(G)), 2))
    c = all((G[i] * G[j] * G[k]).sum() % 2 == 0 for i, j, k in itertools.combinations(range(len(G)), 3))
    return a, b, c


# ----------------------------------------------------------------- statevector helpers
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}
KET0 = np.array([1, 0], dtype=complex)
KETP = np.array([1, 1], dtype=complex) / np.sqrt(2)
T_GATE = np.diag([1, np.exp(1j * np.pi / 4)])
KET_M = T_GATE @ KETP                 # |m> = |0> + e^{i pi/4}|1>  (normalised)


def kron_all(ops):
    out = np.array([[1]], dtype=complex)
    for o in ops:
        out = np.kron(out, o)
    return out


def pauli_on(n, paulistr_by_qubit):
    """Dense Pauli operator; paulistr_by_qubit: dict qubit -> 'X'|'Y'|'Z'.  Qubit 0 = most significant."""
    return kron_all([PAULI[paulistr_by_qubit.get(i, 'I')] for i in range(n)])


def apply_1q(psi, n, q, U):
    psi = psi.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)


def z_rotation_diag(n, qubits, phi):
    """Diagonal of exp(-i phi Z_{q1} Z_{q2} ...)."""
    idx = np.arange(2 ** n)
    par = np.zeros(2 ** n, dtype=int)
    for q in qubits:
        par ^= (idx >> (n - 1 - q)) & 1
    return np.exp(-1j * phi * (1 - 2 * par))


def measure_pauli(psi, P, rng=None, forced=None):
    """Projective measurement of Hermitian Pauli matrix P.  Returns (outcome +-1, post-state, prob)."""
    Ppsi = P @ psi
    p_plus = np.real(np.vdot(psi, psi + Ppsi)) / 2
    if forced is None:
        out = 1 if rng.random() < p_plus else -1
    else:
        out = forced
    post = (psi + out * Ppsi) / 2
    prob = p_plus if out == 1 else 1 - p_plus
    if prob > 1e-12:
        post = post / np.sqrt(prob)
    return out, post, prob


def equal_up_to_phase(a, b, tol=1e-9):
    ov = np.vdot(a, b)
    return abs(abs(ov) - np.linalg.norm(a) * np.linalg.norm(b)) < tol


# ----------------------------------------------------------------- 15-to-1, 5-qubit circuit (Fig. 15)
def fig15_rotations():
    """The 11 Z-type pi/8 rotations of Fig. 15 = columns 5..15 of M15 (rows = qubits 1..5).
    Columns 1..4 (weight-1 on qubits 4,3,2,1) are absorbed into the |m> initial states."""
    rots = []
    for j in range(4, 15):
        rots.append(tuple(int(i) for i in np.nonzero(M15[:, j])[0]))
    return rots


def run_15to1_5qubit(inputs=None, phi=np.pi / 8, rng=None):
    """Simulate Fig. 15.  inputs: list of 4 single-qubit states for qubits 1-4 (default ideal |m>).
    Returns dict outcome-tuple -> (probability, output state of qubit 5)."""
    n = 5
    if inputs is None:
        inputs = [KET_M] * 4
    psi = kron_all([s.reshape(2, 1) for s in inputs] + [KETP.reshape(2, 1)]).reshape(-1)
    for qs in fig15_rotations():
        psi = z_rotation_diag(n, qs, phi) * psi
    # measure qubits 1-4 in X basis: enumerate all outcomes
    res = {}
    for outs in itertools.product((1, -1), repeat=4):
        st = psi.copy()
        prob = 1.0
        for q, o in enumerate(outs):
            _, st, p = measure_pauli(st, pauli_on(n, {q: 'X'}), forced=o)
            prob *= p
            if p < 1e-14:
                break
        if prob < 1e-14:
            res[outs] = (0.0, None)
            continue
        # reduced state of qubit 5 (qubits 1-4 are in X eigenstates -> product)
        st = st.reshape(16, 2)
        # project onto the X-eigenstates of qubits 1..4
        xs = [np.array([1, o], dtype=complex) / np.sqrt(2) for o in outs]
        bra = kron_all([x.reshape(1, 2) for x in xs]).reshape(-1)
        out5 = bra.conj() @ st
        out5 = out5 / np.linalg.norm(out5)
        res[outs] = (prob, out5)
    return res


# ----------------------------------------------------------------- 15-qubit code check (Fig. 14)
def rm15_logical_check(gate='T'):
    """Prepare |+_L> of the 15-qubit code (X stabilisers = rows 1-4 of M15, logical X = row 5,
    Z stabilisers = F2-orthogonal complement), apply transversal T (or T^dag) and return the
    logical expectation values <X_L>, <Y_L-like> to identify the logical gate."""
    n = 15
    from_rows = M15.astype(int)
    # Z stabilisers: nullspace of M15 over F2
    Hz = f2_nullspace_simple(M15)
    dim = 2 ** n
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    idx = np.arange(dim)
    # |+_L> = prod (1+X_s)/2 (1+X_L)/2 |0..0>  (|0..0> is +1 eigenstate of all Z stabilisers)
    for row in from_rows:
        mask = int(''.join(str(b) for b in row), 2)
        psi = (psi + psi[idx ^ mask]) / 2
    psi /= np.linalg.norm(psi)
    # transversal gate
    w = np.array([bin(i).count('1') for i in idx])
    ph = np.exp(1j * np.pi / 4 * w) if gate == 'T' else np.exp(-1j * np.pi / 4 * w)
    psi = ph * psi
    # logical operators: X_L = X(row5), Z_L = Z on a vector z with z.row5 = 1, z.rows1-4 = 0
    xl = int(''.join(str(b) for b in from_rows[4]), 2)
    zvec = find_logical_z(M15)
    zl = int(''.join(str(b) for b in zvec), 2)
    par = np.array([bin(i & zl).count('1') % 2 for i in idx])
    Zl_psi = (1 - 2 * par) * psi
    Xl_psi = psi[idx ^ xl]
    # Y_L = i X_L Z_L
    Yl_psi = 1j * Zl_psi[idx ^ xl]
    ex = np.real(np.vdot(psi, Xl_psi))
    ey = np.real(np.vdot(psi, Yl_psi))
    ez = np.real(np.vdot(psi, Zl_psi))
    # stabiliser expectation values (X stabilisers should be lost only up to the Clifford
    # correction; Z stabilisers preserved)
    zst = []
    for hz in Hz:
        m = int(''.join(str(b) for b in hz), 2)
        par = np.array([bin(i & m).count('1') % 2 for i in idx])
        zst.append(np.real(np.vdot(psi, (1 - 2 * par) * psi)))
    return dict(X_L=ex, Y_L=ey, Z_L=ez, min_Zstab=min(zst), n_Zstab=len(Hz))


def f2_nullspace_simple(M):
    M = M.copy() % 2
    rows, cols = M.shape
    A = M.copy()
    piv = []
    r = 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if A[i, c]), None)
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        piv.append(c)
        r += 1
    free = [c for c in range(cols) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for i, p in enumerate(piv):
            if A[i, f]:
                v[p] = 1
        basis.append(v)
    return np.array(basis)


def find_logical_z(M):
    """z with M[0:4] z = 0, M[4] z = 1 (brute force over weight)."""
    for w in range(1, 16):
        for sup in itertools.combinations(range(15), w):
            z = np.zeros(15, dtype=np.uint8)
            z[list(sup)] = 1
            s = (M.astype(int) @ z) % 2
            if not s[:4].any() and s[4] == 1:
                return z
    raise RuntimeError


# ----------------------------------------------------------------- auto-corrected pi/8 (Fig. 17b)
def auto_corrected_pi8(P_data, psi_data, rng=None):
    """Fig. 17b on k data qubits + |m> + |0>.  P_data: dict qubit->Pauli on the data.
    Enumerates all measurement branches; for each, returns the Pauli correction needed and
    whether the post-correction data state equals exp(-i pi/8 P)|psi>.
    Measurement sequence (Litinski Fig. 17b):
      1. measure P (x) Z_m  and  Z_m (x) Y_0   (commuting, simultaneous)
      2. if P(x)Z_m = +1: measure |0>-qubit in Z ; else in X     (selective pi/4, Fig. 17a)
      3. measure |m> in X
      4. Pauli correction P^{c} with c determined by the outcomes (P_{pi/2} = P up to phase)."""
    k = int(np.log2(len(psi_data)))
    n = k + 2
    m, a = k, k + 1
    psi = np.kron(np.kron(psi_data, KET_M), KET0)
    P1 = pauli_on(n, {**P_data, m: 'Z'})
    P2 = pauli_on(n, {m: 'Z', a: 'Y'})
    Pd = pauli_on(k, P_data)
    target = np.cos(np.pi / 8) * psi_data - 1j * np.sin(np.pi / 8) * (Pd @ psi_data)
    branches = []
    for o1 in (1, -1):
        for o2 in (1, -1):
            s1, st, p1 = measure_pauli(psi, P1, forced=o1)
            if p1 < 1e-14:
                continue
            s2, st, p2 = measure_pauli(st, P2, forced=o2)
            if p2 < 1e-14:
                continue
            basis = 'Z' if o1 == 1 else 'X'
            for o3 in (1, -1):
                s3, st3, p3 = measure_pauli(st, pauli_on(n, {a: basis}), forced=o3)
                if p3 < 1e-14:
                    continue
                for o4 in (1, -1):
                    s4, st4, p4 = measure_pauli(st3, pauli_on(n, {m: 'X'}), forced=o4)
                    if p4 < 1e-14:
                        continue
                    # extract data state: ancillas are in known product eigenstates
                    anc_m = np.array([1, o4], dtype=complex) / np.sqrt(2)
                    anc_a = (np.array([1, 0], dtype=complex) if o3 == 1 else np.array([0, 1], dtype=complex)) \
                        if basis == 'Z' else np.array([1, o3], dtype=complex) / np.sqrt(2)
                    bra = np.kron(anc_m, anc_a)
                    dat = st4.reshape(2 ** k, 4) @ bra.conj()
                    dat /= np.linalg.norm(dat)
                    ok0 = equal_up_to_phase(dat, target)
                    ok1 = equal_up_to_phase(Pd @ dat, target)
                    branches.append(dict(PZ=o1, ZY=o2, anc_basis=basis, anc=o3, Xm=o4,
                                         prob=p1 * p2 * p3 * p4,
                                         correction='I' if ok0 else ('P' if ok1 else 'FAIL')))
    return branches


# ----------------------------------------------------------------- tile layout (Fig. 17c)
# Grid of 3 rows x 5 columns (row 0 = top).  Each entry: label of the patch occupying the tile.
FIG17C_STEP1 = [
    [None, None, 'q1', 'q3', None],
    [None, None, None, None, 'q5'],
    [None, None, 'q2', 'q4', None]]
FIG17C_ROTATION = [
    ['zero', 'm', 'q1', 'q3', None],
    ['zero', 'm', 'anc', 'anc', 'q5'],
    [None, None, 'q2', 'q4', None]]
FIG17C_FINAL = [
    ['zero', 'm', 'q1', 'q3', None],
    ['zero', 'm', None, None, 'q5'],
    [None, None, 'q2', 'q4', None]]
# NOTE: in Step 1 the 2x2 block on the left is shaded as part of the block (reserved) and the
# |m> states for the rotations are prepared there; Litinski counts 11 tiles for the block.


def tile_schedule(n_rot=11):
    """List of (label, tau, layout) for the Fig. 17c protocol.  tau = time steps (1 = d cycles)."""
    sched = [('step1_init', 0, FIG17C_STEP1)]
    for r in range(n_rot):
        sched.append((f'rotation_{r + 1}', 1, FIG17C_ROTATION))
        sched.append((f'readout_{r + 1}', 0, FIG17C_FINAL))
    sched.append(('final_X_measurement', 0, FIG17C_FINAL))
    return sched


def tiles_used(sched):
    used = set()
    for _, _, lay in sched:
        for i, row in enumerate(lay):
            for j, lab in enumerate(row):
                if lab is not None:
                    used.add((i, j))
    return used


import numpy as np


def rotated_checks(rows, cols, origin=(0, 0)):
    """Stabilizers of a rotated planar code on data qubits (r,c), 0<=r<rows, 0<=c<cols (shifted by
    origin).  Check type from global parity (r+c)%2 of the plaquette corner -> consistent tiling.
    Weight-2 Z checks on top/bottom, weight-2 X checks on left/right, so that the logical Z
    string runs vertically, along the left/right boundaries (the boundaries used for Z(x)Z surgery)."""
    r0, c0 = origin
    checks = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            qs = [(r0 + rr, c0 + cc) for rr in (r - 1, r) for cc in (c - 1, c)
                  if 0 <= rr < rows and 0 <= cc < cols]
            typ = 'X' if (r0 + r + c0 + c) % 2 == 0 else 'Z'
            if len(qs) == 4:
                checks.append((typ, tuple(qs)))
            elif len(qs) == 2:
                horizontal_edge = (r == 0 or r == rows)
                if (horizontal_edge and typ == 'Z') or ((c == 0 or c == cols) and typ == 'X'):
                    checks.append((typ, tuple(qs)))
    return checks


def patch_gadgets(checks, qindex, anc_start):
    gads = []
    a = anc_start
    for typ, qs in checks:
        idx = [qindex[q] for q in qs]
        if typ == 'X':
            gads.append(pauli_check_gadget(a, xq=idx))
        else:
            gads.append(z_check_gadget(a, idx))
        a += 1
    return gads


def patch_qubit_count(d):
    ch = rotated_checks(d, d)
    return d * d, len(ch)


def lattice_surgery_zz_check(d, gap=None, seed=0):
    """Two d x d rotated patches separated by `gap` columns (default d, i.e. one tile).
    Prepare both in |+_L>, perform one noiseless merge round on the full rectangle, and verify
    with stim that  Zbar1 Zbar2  becomes deterministic and equals the product of the merge-round
    outcomes of an F2 combination of merged Z checks.  Returns a dict of results."""
    import stim
    if gap is None:
        gap = d
    W = 2 * d + gap
    data = [(r, c) for r in range(d) for c in range(W)]
    qi = {q: i for i, q in enumerate(data)}
    p1 = rotated_checks(d, d, (0, 0))
    p2 = rotated_checks(d, d, (0, d + gap))
    merged = rotated_checks(d, W, (0, 0))
    nq = len(data)

    def vec(qs):
        v = np.zeros(nq, dtype=np.uint8)
        for q in qs:
            v[qi[q]] ^= 1
        return v

    def logical_z(checks, cols):
        # Z string along a column that commutes with the patch's X checks
        X = [vec(qs) for t, qs in checks if t == 'X']
        for c in cols:
            z = vec([(r, c) for r in range(d)])
            if all((x.astype(int) @ z) % 2 == 0 for x in X):
                return z
        raise RuntimeError('no column logical')

    z1 = logical_z(p1, range(0, d))
    z2 = logical_z(p2, range(d + gap, W))
    O = z1 ^ z2
    sim = stim.TableauSimulator()
    sim.set_num_qubits(nq)
    rng = np.random.default_rng(seed)
    # |+> on all data of both patches and on the gap (Litinski Fig. 41: ancilla region in |+>)
    for i in range(nq):
        sim.h(i)

    def meas(typ, qs):
        ps = stim.PauliString(nq)
        for q in qs:
            ps[qi[q]] = typ
        return sim.measure_observable(ps)

    pre = {}
    for t, qs in p1 + p2:
        pre[(t, qs)] = meas(t, qs)
    ps = stim.PauliString(nq)
    for i in np.nonzero(O)[0]:
        ps[int(i)] = 'Z'
    before = sim.peek_observable_expectation(ps)
    out = {}
    for t, qs in merged:
        out[(t, qs)] = meas(t, qs)
    after = sim.peek_observable_expectation(ps)
    Zm = [(k, vec(k[1])) for k in merged if k[0] == 'Z']
    A = np.array([v for _, v in Zm])
    coef = f2_solve(A, O)
    parity = 1
    for c, (k, _) in zip(coef, Zm):
        if c:
            parity *= -1 if out[k] else 1
    n_checks_merged = len(merged)
    return dict(d=d, gap=gap, data_qubits=nq, merged_checks=n_checks_merged,
                total_qubits=nq + n_checks_merged,
                exp_before=before, exp_after=after, product_of_outcomes=parity,
                consistent=(after == parity), O_in_span=coef is not None)

# ----------------------------------------------------------------- Litinski (15-to-1)
# ---------------------------------------------------------- Fig. 17c multi-patch lattice surgery (physical level)
def _tile_rect(r, c, d, h=1, w=1):
    return (r * d, (r + h) * d, c * d, (c + w) * d)


def fig17_layout(d):
    """Patches of the rotation step of [Lit] Fig. 17c on a grid of d x d-data-qubit tiles
    (3 tile rows x 5 tile columns, row 0 = top).  Each entry: (rect, rule) with rect =
    (r0, r1, c0, c1) data-qubit ranges and rule(side, r, c) -> 'X'|'Z' = type of the weight-2
    boundary checks on that side.  For every data patch the boundary facing the ancilla strip is of
    'X-check' type, so that its logical Z string runs along it ([Lit] 'Z edge')."""
    tb = lambda side, r, c: 'X' if side in ('top', 'bottom') else 'Z'       # q1..q4 (merge via top/bottom)
    lr = lambda side, r, c: 'X' if side in ('left', 'right') else 'Z'       # q5 (merge via its left edge)
    mr = _tile_rect(0, 1, d, h=2)

    def m_rule(side, r, c, r0=mr[0]):
        # |m> occupies two tiles; its corners are moved ([Lit] Fig. 2b) so that only the lower half of
        # its right edge (facing the ancilla strip) is a Z edge: segments Z(top+upper right), X(lower
        # right), Z(bottom), X(left).
        if side == 'right':
            return 'X' if r >= r0 + d else 'Z'
        return 'X' if side == 'left' else 'Z'
    return {'q1': (_tile_rect(0, 2, d), tb), 'q3': (_tile_rect(0, 3, d), tb),
            'q2': (_tile_rect(2, 2, d), tb), 'q4': (_tile_rect(2, 3, d), tb),
            'q5': (_tile_rect(1, 4, d), lr), 'm': (mr, m_rule)}


def _owner_map(patches, anc_rect=None):
    own = {}
    for name, ((r0, r1, c0, c1), _) in patches.items():
        for r in range(r0, r1):
            for c in range(c0, c1):
                own[(r, c)] = name
    if anc_rect:
        r0, r1, c0, c1 = anc_rect
        for r in range(r0, r1):
            for c in range(c0, c1):
                own[(r, c)] = 'anc'
    return own


def _side_of(rect, r, c):
    r0, r1, c0, c1 = rect
    if r == r0: return 'top'
    if r == r1: return 'bottom'
    if c == c0: return 'left'
    if c == c1: return 'right'
    return None


def patch_checks(name, rect, rule):
    """Checks (type, qubits) of one standalone rotated patch with the given boundary rule."""
    r0, r1, c0, c1 = rect
    out = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            typ = 'X' if (r + c) % 2 == 0 else 'Z'
            qs = [(rr, cc) for rr in (r - 1, r) for cc in (c - 1, c) if r0 <= rr < r1 and c0 <= cc < c1]
            if len(qs) == 4:
                out.append((typ, tuple(qs)))
            elif len(qs) == 2 and typ == rule(_side_of(rect, r, c), r, c):
                out.append((typ, tuple(qs)))
    return out


def surgery_new_checks(d, merged, anc_free='Z', w3=('Z',), seam_end=('Z',), corner1=('Z',)):
    """Checks newly measured when the ancilla strip (tiles (1,2),(1,3), all qubits prepared in |+>)
    is merged with the patches in `merged` ([Lit] App. A, Fig. 41): every plaquette that touches an
    ancilla qubit, restricted to the merged region.  Truncated plaquettes: weight 3 kept if its type is
    in w3; weight 2 on the free ancilla boundary kept if its type == anc_free; weight 2 straddling
    ancilla/patch kept if its type is in seam_end; an ancilla corner qubit left without a weight-2 check
    (both neighbouring edges free) is measured in Z if the plaquette type is in corner1."""
    pats = fig17_layout(d)
    anc_rect = _tile_rect(1, 2, d, w=2)
    own = _owner_map(pats, anc_rect)
    region = {q for q, o in own.items() if o == 'anc' or o in set(merged)}
    r0, r1, c0, c1 = anc_rect
    N = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            typ = 'X' if (r + c) % 2 == 0 else 'Z'
            qs = [(rr, cc) for rr in (r - 1, r) for cc in (c - 1, c) if (rr, cc) in region]
            if not any(own[q] == 'anc' for q in qs):
                continue
            if len(qs) == 4:
                N.append((typ, tuple(qs)))
            elif len(qs) == 1 and typ in corner1:
                N.append((typ, tuple(qs)))           # isolated ancilla corner qubit: measured in Z
            elif len(qs) == 3 and typ in w3:
                N.append((typ, tuple(qs)))
            elif len(qs) == 2:
                if all(own[q] == 'anc' for q in qs):
                    af = anc_free(_side_of(anc_rect, r, c), r, c) if callable(anc_free) else anc_free
                    if typ == af:
                        N.append((typ, tuple(qs)))
                elif typ in seam_end:
                    N.append((typ, tuple(qs)))
    return N, pats, anc_rect, own


def _vec(qs, idx, n):
    v = np.zeros(n, dtype=np.uint8)
    for q in qs:
        v[idx[q]] ^= 1
    return v


def _mats(checks, idx, n):
    X = np.array([_vec(qs, idx, n) for t, qs in checks if t == 'X'], dtype=np.uint8).reshape(-1, n)
    Z = np.array([_vec(qs, idx, n) for t, qs in checks if t == 'Z'], dtype=np.uint8).reshape(-1, n)
    return X, Z


def _rank(M):
    return f2_rank(M) if len(M) else 0


def _span_dim_intersection(A, B):
    """dim(span A  intersect  span B)."""
    if len(A) == 0 or len(B) == 0:
        return 0
    return _rank(A) + _rank(B) - _rank(np.vstack([A, B]))


def commutant(G, Nopp):
    """Basis of {g in span(G) : g . v = 0 for all rows v of Nopp}, plus the indices of the rows of G
    that individually commute (kept as measured checks)."""
    if len(G) == 0:
        return np.zeros((0, Nopp.shape[1]), np.uint8), []
    Bm = (G.astype(int) @ Nopp.T.astype(int)) % 2 if len(Nopp) else np.zeros((len(G), 0), int)
    keep = [i for i in range(len(G)) if not Bm[i].any()]
    if Bm.shape[1] == 0:
        return G.copy(), keep
    C = f2_nullspace(Bm.T.astype(np.uint8), len(G))
    return (C.astype(int) @ G.astype(int) % 2).astype(np.uint8), keep


def css_logical_bases(SX, SZ, n):
    """Independent logical representatives (modulo stabilizers) of a CSS code."""
    def indep(base, cands):
        out = []
        cur = base if len(base) else np.zeros((0, n), np.uint8)
        r = _rank(cur)
        for v in cands:
            nxt = np.vstack([cur, v[None, :]])
            rn = _rank(nxt)
            if rn > r:
                out.append(v); cur = nxt; r = rn
        return out
    Lz = indep(SZ, list(f2_nullspace(SX, n)) if len(SX) else list(np.eye(n, dtype=np.uint8)))
    Lx = indep(SX, list(f2_nullspace(SZ, n)) if len(SZ) else list(np.eye(n, dtype=np.uint8)))
    return Lx, Lz


def verify_multipatch(d, merged, distance=False, **opts):
    """Exact (F2) verification that measuring the new checks N implements the logical measurement of
    O = prod_{p in merged} Zbar_p on the Fig. 17c layout and nothing else.
      V1 N mutually commute;  V2 every standalone patch has k = 1 (distance d if distance=True);
      V3 O in span(N_Z) + span(G0_Z) and exactly one independent Z-logical is measured;
      V4 no X-logical is measured;
      V5 every logical commuting with O (all Zbar_p, Xbar_p for p not merged, Xbar_p Xbar_q for merged
         pairs) has a representative (mod G0) commuting with N;
      V6 (distance=True) merged code G1 = <N, commutant of N in G0> has k = 5 and distance >= d.
    Also returns the physically measured generating set during the merged rounds."""
    N, pats, anc_rect, own = surgery_new_checks(d, merged, **opts)
    qs = sorted(own); idx = {q: i for i, q in enumerate(qs)}; n = len(qs)
    G0 = []
    for name, (rect, rule) in pats.items():
        G0 += patch_checks(name, rect, rule)
    G0X, G0Z = _mats(G0, idx, n)
    anc_q = [idx[q] for q in qs if own[q] == 'anc']
    G0Xa = np.vstack([G0X, np.eye(n, dtype=np.uint8)[anc_q]])          # ancilla |+> : single-qubit X
    NX, NZ = _mats(N, idx, n)
    res = {'n_data': n, 'n_new_checks': len(N)}
    res['V1_new_checks_commute'] = not ((NX.astype(int) @ NZ.T.astype(int)) % 2).any()
    # standalone logicals
    Zb, Xb, ok2, dists = {}, {}, True, {}
    for name, (rect, rule) in pats.items():
        ch = patch_checks(name, rect, rule)
        sub = sorted({q for _, c in ch for q in c} | {(r, c) for r in range(rect[0], rect[1]) for c in range(rect[2], rect[3])})
        sidx = {q: i for i, q in enumerate(sub)}
        SX, SZ = _mats(ch, sidx, len(sub))
        Lx, Lz = css_logical_bases(SX, SZ, len(sub))
        ok2 &= len(Lz) == 1 and len(Lx) == 1
        emb = lambda v: _vec([sub[i] for i in np.nonzero(v)[0]], idx, n)
        Zb[name], Xb[name] = emb(Lz[0]), emb(Lx[0])
        if distance:
            dz = min_weight_odd(SX, Lx[0], list(range(len(sub))))[0]
            dx = min_weight_odd(SZ, Lz[0], list(range(len(sub))))[0]
            dists[name] = (dx, dz)
    res['V2_patches_k1'] = ok2
    if distance:
        res['V2_patch_distances'] = dists
        res['V2_patch_distance_ok'] = all(min(v) >= d for v in dists.values())
    names = list(pats)
    O = np.zeros(n, np.uint8)
    for p in merged:
        O ^= Zb[p]
    AZ = np.vstack([G0Z, NZ])
    res['V3_O_determined'] = _rank(np.vstack([AZ, O[None, :]])) == _rank(AZ)
    ZL = np.array([Zb[p] for p in names])
    res['V3_num_Z_logicals_measured'] = _span_dim_intersection(AZ, np.vstack([G0Z, ZL])) - _rank(G0Z)
    AX = np.vstack([G0Xa, NX])
    XL = np.array([Xb[p] for p in names])
    res['V4_num_X_logicals_measured'] = _span_dim_intersection(AX, np.vstack([G0Xa, XL])) - _rank(G0Xa)

    def has_rep(L, G, Nopp):
        if len(Nopp) == 0:
            return True
        lhs = (Nopp.astype(int) @ L.astype(int)) % 2
        if not lhs.any():
            return True
        Bm = (Nopp.astype(int) @ G.T.astype(int)) % 2
        return f2_solve(Bm.T.astype(np.uint8), lhs.astype(np.uint8)) is not None
    ok5 = all(has_rep(Zb[p], G0Z, NX) for p in names)
    ml = sorted(merged)
    xs = [Xb[p] for p in names if p not in merged] + [Xb[a] ^ Xb[b] for a, b in zip(ml, ml[1:])]
    ok5 &= all(has_rep(v, G0Xa, NZ) for v in xs)
    res['V5_commuting_logicals_preserved'] = ok5
    # merged-phase measured generating set: N + old checks that commute with N + products for the rest
    CZ_, keepZ = commutant(G0Z, NX)
    CX_, keepX = commutant(G0X, NZ)
    G1X = np.vstack([NX, CX_]); G1Z = np.vstack([NZ, CZ_])
    extraZ = _rank(np.vstack([NZ, G0Z[keepZ]] if keepZ else [NZ, np.zeros((0, n), np.uint8)]))
    res['n_old_checks_kept'] = len(keepZ) + len(keepX)
    res['n_old_checks_combined'] = (_rank(G1Z) - _rank(np.vstack([NZ] + ([G0Z[keepZ]] if keepZ else []))) +
                                    _rank(G1X) - _rank(np.vstack([NX] + ([G0X[keepX]] if keepX else []))))
    k1 = n - _rank(G1X) - _rank(G1Z)
    res['merged_k'] = k1
    res['valid'] = (res['V1_new_checks_commute'] and ok2 and res['V3_O_determined'] and
                    res['V3_num_Z_logicals_measured'] == 1 and res['V4_num_X_logicals_measured'] == 0 and ok5
                    and k1 == len(names) - 1)
    if distance and res['valid']:
        Lx, Lz = css_logical_bases(G1X, G1Z, n)
        dz = min(min_weight_odd(G1X, lx, list(range(n)))[0] for lx in Lx)
        dx = min(min_weight_odd(G1Z, lz, list(range(n)))[0] for lz in Lz)
        res['V6_merged_distance'] = (dx, dz)
        res['valid'] &= min(dx, dz) >= d
    res['_data'] = dict(N=N, G0=G0, own=own, idx=idx, pats=pats, anc_rect=anc_rect, G1X=G1X, G1Z=G1Z,
                        keepX=keepX, keepZ=keepZ, CX=CX_, CZ=CZ_, G0X=G0X, G0Z=G0Z, NX=NX, NZ=NZ, O=O)
    return res


def rotation_measured_checks(d, merged, res=None):
    """Explicit list of checks measured in every round of the merged phase of one rotation:
    the new checks N, the old patch checks that commute with N, and (if needed) products of old checks
    completing the commutant (local, weight <= 8).  Returns (checks, res)."""
    res = res or verify_multipatch(d, merged)
    D = res['_data']
    qs = sorted(D['own']); n = len(qs)
    out = list(D['N'])
    G0 = D['G0']
    G0Xc = [c for c in G0 if c[0] == 'X']; G0Zc = [c for c in G0 if c[0] == 'Z']
    out += [G0Xc[i] for i in D['keepX']] + [G0Zc[i] for i in D['keepZ']]
    for typ, C, base in [('X', D['CX'], np.vstack([D['NX']] + ([D['G0X'][D['keepX']]] if D['keepX'] else []))),
                         ('Z', D['CZ'], np.vstack([D['NZ']] + ([D['G0Z'][D['keepZ']]] if D['keepZ'] else [])))]:
        cur = base
        for v in sorted(C, key=lambda v: int(v.sum())):
            nxt = np.vstack([cur, v[None, :]])
            if _rank(nxt) > _rank(cur):
                supp = tuple(qs[i] for i in np.nonzero(v)[0])
                assert len(supp) <= 8, 'non-local combined generator'
                out.append((typ, supp)); cur = nxt
    return out, res


def stim_multipatch_check(d, merged, seed=0):
    """Noiseless stim run: patches in |+_L> (data |+>, one round of their checks), ancilla strip in |+>,
    then one round of the merged-phase checks.  Checks that O = prod Zbar is random before and
    deterministic after, with value equal to the product of the merged-round outcomes of an F2
    combination of measured Z checks (times recorded pre-merge Z-check outcomes)."""
    import stim
    checks, res = rotation_measured_checks(d, merged)
    D = res['_data']
    qs = sorted(D['own']); idx = {q: i for i, q in enumerate(qs)}; n = len(qs)
    sim = stim.TableauSimulator(seed=seed)
    sim.set_num_qubits(n)
    for i in range(n):
        sim.h(i)
    def meas(t, cq):
        ps = stim.PauliString(n)
        for q in cq: ps[idx[q]] = t
        return sim.measure_observable(ps)
    pre = [(t, cq, meas(t, cq)) for t, cq in D['G0']]
    Ops = stim.PauliString(n)
    for i in np.nonzero(D['O'])[0]: Ops[int(i)] = 'Z'
    before = sim.peek_observable_expectation(Ops)
    post = [(t, cq, meas(t, cq)) for t, cq in checks]
    after = sim.peek_observable_expectation(Ops)
    rows = [(_vec(cq, idx, n), o) for t, cq, o in post + pre if t == 'Z']
    A = np.array([r[0] for r in rows]); coef = f2_solve(A, D['O'])
    pred = 1
    for c_, (_, o) in zip(coef, rows):
        if c_: pred *= -1 if o else 1
    return dict(before=before, after=after, predicted=pred, consistent=(after == pred) and before == 0)


def rotation_measured_checks_fast(d, merged):
    """Same measured set as rotation_measured_checks, computed without the global verification:
    old checks that commute with all new checks are kept; among the (few, seam-adjacent) old checks
    that anticommute, the commuting combinations are found by a small nullspace computation.
    (Equality with the fully verified construction is checked in the validation notebook.)"""
    N, pats, anc_rect, own = surgery_new_checks(d, merged)
    G0 = []
    for name, (rect, rule) in pats.items():
        G0 += patch_checks(name, rect, rule)
    out = list(N)
    for typ, opp in [('X', 'Z'), ('Z', 'X')]:
        olds = [c for c in G0 if c[0] == typ]
        news = [set(c[1]) for c in N if c[0] == opp]
        anti = []
        for c in olds:
            s = set(c[1])
            if any(len(s & nn) % 2 for nn in news):
                anti.append(c)
            else:
                out.append(c)
        if anti:
            qset = sorted({q for c in anti for q in c[1]} | {q for nn in news for q in nn})
            li = {q: i for i, q in enumerate(qset)}
            Aold = np.array([_vec(c[1], li, len(qset)) for c in anti])
            Nn = np.array([_vec(nn, li, len(qset)) for nn in news if nn & set(li)])
            C, _ = commutant(Aold, Nn)
            cur = np.zeros((0, len(qset)), np.uint8)
            for v in sorted(C, key=lambda v: int(v.sum())):
                nxt = np.vstack([cur, v[None, :]])
                if _rank(nxt) > _rank(cur):
                    out.append((typ, tuple(qset[i] for i in np.nonzero(v)[0]))); cur = nxt
    return out, pats, own


class MSDCostData:
    """d-dependent counts of the Litinski 15-to-1 block (Fig. 17c).
    * tile model: Litinski's accounting (computed in msd_stages).
    * physical model: explicit construction of every rotation's merged phase (surgery_new_checks /
      rotation_measured_checks_fast): data qubits of q1..q5, |m> (2 tiles, moved corners), the ancilla
      strip (2 tiles) and the |0> patch (2 tiles), one measurement qubit per measured check, CNOTs per
      round and round time from the shared scheduler.  NOT constructed: the twist-based Z_m (x) Y_0
      surgery between |m> and |0> (its extra seam checks are not counted)."""

    def __init__(self, d, dur=None, explicit=True):
        dur = dur or DUR_UNIT
        self.d = d
        self.dur = dur
        ch = rotated_checks(d, d)
        qi = {(r, c): r * d + c for r in range(d) for c in range(d)}
        g = patch_gadgets(ch, qi, d * d)
        ub = round_schedule_classes(g, dur)
        lb = round_lower_bound(g, dur)
        self.round = dict(time_lb=lb['time_lb'], time_ub=ub['time_ub'])       # one isolated patch
        self.sq_patch = d * d + len(ch)                                       # 2d^2 - 1
        ch2 = rotated_checks(d, 2 * d)
        self.tall_patch = 2 * d * d + len(ch2)                                # 4d^2 - 1
        self.cnot_sq = sum(len(qs) for _, qs in ch)
        self.cnot_tall = sum(len(qs) for _, qs in ch2)
        ch3 = rotated_checks(d, 3 * d)
        self.gap_tile = 3 * d * d + len(ch3) - 2 * self.sq_patch               # 2-patch Z(x)Z merge increment
        if not explicit:
            return
        pats = fig17_layout(d)
        idle_checks = []
        for name, (rect, rule) in pats.items():
            idle_checks += patch_checks(name, rect, rule)
        zero_rect = (0, 2 * d, -d, 0)                                         # |0> patch: tiles (0,0),(1,0)
        zero_checks = patch_checks('zero', zero_rect, lambda side, r, c: 'X' if side in ('left', 'right') else 'Z')
        def cost(checks, n_data):
            coords = sorted({q for _, qq in checks for q in qq})
            qi_ = {q: i for i, q in enumerate(coords)}
            gads = []
            for j, (t, qq) in enumerate(checks):
                idx_ = [qi_[q] for q in qq]
                gads.append(pauli_check_gadget(10 ** 6 + j, xq=idx_) if t == 'X' else z_check_gadget(10 ** 6 + j, idx_))
            return dict(data=n_data, checks=len(checks), qubits=n_data + len(checks),
                        cnot=sum(len(qq) for _, qq in checks),
                        time_lb=round_lower_bound(gads, dur)['time_lb'],
                        time_ub=round_schedule_classes(gads, dur)['time_ub'])
        n_patch_data = 7 * d * d                     # q1..q5 + |m> (2 tiles)
        self.idle = cost(idle_checks + zero_checks, n_patch_data + 2 * d * d)
        self.rot = []
        for r_ in fig15_rotations():
            P = tuple(sorted({'q%d' % (i + 1) for i in r_})) + ('m',)
            chk, _, _ = rotation_measured_checks_fast(d, P)
            self.rot.append(dict(P=P, **cost(chk + zero_checks, n_patch_data + 2 * d * d + 2 * d * d)))


def msd_stages(md, model, params=DEFAULT_PARAMS, comparison='A', level='physical', data_only=False):
    """level='tile': Litinski's accounting (tiles x d^2 data qubits, or x 2d^2 incl. measurement
    qubits; time = rounds x isolated-patch round time).  level='physical': explicit per-rotation
    construction (see MSDCostData)."""
    d = md.d
    tau = rounds_for(model, params['tau_rounds'], d)
    rows = []
    if level == 'tile':
        per_tile = d * d if data_only else 2 * d * d
        rot = [dict(qubits=11 * per_tile, cnot=None, time_lb=md.round['time_lb'], time_ub=md.round['time_ub'])] * 11
        idle = dict(qubits=9 * per_tile, time_lb=md.round['time_lb'], time_ub=md.round['time_ub'])
        src_rot = SOURCE_LIT
    else:
        key = 'data' if data_only else 'qubits'
        rot = [dict(qubits=r[key], cnot=r['cnot'], time_lb=r['time_lb'], time_ub=r['time_ub']) for r in md.rot]
        idle = dict(qubits=md.idle[key], time_lb=md.idle['time_lb'], time_ub=md.idle['time_ub'])
        src_rot = SOURCE_COUNT
    meas_layer = md.dur['MEAS_X']

    def add(stage, qubits, rounds, rt, extra_time, gates, source, note=''):
        rows.append(dict(protocol='Litinski_15to1', comparison=comparison, model=model, d=d, stage=stage,
                         qubits=qubits, rounds=rounds,
                         time_lb=rounds * rt['time_lb'] + extra_time,
                         time_ub=rounds * rt['time_ub'] + extra_time,
                         gates=dict(gates), source=source, note=note))

    inj = 0
    if comparison == 'B':
        inj = params['t_inj_FT'] if model == 'FT_ref' else params['t_inj_NL']
        add('L0 inject 4 |m> (qubits 1-4, parallel)', idle['qubits'], inj, idle, 0, Counter({'T': 4}), SOURCE_LIT,
            "Litinski App. A / Li: 3+3 cycles, d-independent; physical T gate per |m>")
    add('L1 init qubit 5 |+> (step 1)', idle['qubits'], 0, idle, 0, Counter(), SOURCE_LIT, '0 tau (Litinski)')
    for k in range(11):
        if comparison == 'B':
            add(f'L2.{k + 1}a inject |m> for rotation {k + 1}', idle['qubits'], inj, idle, 0, Counter({'T': 1}),
                SOURCE_LIT, 'serial in the 11-tile layout (Litinski "Caveat")')
        g = Counter({'CNOT': rot[k]['cnot'] * tau}) if rot[k]['cnot'] else Counter()
        add(f'L2.{k + 1} auto-corrected pi/8 rotation {k + 1}', rot[k]['qubits'], tau, rot[k], 0, g, src_rot,
            '1 tau = d cycles: P(x)Z_m surgery (explicit) and Z_m(x)Y_0 (twist not constructed)')
        add(f'L2.{k + 1}b readout |0>,|m> (single-qubit meas.)', idle['qubits'], 0, idle, meas_layer, Counter(),
            SOURCE_LIT, '0 tau; classical processing time not counted')
    add('L3 X measurement of qubits 1-4', idle['qubits'], 0, idle, meas_layer, Counter(), SOURCE_LIT, '0 tau')
    add('L4 logical X frame: T^dag|+> -> T|+>', 0, 0, idle, 0, Counter(), SOURCE_COUNT, 'Pauli frame')
    return rows





# ----------------------------------------------------------------- full resource tables (used by main + comparison)
def msd_resource_tables(D_LIST, dur=None, params=None, verbose=True):
    """Per-stage and summary tables of the Litinski 15-to-1 block for every d in D_LIST, level
    ('tile' | 'physical'), counting variant, time model and comparison A/B.  Litinski uses only
    Clifford + T, so the 'primitive' and 'clifford_t' gate-set rows are identical.
    Returns (stage_df, sum_df, per_d_df)."""
    import pandas as pd, time
    dur = dur or DUR_UNIT
    params = params or DEFAULT_PARAMS
    records, summaries, per_d = [], [], []
    for d in D_LIST:
        t0 = time.time()
        md_ = MSDCostData(d, dur)
        for level in ['tile', 'physical']:
            for data_only in [False, True]:
                for model in ['FT_ref', 'NL_min']:
                    for comp in ['A', 'B']:
                        st = msd_stages(md_, model, params, comp, level=level, data_only=data_only)
                        for gs in ['primitive', 'clifford_t']:
                            records += [dict(r, level=level, gateset=gs, data_only=data_only) for r in st]
                            s = summarize(st)
                            s.update(protocol='Litinski_15to1', d=d, model=model, comparison=comp, level=level,
                                     gateset=gs, data_only=data_only)
                            summaries.append(s)
        per_d.append(dict(d=d, rotation_qubits_min=min(r['qubits'] for r in md_.rot),
                          rotation_qubits_max=max(r['qubits'] for r in md_.rot), idle_qubits=md_.idle['qubits'],
                          rotation_round_lb=max(r['time_lb'] for r in md_.rot),
                          rotation_round_ub=max(r['time_ub'] for r in md_.rot),
                          CNOT_per_round_max=max(r['cnot'] for r in md_.rot), tile_model_rotation_qubits=22 * d * d,
                          isolated_patch_round_lb=md_.round['time_lb'], isolated_patch_round_ub=md_.round['time_ub']))
        if verbose:
            print(f'Litinski d={d}: {time.time() - t0:.1f}s')
    return jsonify_columns(pd.DataFrame(records)), jsonify_columns(pd.DataFrame(summaries)), pd.DataFrame(per_d)
