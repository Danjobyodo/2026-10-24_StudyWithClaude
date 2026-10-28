"""msd_noisy_lib: output error and cost of 15-to-1 magic-state distillation under circuit-level noise.

Error model (D. Litinski, "Magic State Distillation: Not as Costly as You Think", Quantum 3, 205 (2019),
and its published resource-estimation code github.com/litinski/magicstates, Apache-2.0; re-implemented
here, the error budgets are transcribed from that code):
  * the distillation circuit acts on 5 logical qubits (surface-code patches) and consists of Z-type
    pi/8 Pauli-product rotations; a faulty rotation is followed by a P_{pi/2}, P_{-pi/4} or P_{pi/4}
    error with probabilities p1, p2, p3 (faulty magic state: p/3 each; lattice-surgery measurement
    errors and errors in the ancilla region add to p1, p2);
  * patches suffer X / Z storage errors with probability 0.5 * p_L(d) per code cycle each;
  * p_L(p, d) = 0.1 (100 p)^((d+1)/2) per patch and code cycle (GoSC Eq. (10)), or a calibration of the
    same form obtained by stim simulations of the rotated surface code under the SAME circuit-level
    noise model as the D4 simulation (`calibrate_surface_code`);
  * the output error is the infidelity of the post-selected output state (exact density-matrix
    calculation, 128-bit arithmetic), the success probability is the post-selection probability.
Protocols:
  * `litinski2019_one_level(p, dx, dz, dm)` and `litinski2019_two_level(...)`: Litinski's optimised
    factories (transcribed; validated against the outputs stored in his notebook);
  * `gosc_15to1(p, d)` and `gosc_225to1(p, d1, d2)`: the A Game of Surface Codes Sec. 3 layouts
    (Fig. 17c: 11 tiles x 11 tau; Fig. 19: 176 tiles x 15 tau) evaluated with the same error model.
    The per-rotation error budget of these layouts is OUR adaptation of Litinski (2019) to the GoSC
    geometry (single distance d, two ancilla tiles per multi-patch measurement) - see `gosc_steps`.
"""
import numpy as np
import itertools
import mpmath
from mpmath import mp

mp.prec = 128


def plog_gosc(p, d):
    """GoSC Eq. (10) / Litinski (2019): logical error per patch per code cycle."""
    return mpmath.mpf('0.1') * (100 * mpmath.mpf(p)) ** (mpmath.mpf(d + 1) / 2)


# ------------------------------------------------------------------ 5-qubit density-matrix engine
class DM5:
    """Density matrix of n (=5) qubits.  backend='ld': numpy extended precision (np.clongdouble, ~19 digits,
    fast; used for the searches); backend='mp': numpy object array of mpmath.mpc (128 bits; used to check
    the transcription against Litinski's published outputs down to 1e-25).  Qubit 1 = most significant bit
    (Litinski's kron ordering)."""

    def __init__(self, n=5, singles=None, backend='ld'):
        self.n, self.backend = n, backend
        N = 2 ** n
        idx = np.arange(N)
        self.bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
        if singles is None:
            h = 1 / mpmath.sqrt(2) if backend == 'mp' else np.longdouble(1) / np.sqrt(np.longdouble(2))
            singles = [[h, h]] * n
        if backend == 'mp':
            psi = np.array([mpmath.mpc(1)] * N, dtype=object)
            for q, s in enumerate(singles):
                psi = psi * np.array([mpmath.mpc(s[b]) for b in self.bits[q]], dtype=object)
        else:
            psi = np.ones(N, dtype=np.clongdouble)
            for q, s in enumerate(singles):
                psi = psi * np.array([s[b] for b in self.bits[q]], dtype=np.clongdouble)
        self.rho = np.outer(psi, np.conj(psi))

    def _f(self, x):
        return mpmath.mpf(x) if self.backend == 'mp' else np.longdouble(float(x)) if not isinstance(x, np.longdouble) else x

    def _eig(self, axis):
        s = np.ones(2 ** self.n, dtype=int)
        for q, a in enumerate(axis):
            if a:
                s = s * (1 - 2 * self.bits[q])
        return s

    def rot(self, axis, p1=0, p2=0, p3=0):
        """Z-type Pauli-product rotation exp(i pi/8 P) followed by the faulty-rotation channel of
        Litinski's `apply_rot`: with prob p1 / p2 / p3 the angle is 5pi/8 / -pi/8 / 3pi/8."""
        s = self._eig(axis)
        diff = s[:, None] - s[None, :]            # in {-2, 0, 2}
        if self.backend == 'mp':
            p1, p2, p3 = mpmath.mpf(p1), mpmath.mpf(p2), mpmath.mpf(p3)
            w = [(1 - p1 - p2 - p3, mp.pi / 8), (p1, 5 * mp.pi / 8), (p2, -mp.pi / 8), (p3, 3 * mp.pi / 8)]
            m2 = sum(wk * mpmath.expj(2 * ph) for wk, ph in w)
            M = np.where(diff == 2, m2, np.where(diff == -2, mpmath.conj(m2), mpmath.mpf(1)))
        else:
            p1, p2, p3 = (np.longdouble(float(x)) for x in (p1, p2, p3))
            pi = np.longdouble('3.14159265358979323846264338327950288')
            w = [(1 - p1 - p2 - p3, pi / 8), (p1, 5 * pi / 8), (p2, -pi / 8), (p3, 3 * pi / 8)]
            m2 = sum(wk * (np.cos(2 * ph) + 1j * np.sin(2 * ph)) for wk, ph in w)
            M = np.where(diff == 2, m2, np.where(diff == -2, np.conj(m2), 1)).astype(np.clongdouble)
        self.rho = self.rho * M

    def z_err(self, q, p):
        if p == 0:
            return
        p = mpmath.mpf(p) if self.backend == 'mp' else np.longdouble(float(p))
        sg = (1 - 2 * self.bits[q])
        self.rho = self.rho * np.where(sg[:, None] * sg[None, :] == 1, 1, 1 - 2 * p)

    def x_err(self, q, p):
        if p == 0:
            return
        p = mpmath.mpf(p) if self.backend == 'mp' else np.longdouble(float(p))
        f = np.arange(2 ** self.n) ^ (1 << (self.n - 1 - q))
        self.rho = (1 - p) * self.rho + p * self.rho[np.ix_(f, f)]

    def store(self, px, pz):
        for q in range(self.n):
            self.x_err(q, px[q])
            self.z_err(q, pz[q])

    def postselect_x(self, measured):
        """Project the qubits in `measured` onto |+> (X = +1).  Returns (success probability, normalised
        2x2 density matrix of the remaining qubit)."""
        keep = [q for q in range(self.n) if q not in measured]
        assert len(keep) == 1
        k = keep[0]
        out = np.zeros((2, 2), dtype=self.rho.dtype)
        for a in range(2):
            for b in range(2):
                ia = self.bits[k] == a
                ib = self.bits[k] == b
                out[a, b] = self.rho[np.ix_(ia, ib)].sum() / 2 ** len(measured)
        ps = (out[0, 0] + out[1, 1]).real
        return ps, out / ps


def run_steps(steps, n=5, out_qubit=0, init=None, backend='ld'):
    """Evaluate a protocol given as a list of steps
       ('rot', axis (0/1 tuple), p1, p2, p3)   faulty Z-type pi/8 rotation
       ('store', px (list), pz (list))        X / Z Pauli channels
    on |+>^n; the qubits != out_qubit are measured in X and post-selected on +1.  Returns
    (success probability, output infidelity).  The ideal output is taken from the same circuit without
    errors; the infidelity is evaluated as the population <psi_perp|rho|psi_perp> of the orthogonal
    state (no cancellation in 1 - F)."""
    def go(noisy):
        dm = DM5(n, init, backend)
        for st in steps:
            if st[0] == 'rot':
                dm.rot(st[1], *(st[2:5] if noisy else (0, 0, 0)))
            elif st[0] == 'store' and noisy:
                dm.store(st[1], st[2])
        return dm.postselect_x([q for q in range(n) if q != out_qubit])
    ps0, r0 = go(False)
    # ideal output |psi> from the noiseless (pure) reduced state: rho0 = |psi><psi|
    if backend == 'mp':
        a = mpmath.sqrt(r0[0, 0].real)
        psi = [a, r0[1, 0] / a] if a != 0 else [mpmath.mpc(0), mpmath.mpc(1)]
        perp = [-mpmath.conj(psi[1]), mpmath.conj(psi[0])]
    else:
        a = np.sqrt(r0[0, 0].real)
        psi = [a, r0[1, 0] / a]
        perp = [-np.conj(psi[1]), np.conj(psi[0])]
    ps, r = go(True)
    eps = sum(np.conj(perp[i]) * r[i, j] * perp[j] for i in range(2) for j in range(2)).real
    return ps, eps


def infidelity(rho1, ideal1):
    """1 - <ideal|rho|ideal> for a single-qubit rho (2x2) and an ideal ket."""
    f = sum(np.conj(ideal1[a]) * rho1[a, b] * ideal1[b] for a in range(2) for b in range(2))
    return 1 - f.real


# ------------------------------------------------------------------ Litinski (2019), transcribed
_Z = 1


def _ax(*qs):
    return tuple(int(q in qs) for q in range(5))


def litinski2019_one_level_steps(p, dx, dz, dm, plog=plog_gosc):
    """Error budget of Litinski's (15-to-1)_{dx,dz,dm} (onelevel15to1.py), qubit 1 = output (index 0)."""
    p = mpmath.mpf(p)
    px, pz, pm = plog(p, dx), plog(p, dz), plog(p, dm)
    t = p / 3
    S = []
    single = lambda q: ('rot', _ax(q), t + 0.5 * (dm / dz) * pz * dm, t + 0.5 * dz * pm, t)
    multi = lambda qs, w: ('rot', _ax(*qs), t + 0.5 * pm * dm, t + 0.5 * pm * dm + 0.5 * w * dx / dm * pm, t)
    sx = lambda c0, c: ('store', [c0] + [c] * 4, [0] * 5)
    S += [single(1), single(2), single(3), multi((1, 2, 3), 3 * dz)]
    S += [('store', [0] + [0.5 * (dz / dx) * px * dm] * 3 + [0], [0] + [0.5 * (dx / dz) * pz * dm] * 3 + [0])]
    S += [multi((0, 1, 2), dx + 2 * dz), multi((0, 1, 3), dx + 3 * dz)]
    S += [('store', [0] * 5, [0.5 * ((dx + 2 * dz) + (dx + 3 * dz)) / dx * px * dm, 0, 0, 0, 0])]
    S += [('store', [0.5 * px * dm] + [0.5 * (dz / dx) * px * dm] * 3 + [0],
           [0.5 * px * dm] + [0.5 * (dx / dz) * pz * dm] * 3 + [0])]
    S += [multi((0, 2, 3), dx + 3 * dz), multi((0, 3, 4), dx + 4 * dz), single(4)]
    S += [('store', [0] * 5, [0.5 * ((dx + 3 * dz) + (dx + 4 * dz)) / dx * px * dm, 0, 0, 0, 0])]
    full = lambda a: ('store', [a * px] + [0.5 * (dz / dx) * px * dm] * 4, [a * px] + [0.5 * (dx / dz) * pz * dm] * 4)
    S += [full(0.5 * dm)]
    S += [multi((0, 1, 4), dx + 4 * dz), multi((0, 2, 4), dx + 4 * dz)]
    S += [('store', [0] * 5, [0.5 * ((dx + 4 * dz) + (dx + 4 * dz)) / dx * px * dm, 0, 0, 0, 0])]
    S += [full(0.5 * dm)]
    S += [multi((0, 1, 2, 3, 4), dx + 4 * dz), multi((2, 3, 4), 3 * dz)]
    S += [('store', [0] * 5, [0.5 * (dx + 4 * dz) / dx * px * dm, 0, 0, 0, 0])]
    S += [full(0.5 * (dm + 2 * dx))]
    S += [multi((1, 3, 4), 4 * dz), multi((1, 2, 4), 4 * dz)]
    S += [('store', [0] + [0.5 * (dz / dx) * px * dm] * 4, [0] + [0.5 * (dx / dz) * pz * dm] * 4)]
    return S


def litinski2019_one_level(p, dx, dz, dm, plog=plog_gosc, backend='ld'):
    ps, eps = run_steps(litinski2019_one_level_steps(p, dx, dz, dm, plog), out_qubit=0, backend=backend)
    qubits = 2 * ((dx + 4 * dz) * 3 * dx + 2 * dm)
    cycles = 6 * dm / ps
    return dict(protocol='Litinski2019 (15-to-1)_{%d,%d,%d}' % (dx, dz, dm), eps=float(eps), p_success=float(ps),
                qubits=int(qubits), cycles=float(cycles), volume=float(qubits * cycles), levels=1)


def litinski2019_two_level_steps(p, dx, dz, dm, dx2, dz2, dm2, nl1, plog=plog_gosc, backend='ld'):
    p = mpmath.mpf(p)
    px2, pz2, pm2 = plog(p, dx2), plog(p, dz2), plog(p, dm2)
    ps1, pl1 = run_steps(litinski2019_one_level_steps(p, dx, dz, dm, plog), out_qubit=0, backend=backend)
    ps1, pl1 = mpmath.mpf(float(ps1)) if backend != 'mp' else ps1, mpmath.mpf(float(pl1)) if backend != 'mp' else pl1
    pfail = 1 - ps1
    l1time = max(6 * dm / (mpmath.mpf(nl1) / 2) / (1 - pfail), dm2)
    lmove = 10 * dm2 + mpmath.mpf(nl1) / 4 * (dx + 4 * dz)
    R = lambda qs, w: ('rot', _ax(*qs), pl1 + 0.5 * lmove * pm2,
                       0.5 * lmove * pm2 + 0.5 * w * dx2 / dm2 * pm2, 0)
    st = lambda mask, c0=0: ('store', [c0] + [0.5 * (dz2 / dx2) * px2 * l1time if m else 0 for m in mask],
                             [c0] + [0.5 * (dx2 / dz2) * pz2 * l1time if m else 0 for m in mask])
    zs = lambda v: ('store', [0] * 5, [v, 0, 0, 0, 0])
    a = 0.5 * px2 * l1time
    S = [R((1,), dx2 + dz2 + dm2), R((2,), 3 * dz2 + dm2), st([1, 1, 0, 0]),
         R((3,), dx2 + 3 * dz2 + dm2), R((4,), dz2 + dm2), st([1, 1, 1, 1]),
         R((0, 1, 2), dx2 + 2 * dz2 + dm2), R((1, 2, 3), 4 * dz2 + dm2),
         zs(0.5 * (dx2 + 2 * dz2 + dm2) * dm2 / dx2 * px2), st([1, 1, 1, 1], a),
         R((0, 2, 3), dx2 + 3 * dz2 + dm2), R((0, 1, 3), dx2 + 4 * dz2 + dm2),
         zs(0.5 * ((dx2 + 3 * dz2 + dm2) + (dx2 + 4 * dz2 + dm2)) * dm2 / dx2 * px2), st([1, 1, 1, 1], a),
         R((0, 1, 4), dx2 + 4 * dz2 + dm2), R((0, 3, 4), dx2 + 4 * dz2 + dm2),
         zs(0.5 * 2 * (dx2 + 4 * dz2 + dm2) * dm2 / dx2 * px2), st([1, 1, 1, 1], a),
         R((0, 2, 4), dx2 + 4 * dz2 + dm2), R((0, 1, 2, 3, 4), dx2 + 4 * dz2 + dm2),
         zs(0.5 * 2 * (dx2 + 4 * dz2 + dm2) * dm2 / dx2 * px2), st([1, 1, 1, 1], a),
         R((1, 3, 4), dx2 + 4 * dz2 + dm2), R((2, 3, 4), 3 * dz2 + dm2),
         zs(0.5 * (dx2 + 4 * dz2 + dm2) * dm2 / dx2 * px2), st([1, 1, 1, 1], 0.5 * px2 * (dm2 + 2 * dx2)),
         R((1, 2, 4), 4 * dz2 + dm2), st([1, 1, 0, 1])]
    # note: Litinski's code passes the X/Z storage of qubit 1 only where stated (copied as above)
    return S, ps1, pl1, l1time


def litinski2019_two_level(p, dx, dz, dm, dx2, dz2, dm2, nl1, plog=plog_gosc, backend='ld'):
    S, ps1, pl1, l1time = litinski2019_two_level_steps(p, dx, dz, dm, dx2, dz2, dm2, nl1, plog, backend)
    ps2, eps = run_steps(S, out_qubit=0, backend=backend)
    qubits = 2 * int((dx2 + 4 * dz2) * 3 * dx2 + nl1 * ((dx + 4 * dz) * (3 * dx + dm2 / 2) + 2 * dm)
                     + 20 * dm2 * dm2 + 2 * dx2 * dm2)
    cycles = 7.5 * l1time / ps2
    return dict(protocol='Litinski2019 (15-to-1)x(15-to-1)_{%d,%d,%d;%d,%d,%d;%d}' % (dx, dz, dm, dx2, dz2, dm2, nl1),
                eps=float(eps), p_success=float(ps2), qubits=int(qubits), cycles=float(cycles),
                volume=float(qubits * cycles), levels=2, eps_level1=float(pl1))


# ------------------------------------------------------------------ surface-code p_L calibration (stim)
def noisy_surface_code(d, rounds, basis, p, p_idle=None):
    """Rotated surface-code memory (stim's generated circuit) with the SAME circuit-level noise model and
    time-step convention as the D4 simulation (noisy_d4_lib.NoiseModel / D4SwitchingCircuit rounds):
    X-type ancillas are prepared / measured with RX / MX (no H layers); stim's combined MR is split into a
    measurement step and a reset step, so that one round = reset, 4 CNOT layers, measurement = 6 time
    steps; 2-qubit gates are followed by DEPOLARIZE2(p), resets by a flip, measurements are preceded by a
    flip, and the idle qubits of every time step are depolarized with p_idle."""
    p_idle = p if p_idle is None else p_idle
    base = stim.Circuit.generated('surface_code:rotated_memory_' + basis, distance=d, rounds=rounds).flattened()
    allq, hq = set(), set()
    for ins in base:
        if ins.name in ('R', 'CX', 'M', 'MR', 'H'):
            allq |= {t.value for t in ins.targets_copy()}
        if ins.name == 'H':
            hq |= {t.value for t in ins.targets_copy()}
    out = stim.Circuit()

    def emit(pre, body, post, touched):
        for e in pre:
            out.append(*e)
        for b in body:
            out.append(*b) if isinstance(b, tuple) else out.append(b)
        for e in post:
            out.append(*e)
        idle = sorted(allq - touched)
        if idle and p_idle > 0:
            out.append('DEPOLARIZE1', idle, p_idle)
        out.append('TICK')

    def flush(layer):
        touched, pre, body, post, resets = set(), [], [], [], []
        for ins in layer:
            nm = ins.name
            ts = [t.value for t in ins.targets_copy() if t.is_qubit_target]
            if nm == 'H':
                continue
            if nm in ('R', 'RX'):
                for q in ts:
                    x = q in hq or nm == 'RX'
                    body.append(('RX' if x else 'R', [q]))
                    post.append(('Z_ERROR' if x else 'X_ERROR', [q], p))
            elif nm in ('MR', 'M', 'MX'):
                # one target per instruction keeps stim's measurement-record order
                for q in ts:
                    x = q in hq or nm == 'MX'
                    pre.append(('Z_ERROR' if x else 'X_ERROR', [q], p))
                    body.append(('MX' if x else 'M', [q]))
                    if nm == 'MR':
                        resets.append(q)
            elif nm == 'CX':
                body.append(('CX', ts))
                post.append(('DEPOLARIZE2', ts, p))
            else:
                body.append(ins)
                continue
            touched |= set(ts)
        if not body:
            return                                   # H-only layer: no time step
        emit(pre, body, post, touched)
        if resets:                                   # the reset half of MR: its own time step
            emit([], [('RX' if q in hq else 'R', [q]) for q in resets],
                 [('Z_ERROR' if q in hq else 'X_ERROR', [q], p) for q in resets], set(resets))
    layer = []
    for ins in base:
        if ins.name == 'TICK':
            flush(layer)
            layer = []
        elif ins.name in ('DETECTOR', 'OBSERVABLE_INCLUDE', 'QUBIT_COORDS', 'SHIFT_COORDS'):
            if layer:
                flush(layer)
                layer = []
            out.append(ins)
        else:
            layer.append(ins)
    if layer:
        flush(layer)
    return out


def surface_code_logical_rate(d, p, shots, rounds=None, seed=0):
    """Logical error per round (X + Z memories summed) of the rotated surface code (MWPM)."""
    rounds = rounds or 3 * d
    tot = 0.0
    out = {}
    for basis in ('x', 'z'):
        c = noisy_surface_code(d, rounds, basis, p)
        m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        det, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        pred = m.decode_batch(det)
        PL = float(np.mean(pred[:, 0] != obs[:, 0]))
        eps = (1 - (1 - 2 * PL) ** (1 / rounds)) / 2 if PL < 0.5 else 0.5
        out[basis] = (PL, eps)
        tot += eps
    return tot, out


def fit_plog(rows):
    """Fit p_L = A (p/p_th)^((d+1)/2) to rows (d, p, p_L) by least squares in log p_L."""
    ds = np.array([r[0] for r in rows], float)
    ps = np.array([r[1] for r in rows], float)
    pl = np.array([r[2] for r in rows], float)
    k = (ds + 1) / 2
    # log pl = log A + k log p - k log pth  -> linear in (log A, log pth)
    A_ = np.stack([np.ones_like(k), -k], axis=1)
    y = np.log(pl) - k * np.log(ps)
    (lA, lpth), *_ = np.linalg.lstsq(A_, y, rcond=None)
    return float(np.exp(lA)), float(np.exp(lpth))


def make_plog(A, pth):
    return lambda p, d: mpmath.mpf(A) * (mpmath.mpf(p) / mpmath.mpf(pth)) ** (mpmath.mpf(d + 1) / 2)


# ------------------------------------------------------------------ GoSC Sec. 3 layouts with the same model
# the 11 rotations of GoSC Fig. 15 (= msd_lib.fig15_rotations(): columns 5..15 of the 15-to-1 matrix,
# qubit indices 0..4 = qubits 1..5; validated in MSD_15to1_validation M1)
GOSC_ROTATIONS = [(2, 3, 4), (1, 3, 4), (1, 2, 4), (1, 2, 3), (0, 3, 4), (0, 2, 4), (0, 2, 3),
                  (0, 1, 4), (0, 1, 3), (0, 1, 2), (0, 1, 2, 3, 4)]


def gosc_15to1_steps(p, d, plog=plog_gosc, p_in=None, anc_tiles=2, in_channel='pauli'):
    """GoSC Fig. 15 / 17c: qubits 1-4 start in |m> (faulty, error p_in), qubit 5 in |+>; the 11 Z-type
    pi/8 rotations of Fig. 15 (msd_lib.fig15_rotations) are auto-corrected pi/8 rotations of 1 tau = d
    code cycles each; qubits 1-4 are measured in X, qubit 5 is the output.
    Error budget per rotation (our adaptation of Litinski 2019 to the single-distance GoSC layout):
      p1 (P_{pi/2})  = e1 + 0.5 p_L d                        (time-like error of the P x Z_m measurement)
      p2 (P_{-pi/4}) = e2 + 0.5 p_L d + 0.5 anc_tiles p_L d   (space-like errors in the ancilla tiles)
      p3 (P_{pi/4})  = e3
    with (e1, e2, e3) = (p_in/3,)*3 for a raw state with a random Pauli error (in_channel='pauli', GoSC
    Sec. 4 assumption) or (p_in, 0, 0) for a distilled input (in_channel='z', Litinski two-level code);
    every data tile suffers X and Z storage errors 0.5 p_L d per tau."""
    p = mpmath.mpf(p)
    p_in = p if p_in is None else mpmath.mpf(p_in)
    pl = plog(p, d)
    e = (p_in / 3,) * 3 if in_channel == 'pauli' else (p_in, 0, 0)
    S = [('rot', _ax(q), *e) for q in range(4)]               # the four |m> states (0 tau)
    st = ('store', [0.5 * pl * d] * 5, [0.5 * pl * d] * 5)
    for qs in GOSC_ROTATIONS:
        S.append(('rot', _ax(*qs), e[0] + 0.5 * pl * d, e[1] + 0.5 * pl * d + 0.5 * anc_tiles * pl * d, e[2]))
        S.append(st)
    return S


def gosc_15to1(p, d, plog=plog_gosc, backend='ld', **kw):
    """Output error and cost of the GoSC 11-tile 15-to-1 block (Fig. 17c): 11 tiles x 11 tau."""
    ps, eps = run_steps(gosc_15to1_steps(p, d, plog, **kw), out_qubit=4, backend=backend)
    qubits = 11 * 2 * d * d                       # 11 tiles, d^2 data + d^2 measurement qubits each
    cycles = 11 * d / ps
    return dict(protocol='GoSC 15-to-1 (11 tiles x 11 tau), d=%d' % d, eps=float(eps), p_success=float(ps),
                qubits=int(qubits), cycles=float(cycles), volume=float(qubits * cycles), levels=1, d=d)


def gosc_225to1(p, d1, d2, plog=plog_gosc, move_tiles=5, backend='ld', level1=None):
    """GoSC Fig. 19 (176 tiles, 15 tau): 11 level-1 15-to-1 blocks (distance d1) feed a level-2 15-to-1
    whose 15 rotations (incl. the 4 single-qubit ones) consume distilled states.  Level-1 output error
    enters as a Z-type (P_{pi/2}) error (Litinski two-level code); moving a level-1 state through the
    blue ancilla of Fig. 19 adds 0.5 * move_tiles * p_L(d2) * d2 (our assumption, move_tiles = 5)."""
    ps1, e1 = level1 if level1 is not None else run_steps(gosc_15to1_steps(p, d1, plog), out_qubit=4, backend=backend)
    ps1, e1 = mpmath.mpf(float(ps1)), mpmath.mpf(float(e1))
    pl2 = plog(mpmath.mpf(p), d2)
    mv = 0.5 * move_tiles * pl2 * d2
    S = []
    st = ('store', [0.5 * pl2 * d2] * 5, [0.5 * pl2 * d2] * 5)
    for q in range(4):
        S.append(('rot', _ax(q), e1 + mv + 0.5 * pl2 * d2, 0.5 * pl2 * d2 + pl2 * d2, 0))
        S.append(st)
    for qs in GOSC_ROTATIONS:
        S.append(('rot', _ax(*qs), e1 + mv + 0.5 * pl2 * d2, 0.5 * pl2 * d2 + pl2 * d2, 0))
        S.append(st)
    ps2, eps = run_steps(S, out_qubit=4, backend=backend)
    qubits = 2 * (121 * d1 * d1 + 55 * d2 * d2)   # 11 level-1 blocks of 11 tiles + 55 level-2 tiles
    cycles = 15 * d2 / (ps2 * ps1)                # a level-2 step is skipped when no level-1 state is ready
    return dict(protocol='GoSC 225-to-1 (176 tiles x 15 tau), d1=%d, d2=%d' % (d1, d2), eps=float(eps),
                p_success=float(ps2), qubits=int(qubits), cycles=float(cycles), volume=float(qubits * cycles),
                levels=2, d=d2, d1=d1, eps_level1=float(e1))
