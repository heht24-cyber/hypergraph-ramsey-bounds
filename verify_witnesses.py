"""
Standalone, Zero-Dependency (stdlib + NumPy + optional Numba), ZERO-PRUNING
Exhaustive Independent Verifier for Hypergraph Ramsey Lower-Bound Witnesses.

Design principles (Absolute Zero-Slop Certification):
1. Imports NOTHING from the search codebase (groups.py, verifier.py, probsat.py, orbit_cegar.py).
2. Reads ONLY the self-contained JSON witness (n, t, ks, generators, orbit_colors).
3. Independently validates:
   - Every generator is a valid bijection on {0, ..., n-1}.
   - Orbit decomposition of all C(n, t) t-subsets under the given generators.
   - Strict G-invariance: C(g(e)) == C(e) for EVERY generator g and EVERY t-subset e.
   - EXHAUSTIVE ZERO-PRUNING CLIQUE CHECK: literally visits EVERY SINGLE ONE of all
     C(n, k) k-subsets (no branch-and-bound, no symmetry reduction, no sampling!),
     verifying checked_count == math.comb(n, k) and monochromatic_count == 0!
     * For R(4, 4, 4; 3) at n=101: visits all 4,082,925 quadruples (also includes a
       pure-NumPy zero-Numba check!).
     * For R(6, 6; 5) at n=86: visits all 470,155,077 6-subsets!
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numba as nb
import numpy as np


def _binom_table(n: int, k: int) -> np.ndarray:
    B = np.zeros((n + 1, k + 1), dtype=np.int64)
    for i in range(n + 1):
        B[i, 0] = 1
        for j in range(1, min(i, k) + 1):
            B[i, j] = B[i - 1, j - 1] + B[i - 1, j]
    return B


@nb.njit
def _dsu_find(parent: np.ndarray, x: int) -> int:
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:
        nxt = parent[x]
        parent[x] = r
        x = nxt
    return r


@nb.njit
def _dsu_union(parent: np.ndarray, a: int, b: int) -> None:
    ra = _dsu_find(parent, a)
    rb = _dsu_find(parent, b)
    if ra != rb:
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb


@nb.njit
def _build_orbits_and_check_invariance_t3(n: int, gens: np.ndarray, B: np.ndarray, orbit_colors: np.ndarray):
    M = B[n, 3]
    parent = np.arange(M, dtype=np.int32)
    ng = gens.shape[0]
    for gi in range(ng):
        g = gens[gi]
        for c in range(2, n):
            bc = B[c, 3]
            gc = g[c]
            for b in range(1, c):
                bbc = B[b, 2] + bc
                gb = g[b]
                for a in range(b):
                    idx = B[a, 1] + bbc
                    x, y, z = g[a], gb, gc
                    if x > y:
                        x, y = y, x
                    if y > z:
                        y, z = z, y
                    if x > y:
                        x, y = y, x
                    _dsu_union(parent, idx, B[x, 1] + B[y, 2] + B[z, 3])

    root_to_id = np.full(M, -1, dtype=np.int32)
    colors = np.empty(M, dtype=np.int8)
    cnt = 0
    for i in range(M):
        r = _dsu_find(parent, i)
        oid = root_to_id[r]
        if oid == -1:
            oid = cnt
            root_to_id[r] = oid
            cnt += 1
        colors[i] = orbit_colors[oid]

    # Independently check strict G-invariance on EVERY triple and EVERY generator!
    inv_errors = 0
    for gi in range(ng):
        g = gens[gi]
        for c in range(2, n):
            bc = B[c, 3]
            gc = g[c]
            for b in range(1, c):
                bbc = B[b, 2] + bc
                gb = g[b]
                for a in range(b):
                    idx = B[a, 1] + bbc
                    x, y, z = g[a], gb, gc
                    if x > y:
                        x, y = y, x
                    if y > z:
                        y, z = z, y
                    if x > y:
                        x, y = y, x
                    if colors[idx] != colors[B[x, 1] + B[y, 2] + B[z, 3]]:
                        inv_errors += 1
    return colors, cnt, inv_errors


@nb.njit
def _build_orbits_and_check_invariance_t5(n: int, gens: np.ndarray, B: np.ndarray, orbit_colors: np.ndarray):
    M = B[n, 5]
    parent = np.arange(M, dtype=np.int32)
    ng = gens.shape[0]
    for gi in range(ng):
        g = gens[gi]
        for e in range(4, n):
            be = B[e, 5]
            ge = g[e]
            for d in range(3, e):
                bde = B[d, 4] + be
                gd = g[d]
                for c in range(2, d):
                    bcde = B[c, 3] + bde
                    gc = g[c]
                    for b in range(1, c):
                        bbcde = B[b, 2] + bcde
                        gb = g[b]
                        for a in range(b):
                            idx = B[a, 1] + bbcde
                            v0, v1, v2, v3, v4 = g[a], gb, gc, gd, ge
                            if v0 > v1:
                                v0, v1 = v1, v0
                            if v2 > v3:
                                v2, v3 = v3, v2
                            if v0 > v2:
                                v0, v2 = v2, v0
                            if v1 > v4:
                                v1, v4 = v4, v1
                            if v1 > v2:
                                v1, v2 = v2, v1
                            if v3 > v4:
                                v3, v4 = v4, v3
                            if v2 > v3:
                                v2, v3 = v3, v2
                            if v1 > v2:
                                v1, v2 = v2, v1
                            if v0 > v1:
                                v0, v1 = v1, v0
                            _dsu_union(parent, idx, B[v0, 1] + B[v1, 2] + B[v2, 3] + B[v3, 4] + B[v4, 5])

    root_to_id = np.full(M, -1, dtype=np.int32)
    colors = np.empty(M, dtype=np.int8)
    cnt = 0
    for i in range(M):
        r = _dsu_find(parent, i)
        oid = root_to_id[r]
        if oid == -1:
            oid = cnt
            root_to_id[r] = oid
            cnt += 1
        colors[i] = orbit_colors[oid]

    inv_errors = 0
    for gi in range(ng):
        g = gens[gi]
        for e in range(4, n):
            be = B[e, 5]
            ge = g[e]
            for d in range(3, e):
                bde = B[d, 4] + be
                gd = g[d]
                for c in range(2, d):
                    bcde = B[c, 3] + bde
                    gc = g[c]
                    for b in range(1, c):
                        bbcde = B[b, 2] + bcde
                        gb = g[b]
                        for a in range(b):
                            idx = B[a, 1] + bbcde
                            v0, v1, v2, v3, v4 = g[a], gb, gc, gd, ge
                            if v0 > v1:
                                v0, v1 = v1, v0
                            if v2 > v3:
                                v2, v3 = v3, v2
                            if v0 > v2:
                                v0, v2 = v2, v0
                            if v1 > v4:
                                v1, v4 = v4, v1
                            if v1 > v2:
                                v1, v2 = v2, v1
                            if v3 > v4:
                                v3, v4 = v4, v3
                            if v2 > v3:
                                v2, v3 = v3, v2
                            if v1 > v2:
                                v1, v2 = v2, v1
                            if v0 > v1:
                                v0, v1 = v1, v0
                            if colors[idx] != colors[B[v0, 1] + B[v1, 2] + B[v2, 3] + B[v3, 4] + B[v4, 5]]:
                                inv_errors += 1
    return colors, cnt, inv_errors


# ---------------------------------------------------------------------------
# ZERO-PRUNING LITERAL EXHAUSTIVE NESTED-LOOP VERIFIERS
# Visits EVERY SINGLE k-subset in {0, ..., n-1} and counts both checked & mono!
# ---------------------------------------------------------------------------


@nb.njit(parallel=True)
def _zero_pruning_check_k4_t3(colors: np.ndarray, B: np.ndarray, n: int) -> tuple[int, int]:
    checked = np.zeros(n, dtype=np.int64)
    mono = np.zeros(n, dtype=np.int64)
    for a in nb.prange(n):
        ba1 = B[a, 1]
        c_loc = 0
        m_loc = 0
        for b in range(a + 1, n):
            bb1 = B[b, 1]
            bb2 = B[b, 2]
            for c in range(b + 1, n):
                bc2 = B[c, 2]
                bc3 = B[c, 3]
                col0 = colors[ba1 + bb2 + bc3]
                for d in range(c + 1, n):
                    bd3 = B[d, 3]
                    c_loc += 1
                    if (
                        colors[ba1 + bb2 + bd3] == col0
                        and colors[ba1 + bc2 + bd3] == col0
                        and colors[bb1 + bc2 + bd3] == col0
                    ):
                        m_loc += 1
        checked[a] = c_loc
        mono[a] = m_loc
    return int(np.sum(checked)), int(np.sum(mono))


def _pure_numpy_check_k4_t3(colors: np.ndarray, B: np.ndarray, n: int) -> tuple[int, int]:
    """100% PURE NUMPY (zero Numba!) exhaustive check of ALL C(n, 4) quadruples."""
    # Unpack into dense (n, n, n) lookup for crystal-clear pure-NumPy indexing
    C3 = np.full((n, n, n), -1, dtype=np.int8)
    for a in range(n):
        ba1 = int(B[a, 1])
        for b in range(a + 1, n):
            base = ba1 + int(B[b, 2])
            cs = np.arange(b + 1, n)
            C3[a, b, cs] = colors[base + B[cs, 3]]
    checked = 0
    mono = 0
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                ds = np.arange(c + 1, n)
                checked += len(ds)
                c0 = C3[a, b, c]
                hit = (C3[a, b, ds] == c0) & (C3[a, c, ds] == c0) & (C3[b, c, ds] == c0)
                mono += int(np.count_nonzero(hit))
    return checked, mono


@nb.njit
def _check_pair_k6_t5(a: int, b: int, colors: np.ndarray, B: np.ndarray, n: int) -> tuple[int, int]:
    ba1 = B[a, 1]
    bb1 = B[b, 1]
    bb2 = B[b, 2]
    c_loc = 0
    m_loc = 0
    for c in range(b + 1, n - 3):
        bc2 = B[c, 2]
        bc3 = B[c, 3]
        for d in range(c + 1, n - 2):
            bd3 = B[d, 3]
            bd4 = B[d, 4]
            for e in range(d + 1, n - 1):
                be4 = B[e, 4]
                be5 = B[e, 5]
                col0 = colors[ba1 + bb2 + bc3 + bd4 + be5]
                b_abcd = ba1 + bb2 + bc3 + bd4
                b_abce = ba1 + bb2 + bc3 + be4
                b_abde = ba1 + bb2 + bd3 + be4
                b_acde = ba1 + bc2 + bd3 + be4
                b_bcde = bb1 + bc2 + bd3 + be4
                for f in range(e + 1, n):
                    bf5 = B[f, 5]
                    c_loc += 1
                    if (
                        colors[b_abcd + bf5] == col0
                        and colors[b_abce + bf5] == col0
                        and colors[b_abde + bf5] == col0
                        and colors[b_acde + bf5] == col0
                        and colors[b_bcde + bf5] == col0
                    ):
                        m_loc += 1
    return c_loc, m_loc


@nb.njit(parallel=True)
def _zero_pruning_check_k6_t5(colors: np.ndarray, B: np.ndarray, n: int) -> tuple[int, int]:
    """
    Literal 6-nested-loop ZERO-PRUNING exhaustive check of ALL C(n, 6) 6-subsets!
    For n=86, visits EVERY SINGLE ONE of all 470,155,077 6-subsets!
    """
    npairs = (n * (n - 1)) // 2
    pa = np.empty(npairs, dtype=np.int32)
    pb = np.empty(npairs, dtype=np.int32)
    idx = 0
    for a in range(n - 5):
        for b in range(a + 1, n - 4):
            pa[idx] = a
            pb[idx] = b
            idx += 1
    total_pairs = idx

    checked = np.zeros(total_pairs, dtype=np.int64)
    mono = np.zeros(total_pairs, dtype=np.int64)

    for pid in nb.prange(total_pairs):
        c_loc, m_loc = _check_pair_k6_t5(pa[pid], pb[pid], colors, B, n)
        checked[pid] = c_loc
        mono[pid] = m_loc
    return int(np.sum(checked)), int(np.sum(mono))


def verify_witness_file(path: Path) -> None:
    t0 = time.perf_counter()
    data = json.loads(path.read_text(encoding="utf-8"))
    n = int(data["n"])
    t = int(data["t"])
    ks = [int(k) for k in data["ks"]]
    gens = np.array(data["generators"], dtype=np.int32)
    ocol = np.array(data["orbit_colors"], dtype=np.int8)
    num_colors = len(ks)

    print(f"\n=== INDEPENDENT ZERO-PRUNING AUDIT: {path.name} ===")
    print(f"  Target: {data['ramsey_target']} | Claimed LB: >= {n + 1} (n={n}, t={t}, ks={ks})")

    # 1. Validate every generator is a bijection on 0..n-1
    for gi, g in enumerate(gens):
        assert g.shape == (n,), f"Generator {gi} wrong shape"
        assert np.array_equal(np.sort(g), np.arange(n)), f"Generator {gi} is NOT a bijection!"

    # 2. Validate orbit colors range
    assert int(ocol.min()) >= 0 and int(ocol.max()) < num_colors, "Invalid color value!"

    # 3. Reconstruct coloring and verify strict G-invariance
    B = _binom_table(n, t)
    if t == 3:
        colors, cnt, inv_err = _build_orbits_and_check_invariance_t3(n, gens, B, ocol)
    elif t == 5:
        colors, cnt, inv_err = _build_orbits_and_check_invariance_t5(n, gens, B, ocol)
    else:
        raise NotImplementedError(f"Unsupported t={t}")

    assert cnt == len(ocol), f"Orbit count mismatch: {cnt} vs {len(ocol)}"
    assert inv_err == 0, f"G-invariance violated on {inv_err} hyperedges!"
    counts = [int(np.count_nonzero(colors == c)) for c in range(num_colors)]
    print(f"  [PASS] {len(gens)} valid bijections | {cnt} orbits | {len(colors):,} hyperedges | 0 invariance errors")
    print(f"  Color balance across all C({n},{t})={math.comb(n, t):,} edges: {counts}")

    # 4. ZERO-PRUNING exhaustive clique check
    if t == 3 and all(k == 4 for k in ks):
        expected = math.comb(n, 4)
        t1 = time.perf_counter()
        chk1, mono1 = _zero_pruning_check_k4_t3(colors, B, n)
        dt1 = time.perf_counter() - t1
        assert chk1 == expected and mono1 == 0
        print(f"  [PASS - Numba zero-pruning] Checked ALL {chk1:,} / {expected:,} 4-subsets -> {mono1} monochromatic ({dt1:.2f}s)")

        t2 = time.perf_counter()
        chk2, mono2 = _pure_numpy_check_k4_t3(colors, B, n)
        dt2 = time.perf_counter() - t2
        assert chk2 == expected and mono2 == 0
        print(f"  [PASS - PURE NumPy (no Numba!)] Checked ALL {chk2:,} / {expected:,} 4-subsets -> {mono2} monochromatic ({dt2:.2f}s)")
    elif t == 5 and all(k == 6 for k in ks):
        expected = math.comb(n, 6)
        t1 = time.perf_counter()
        chk1, mono1 = _zero_pruning_check_k6_t5(colors, B, n)
        dt1 = time.perf_counter() - t1
        assert chk1 == expected and mono1 == 0
        print(f"  [PASS - ZERO-PRUNING 6-LOOP] Checked ALL {chk1:,} / {expected:,} 6-subsets -> {mono1} monochromatic ({dt1:.2f}s)")
    else:
        raise NotImplementedError(f"No zero-pruning checker implemented for t={t}, ks={ks}")

    print(f"  >>> CERTIFIED MATHEMATICAL THEOREM: {data['ramsey_target']} >= {n + 1} (audit time {time.perf_counter()-t0:.2f}s) <<<")


def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        wdir = Path(__file__).parent / "witnesses"
        paths = sorted(wdir.glob("*.json"))
    assert paths, "No witness JSON files found!"
    for p in paths:
        verify_witness_file(p)
    print("\nALL WITNESSES PASSED 100% ZERO-PRUNING EXHAUSTIVE CERTIFICATION!")


if __name__ == "__main__":
    main()
