# Improved Lower Bounds for Hypergraph Ramsey Numbers $R(4,4,4;3)$ and $R(6,6;5)$

This repository provides self-contained, group-invariant explicit coloring witnesses and a standalone, **zero-pruning exhaustive verifier** establishing two new lower bounds for small hypergraph Ramsey numbers tracked in Stanisław P. Radziszowski's Dynamic Survey *Small Ramsey Numbers* (E-JC #DS1, Section 7.1):

| Hypergraph Ramsey Number | Previous Best Lower Bound (late 2026) | New Lower Bound | Improvement | Explicit Group Action | Witness Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$R(4, 4, 4; 3)$** | $\ge 84$ (H. Yang, Sept 2026; DS1 Rev. 18: $\ge 79$) | **$\ge 102$** ($n = 101$) | **+18** | Diagonal $2 \times \mathrm{AGL}^{(16)}(1, \mathbb{F}_{49}) + 3$ fixed apexes (order $147$, 1,162 triple orbits) | 12.8 KB |
| **$R(6, 6; 5)$** | $\ge 84$ (C. Lysenstøen, Nov 2026, arXiv:2611.25143; DS1 Rev. 18: $\ge 72$) | **$\ge 87$** ($n = 86$) | **+3** | $\mathrm{AGL}(1, \mathbb{F}_{83}) + 3$ fixed apexes (order $6{,}806$, 5,159 5-subset orbits) | 42.6 KB |

---

## Independent Zero-Pruning Verification (1 Command)

Every witness is verified from scratch by [`verify_witnesses.py`](verify_witnesses.py), which has **zero dependencies on the search code** and uses **no branch-and-bound pruning or symmetry reduction** when checking cliques:

```bash
pip install numpy numba
python verify_witnesses.py
```

What `verify_witnesses.py` checks on each JSON witness in `witnesses/`:
1. Every generator is a valid bijection on $\{0, \dots, n-1\}$.
2. Reconstructs the exact orbit partition of all $\binom{n}{t}$ $t$-subsets under the generators and verifies strict $G$-invariance ($C(g(e)) = C(e)$ for every generator $g$ and every $t$-subset $e$).
3. **Exhaustively visits EVERY SINGLE ONE of all $\binom{n}{k}$ $k$-subsets** and asserts `checked == math.comb(n, k)` and `monochromatic == 0`:
   - **$R(4,4,4;3) \ge 102$ ($n = 101$)**: checks all **4,082,925** quadruples twice — once via Numba zero-pruning nested loops (~1.0s) and once via **pure NumPy with no Numba** (~0.9s).
   - **$R(6,6;5) \ge 87$ ($n = 86$)**: checks all **470,155,077** 6-subsets via literal 6-nested loops with zero pruning (~1.5s on a 16-core/32-thread CPU).

---

## Key Methodological Ideas

Prior 2026 group-invariant hypergraph constructions relied primarily on CDCL SAT solvers (e.g., CaDiCaL / Kissat), which restricted searches to very large single-orbit groups (10–40 orbits) that rapidly encounter stabilizer obstructions (order-4/Klein-4 elements for $K_4^{(3)}$, order-3/6 elements for $K_6^{(5)}$) or pair-orbit collapse under apex extensions.

Our constructions overcome this barrier via three complementary mechanisms:
1. **Low-Stabilizer Multi-Orbit Diagonal Actions (No Layer Swap)**: Acting diagonally with an odd-order multiplicative subgroup (stripping the full 2-Sylow subgroup) across multiple vertex copies $V = \mathbb{F}_q \times \{0, \dots, m-1\} \cup \{\infty_1, \dots, \infty_r\}$ without layer-swap symmetry avoids order-4 stabilizers and breaks pair-orbit collapse when adding fixed apex points. For $R(4,4,4;3)$, the order-3 multiplicative subgroup of $\mathbb{F}_{49}^\times$ yields 16 pair-orbits per copy and 1,162 triple orbits on $n = 101$ vertices.
2. **Dynamic Clause-Weighting Local Search (32-Thread PAWS-ProbSAT)**: Rather than CDCL, shared-memory multi-threaded ProbSAT with dynamic hyperedge weighting (PAWS) scales effortlessly to 1,000–5,500 orbit variables and tens of thousands of hypergraph clauses on a single consumer workstation (AMD Ryzen 9 7945HX, 16 GB RAM).
3. **Iterative Apex Warm-Start Climbing**: Extending $n \to n+1$ by copying all existing $n$-vertex orbit colors from a certified $n$-vertex witness, initializing only the newly created apex-incident orbits, and applying a 1–2% random perturbation accelerates $R(6,6;5)$ discovery by over $100\times$ (climbing $n = 84 \to 85 \to 86$ in seconds per step).

---

## Files

- `witnesses/R_4_4_4_3_n101_2x_AGL1_F49_idx16+3inf.json`: Certified witness proving $R(4,4,4;3) \ge 102$.
- `witnesses/R_6_6_5_n86_AGL1_F83+3inf.json`: Certified witness proving $R(6,6;5) \ge 87$.
- `verify_witnesses.py`: Standalone zero-pruning independent verifier.
- `paper.tex`: Short technical note (LaTeX source).

## Transparency Note

The search scripts and experimental workflow were developed with AI coding assistance. All mathematical claims rest solely on the explicit finite witnesses above, each independently audited with zero pruning over every $k$-subset by `verify_witnesses.py`.
