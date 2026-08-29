import math
import numpy as np

from .descriptor_parameters import (
    REF_ANGLE_SUM,
    BO_MIN_DESC,
    EAN_SIGMA,
)
from .pykko_radii import PYYKKO
from .covalent_radii import covalent_radius
from .vdw_radii import descriptor_vdw_radius
from .continuous_graph import (
    BondOrderComponents,
    bond_order_components as resolve_bond_order_components,
    continuous_coordination_number,
    hermite_c1,
    hermite_slope,
    pauling_bond_order,
    principal_quantum_number,
)


# ============================================================
# Utility
# ============================================================


def angle_between(v1, v2):
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den < 1e-12:
        return 0.0
    c = np.clip(num / den, -1.0, 1.0)
    return math.degrees(math.acos(c))


def nval_main_group(Z):
    if Z == 1:
        return 1
    if 3 <= Z <= 10:
        return {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}[Z]
    if 11 <= Z <= 18:
        return {11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8}[Z]
    if 31 <= Z <= 36:
        return {31: 3, 32: 4, 33: 5, 34: 6, 35: 7, 36: 8}[Z]
    if 49 <= Z <= 54:
        return {49: 3, 50: 4, 51: 5, 52: 6, 53: 7, 54: 8}[Z]
    return 0


# ============================================================
# Atomic Synthons
# ============================================================


class AtomicSynthons:
    def __init__(self, Z, coords, neighbors, coordination_numbers=None):
        self.Z = Z
        self.coords = coords
        self.neighbors = neighbors
        self.natoms = len(Z)
        self._theta_bar = None
        self._coordination_numbers = (
            None
            if coordination_numbers is None
            else np.asarray(coordination_numbers, dtype=float).reshape(self.natoms)
        )
        self._all_neighbors = None
        self._bond_order_cache = {}
        self._bond_order_components_cache = {}
        self._effective_radius_cache = np.full(self.natoms, np.nan)

    # --------------------------------------------------------
    # Continuous coordination number (CNA)
    # --------------------------------------------------------

    def cna(self, i):
        if self._coordination_numbers is not None:
            return float(self._coordination_numbers[i])
        if self._all_neighbors is None:
            self._all_neighbors = [
                [neighbor for neighbor in range(self.natoms) if neighbor != atom]
                for atom in range(self.natoms)
            ]
        return continuous_coordination_number(i, self.Z, self.coords, self._all_neighbors)

    # --------------------------------------------------------
    # Effective covalent radius
    # --------------------------------------------------------

    def covalent_radius_eff(self, i):
        """Return the unique synthon effective radius.

        This descriptor interpolation is distinct from the radius internal to
        graph perception; neither is a selectable alternative model.
        """
        cached = self._effective_radius_cache[i]
        if np.isfinite(cached):
            return float(cached)
        atomic_number = int(self.Z[i])
        coordination = self.cna(i)
        table = PYYKKO[atomic_number]
        if not table:
            fallback = covalent_radius(atomic_number)
            if fallback is None:
                raise ValueError(
                    f"no covalent radius is available for atomic number {atomic_number}"
                )
            result = float(fallback)
            self._effective_radius_cache[i] = result
            return result
        keys = sorted(table)
        if coordination <= keys[0]:
            result = table[keys[0]]
            self._effective_radius_cache[i] = result
            return result
        if coordination >= keys[-1]:
            result = table[keys[-1]]
            self._effective_radius_cache[i] = result
            return result
        lower = int(math.floor(coordination))
        if lower not in table:
            lower = max(key for key in keys if key <= lower)
        upper = min(key for key in keys if key > lower)
        fraction = (coordination - lower) / (upper - lower)
        result = hermite_c1(
            fraction,
            table[lower],
            table[upper],
            hermite_slope(table, keys, lower),
            hermite_slope(table, keys, upper),
        )
        self._effective_radius_cache[i] = result
        return result

    # --------------------------------------------------------
    # Bond order
    # --------------------------------------------------------

    def bond_order(self, i, j):
        """Return the sole ORACLE bond order: Mayer or geometric Pauling."""
        key = (i, j) if i < j else (j, i)
        cached = self._bond_order_cache.get(key)
        if cached is not None:
            return cached
        ext = getattr(self, "_external_bond_orders", None)
        if ext:
            if key in ext:
                result = float(ext[key])
                self._bond_order_cache[key] = result
                return result
        result = pauling_bond_order(i, j, self.Z, self.coords)
        self._bond_order_cache[key] = result
        return result

    def bond_order_desc(self, i, j):
        return max(self.bond_order(i, j), BO_MIN_DESC)

    def bond_order_components(self, i, j) -> BondOrderComponents:
        key = (i, j) if i < j else (j, i)
        cached = self._bond_order_components_cache.get(key)
        if cached is None:
            cached = resolve_bond_order_components(self.bond_order(i, j))
            self._bond_order_components_cache[key] = cached
        return cached

    def bond_order_sigma(self, i, j):
        return self.bond_order_components(i, j).sigma

    def bond_order_pi(self, i, j):
        """First pi-bond occupancy (kept as the historical public name)."""
        return self.bond_order_components(i, j).pi

    def bond_order_pi_pi(self, i, j):
        """Second pi-bond occupancy, non-zero for triple-bond character."""
        return self.bond_order_components(i, j).pi_pi

    def bond_order_total_pi(self, i, j):
        return self.bond_order_components(i, j).total_pi

    def bond_order_indices(self, i, j):
        """Return the canonical ``(sigma, pi, pi_pi)`` bond indices."""
        components = self.bond_order_components(i, j)
        return components.sigma, components.pi, components.pi_pi

    def sigma_index(self, i):
        return sum(self.bond_order_sigma(i, j) for j in self.neighbors[i])

    def pi_index(self, i):
        return sum(self.bond_order_pi(i, j) for j in self.neighbors[i])

    def pi_pi_index(self, i):
        return sum(self.bond_order_pi_pi(i, j) for j in self.neighbors[i])

    # --------------------------------------------------------
    # Electronic domains
    # --------------------------------------------------------

    def nlp_nos(self, i):
        Z = int(self.Z[i])
        nval = nval_main_group(Z)
        if nval == 0:
            return 0.0, 0.0

        CNA = self.cna(i)
        Npi = self.pi_index(i) + self.pi_pi_index(i)
        Nres = nval - CNA - Npi

        NOS = float(int(round(Nres)) % 2)
        NLP = 0.5 * (Nres - NOS)
        if NLP < 0.0:
            NLP = 0.0
        return NLP, NOS

    def electron_domains(self, i):
        CNA = self.cna(i)
        NLP, NOS = self.nlp_nos(i)
        return CNA + NLP + NOS

    def _electron_domains(self, i):
        return self.electron_domains(i)

    # --------------------------------------------------------
    # Reference angles
    # --------------------------------------------------------

    def theta_ref(self, N):
        if self._theta_bar is None:
            self._theta_bar = {
                int(k): REF_ANGLE_SUM[k] / (k * (k - 1) / 2) for k in REF_ANGLE_SUM if k >= 2
            }

        keys = sorted(self._theta_bar.keys())
        if N <= keys[0]:
            return self._theta_bar[keys[0]]
        if N >= keys[-1]:
            return self._theta_bar[keys[-1]]

        k0 = int(math.floor(N))
        if k0 not in self._theta_bar:
            k0 = max(k for k in keys if k <= k0)
        k1 = min(k for k in keys if k > k0)

        t = (N - k0) / (k1 - k0)
        y0 = self._theta_bar[k0]
        y1 = self._theta_bar[k1]

        def slope(k):
            return hermite_slope(self._theta_bar, keys, k)

        return hermite_c1(t, y0, y1, slope(k0), slope(k1))

    # --------------------------------------------------------
    # Strain
    # --------------------------------------------------------

    def strain(self, i):
        neigh = self.neighbors[i]
        if len(neigh) < 2:
            return 0.0

        Ndom = self.electron_domains(i)
        theta0 = self.theta_ref(Ndom)
        c0 = math.cos(math.radians(theta0))

        Ri = self.coords[i]
        S = 0.0
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                v1 = self.coords[neigh[a]] - Ri
                v2 = self.coords[neigh[b]] - Ri
                c = math.cos(math.radians(angle_between(v1, v2)))
                S += (c - c0) ** 2

        return math.sqrt(S / (len(neigh) * (len(neigh) - 1) / 2))

    # --------------------------------------------------------
    # Charge / Polarizability / Hindrance
    # --------------------------------------------------------

    def charge(self, i):
        ext = getattr(self, "_external_charges", None)
        if ext and i in ext:
            return float(ext[i])
        from .periodic_properties import periodic_atomic_properties

        Zi = int(self.Z[i])
        chi_i = periodic_atomic_properties(Zi).electronegativity
        ni = principal_quantum_number(Zi)

        q = 0.0
        for j in self.neighbors[i]:
            Zj = int(self.Z[j])
            chi_j = periodic_atomic_properties(Zj).electronegativity
            nj = principal_quantum_number(Zj)
            bo = self.bond_order_desc(i, j)
            q += (chi_j - chi_i) / ((ni + nj) * bo)
        return q

    def polarizability(self, i):
        from .periodic_properties import periodic_atomic_properties

        Zi = int(self.Z[i])
        ai = periodic_atomic_properties(Zi).polarizability_angstrom3
        ni = principal_quantum_number(Zi)

        a = ai
        for j in self.neighbors[i]:
            Zj = int(self.Z[j])
            aj = periodic_atomic_properties(Zj).polarizability_angstrom3
            nj = principal_quantum_number(Zj)
            bo = self.bond_order_desc(i, j)
            a += (aj - ai) / ((ni + nj) * bo)
        return a

    def hindrance(self, i):
        Zi = int(self.Z[i])
        from .periodic_properties import periodic_atomic_properties

        hi = descriptor_vdw_radius(Zi)
        if hi is None:
            hi = periodic_atomic_properties(Zi).vdw_radius_angstrom
        ni = principal_quantum_number(Zi)

        H = hi
        for j in self.neighbors[i]:
            Zj = int(self.Z[j])
            hj = descriptor_vdw_radius(Zj)
            if hj is None:
                hj = periodic_atomic_properties(Zj).vdw_radius_angstrom
            nj = principal_quantum_number(Zj)
            bo = self.bond_order_desc(i, j)
            H += (hj - hi) / ((ni + nj) * bo)
        return H

    # --------------------------------------------------------
    # Covalency / Delocalization
    # --------------------------------------------------------

    def covalency(self, i):
        neigh = self.neighbors[i]
        if not neigh:
            return 0.0
        ri = self.covalent_radius_eff(i)
        Ri = self.coords[i]
        C = 0.0
        for j in neigh:
            rj = self.covalent_radius_eff(j)
            d = np.linalg.norm(self.coords[j] - Ri)
            C += (ri + rj) / (d + ri + rj)
        return C / len(neigh)

    def delocalization(self, i):
        neigh = self.neighbors[i]
        if len(neigh) <= 1:
            return 0.0
        Ri = self.coords[i]
        dists = [np.linalg.norm(self.coords[j] - Ri) for j in neigh]
        dmean = sum(dists) / len(dists)
        if dmean <= 0.0:
            return 0.0
        return sum(abs(d - dmean) for d in dists) / (len(dists) * dmean)

    # --------------------------------------------------------
    # Spin / EAN / Zeff
    # --------------------------------------------------------

    def spin_density(self, i):
        _, NOS = self.nlp_nos(i)
        return NOS

    def EAN(self, i):
        vals = [
            self.charge(i),
            self.covalency(i),
            self.delocalization(i),
            self.strain(i),
        ]
        norm = [math.erf(v / EAN_SIGMA) for v in vals]
        r = math.sqrt(sum(v * v for v in norm))
        return r / (1.0 + r)

    def Zeff(self, i):
        return self.Z[i] - 0.5 + self.EAN(i)

    # --------------------------------------------------------
    # Discrete signature
    # --------------------------------------------------------

    def canonical_signature(self, i):
        D = int(round(self.cna(i)))
        NED = int(round(self.electron_domains(i)))
        return (self.Z[i], False, NED, D)

    def canonical_signature_str(self, i):
        Z, A, NED, D = self.canonical_signature(i)
        return f"{Z}-{int(A)}-{NED}-{D}"
