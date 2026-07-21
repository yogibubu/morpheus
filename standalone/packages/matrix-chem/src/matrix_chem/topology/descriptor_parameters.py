"""
Centralized numerical parameters for continuous molecular descriptors.
All parameters are collected here to allow easy optimization.
"""

# ------------------------------------------------------------
# Continuous Coordination Number (CNA)
# ------------------------------------------------------------
#
# The continuous coordination number is computed as:
#
#   g = 0.5 * [ 1 - erf( CNA_ALPHA * ( R / Reff - 1 ) ) ]
#
# where the effective reference distance is:
#
#   Reff = CNA_SCALE * ( Rcov_i + Rcov_j )
#
# The scaling factor CNA_SCALE follows the Grimme philosophy
# (4/3 in early works) and ensures that
# standard covalent bonds give g ≈ 1.
#
# ------------------------------------------------------------

CNA_ALPHA = 8.0
CNA_SCALE = 4.0 / 3.0

# ------------------------------------------------------------
# Bond order parameters (normalized Pauling form)
# ------------------------------------------------------------
#
# The bond order used in atomic_synthons is:
#
#   BO = exp( - ( R / (Rcov_i + Rcov_j) - 1 ) / lambda )
#
# The decay parameter lambda is dimensionless and related to the
# classical Pauling decay length lambda_P (in Å) through:
#
#   lambda = lambda_P / (Rcov_i + Rcov_j)
#
# To reproduce the Pauling bond order for a C–C bond, we adopt:
#
#   Rcov(C)  = 0.75 Å
#   lambda_P = 0.30 Å   (strong bonds, BO ≳ 1)
#
# giving:
#
#   lambda_strong = 0.30 / (2 * 0.75) = 0.20
#
# It is known that for weak bonds and transition states (BO < 1)
# a larger Pauling decay length provides a more realistic decay.
# We therefore also define:
#
#   lambda_P_weak = 0.60 Å
#   lambda_weak   = 0.60 / (2 * 0.75) = 0.40
#
# The use of these two limits is deferred to atomic_synthons.
# ------------------------------------------------------------

BO_LAMBDA_STRONG = 0.20
BO_LAMBDA_WEAK = 0.40


# Smooth switching parameter for bond-order decay
# Controls the width of the strong-to-weak bond transition
# Typical values: 15–30; default chosen to give a crossover over
# ~0.05–0.1 Å for C–C bonds
ALPHA_LAMBDA = 20.0

# Width of the C2 ramps used to resolve a continuous multiplicity order into
# sigma, first-pi and second-pi (pi-pi) occupancies.  Outside the short ramp
# immediately above integer order 1 or 2, the decomposition is exactly the
# usual ordinal one and its components sum identically to the total BO.
BO_COMPONENT_RAMP_WIDTH = 0.10

# Geometric Pauling bond order used when Mayer data are unavailable:
#
#   M_ij = exp[(Rcov_i + Rcov_j - R_ij) / BO_PAULING_DECAY_ANGSTROM]
#
# Unlike the coordination-normalized topological edge index, this quantity
# deliberately retains single/double/triple-bond radial information and can
# therefore be resolved into sigma, first-pi and second-pi occupancies.
BO_PAULING_DECAY_ANGSTROM = 0.30


# Minimum bond order used in atomic descriptors
# (robustness for TS and weak bonds)
BO_MIN_DESC = 1.0


# ------------------------------------------------------------
# EAN normalization
# ------------------------------------------------------------

EAN_SIGMA = 1.0


# ------------------------------------------------------------
# Reference angular sums for strain (degrees)
# One reference geometry per coordination number
# ------------------------------------------------------------

REF_ANGLE_SUM = {
    2: 180.0,  # linear
    3: 360.0,  # trigonal planar
    4: 6 * 109.47122,  # tetrahedral (Platonic)
    5: 720.0,  # trigonal bipyramidal (approx., non-Platonic)
    6: 12 * 90.0,  # octahedral (Platonic)
    7: 1260.0,  # capped octahedron (approx., non-Platonic)
    8: 24 * 90.0,  # cubic (Platonic)
    12: 30 * 63.43495,  # icosahedral (Platonic)
}
