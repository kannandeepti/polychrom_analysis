r"""
Computing compartment strengths from polychrom simulations
==========================================================

Script to create saddle plots from simulated contact maps
and compute compartment scores. Simulated contact maps are
processed in the same way as experimental Hi-C maps. Interative
correction is applied to raw bin values so that the rows and columns
of the contact map sum to 1. Then, we divide each diagonal of the
balanced map by its mean in order to produce an ``observed over expected''
map. This observed over expected map is mean-centered so that positive
entries indicate enrichment of contacts above the mean and negative entries
indicate depletion of contacts below the mean. The first eigenvector (E1)
of the resulting map tracks A/B compartments. We then sort the rows and
columns of the observed over expected map by quantiles of E1 to produce a
coarse-grained obs/exp map with a smaller dimension. We then average over the
corners of this map to compute compartment scores, defined as
(AA + BB - 2AB) / (AA + BB + 2AB). 

Notes
-----
For real Hi-C data, the first eigenvector is aligned with a separate track, such
as GC content in order to distinguish A from B. Since no such track exists for
simulated contact maps, ideally E1 should first be correlated with the known A/B
identities to identify which corners of the saddle maps correspond to AA vs BB.
This has not yet been implemented programmatically and was checked by hand.

Deepti Kannan. 2022
"""

import numpy as np
from cooltools.lib import numutils
from cooltools.lib.numutils import (LazyToeplitz,
                                    iterative_correction_symmetric,
                                    observed_over_expected)


def process_data(filename, score_quantile=25, n_groups=38):
    """Process a raw simulated contact map (N x N) matrix via iterative correction, compute
    the observed over expected map, calculate eigenvalues and extract compartment scores.

    Parameters
    ----------
    filename : .npy file
        containing N x N contact map
    score_quantile : int in [0, 100]
        percentage quantile where top (quantile)% of contacts are averaged over to compute score
    n_groups : int
        saddle matrices will be of shape (n_groups + 2, n_groups + 2), i.e. the first eigenvector
        is sorted and coarse-grained into n_groups + 2 quantile bins.

    Returns
    -------
    cs2 : float
        over all comp score (AA + BB - 2AB) / (AA + BB + 2AB)
    AA_cs2 : float
        A compartment score (AA - AB) / (AA + AB)
    BB_cs2 : float
        B compartment score (BB - AB) / (BB + AB)

    """
    contactmap = np.load(filename)
    # balance the map
    mat_balanced, totalBias, report = iterative_correction_symmetric(
        contactmap, max_iter=50
    )
    mat_balanced /= np.mean(mat_balanced.sum(axis=1))
    # compute observed over expected
    mat_oe, dist_bins, sum_pixels, n_pixels = observed_over_expected(mat_balanced)
    # mean center and compute e1
    mat_oe_norm = mat_oe - 1.0
    eigvecs, eigvals = numutils.get_eig(mat_oe_norm, 3, mask_zero_rows=True)
    eigvecs /= np.sqrt(np.nansum(eigvecs**2, axis=1))[:, None]
    eigvecs *= np.sqrt(np.abs(eigvals))[:, None]
    e1 = eigvecs[0]
    # compute saddle matrices
    S, C = saddle(e1, mat_oe, n_groups)
    # compute compartment score from saddles
    cs2, AA_cs2, BB_cs2 = comp_score_2(S, C, score_quantile)
    return cs2, AA_cs2, BB_cs2


def saddle(e1, oe, n_bins):
    """Compute two matrices of interaction sums / interaction counts
    from an e1 track and an observed over expected contact map.

    Parameters
    ----------
    e1 : array-like (N,)
        first eigenvector of mean-centered observed over expected map
    oe : array-like (N, N)
        observed over expected contact map (not mean-centered)
    n_bins : int
        number of quantile bins to sort e1 into

    Returns
    -------
    S : np.ndarray[float] (n_bins + 2, n_bins + 2)
        sorted obs/exp contact map containing sum of contacts within each quantile
    C : np.ndarray[float] (n_bins + 2, n_bins + 2)
        matrix storing number of contacts in each quantile bin
    """

    e1_sort_inds = np.argsort(e1)
    sorted_map = oe[e1_sort_inds, :][:, e1_sort_inds]
    interaction_sum = np.zeros((n_bins + 2, n_bins + 2))
    interaction_count = np.zeros((n_bins + 2, n_bins + 2))
    bins_per_quantile = int(sorted_map.shape[0] / (n_bins + 2))
    for n in range(n_bins + 2):
        data = sorted_map[n * bins_per_quantile : (n + 1) * bins_per_quantile, :]
        for m in range(n_bins + 2):
            square = data[:, m * bins_per_quantile : (m + 1) * bins_per_quantile]
            square = square[np.isfinite(square)]
            interaction_sum[n, m] = np.sum(square)
            interaction_count[n, m] = float(len(square))
    interaction_count += interaction_count.T
    interaction_sum += interaction_sum.T
    return interaction_sum, interaction_count


def comp_score_2(S, C, quantile):
    """Compute normalized compartment score from interaction_sum, interaction_count
    saddle matrices. The score is essentially (AA + BB - 2AB) / (AA + BB + 2AB)
    where average contacts in the top quantile are considered.

    Parameters
    ----------
    S, C : 2D arrays, square, same shape
        Saddle sums and counts, respectively
    quantile : int in [0, 100]
        percentage quantile over which to compute comp score

    Returns
    -------
    cs2 : float
        over all comp score (AA + BB - 2AB) / (AA + BB + 2AB)
    AA_cs2 : float
        A compartment score (AA - AB) / (AA + AB)
    BB_cs2 : float
        B compartment score (BB - AB) / (BB + AB)

    """
    m, n = S.shape
    AA_oe, BB_oe, AB_oe, AA_ratios, BB_ratios, ratios = saddle_strength_A_B(S, C)
    ind = int(quantile // (100 / n))
    cs2 = (ratios[ind] - 1) / (ratios[ind] + 1)
    AA_cs2 = (AA_ratios[ind] - 1) / (AA_ratios[ind] + 1)
    BB_cs2 = (BB_ratios[ind] - 1) / (BB_ratios[ind] + 1)
    return cs2, AA_cs2, BB_cs2


def saddle_strength_A_B(S, C):
    """
    Parameters
    ----------
    S, C : 2D arrays, square, same shape
        Saddle sums and counts, respectively

    Returns
    -------
    Astrength : 1d array
        Ratios of cumulative corner interaction scores (AA/AB) with increasing extent.
    Bstrength : 1d array
        Ratios of cumulative corner interaction scores (BB/AB) with increasing extent

    """
    m, n = S.shape
    if m != n:
        raise ValueError("`saddledata` should be square.")

    AA_oe = np.zeros(n)
    BB_oe = np.zeros(n)
    AB_oe = np.zeros(n)
    AA_ratios = np.zeros(n)
    BB_ratios = np.zeros(n)
    ratios = np.zeros(n)
    for k in range(1, n):
        BB_sum = np.nansum(S[0:k, 0:k])
        AA_sum = np.nansum(S[n - k : n, n - k : n])
        BB_count = np.nansum(C[0:k, 0:k])
        AA_count = np.nansum(C[n - k : n, n - k : n])
        AA = AA_sum / AA_count
        BB = BB_sum / BB_count
        intra_sum = AA_sum + BB_sum
        intra_count = AA_count + BB_count
        intra = intra_sum / intra_count
        AB_sum = np.nansum(S[0:k, n - k : n])
        inter_sum = AB_sum + np.nansum(S[n - k : n, 0:k])
        AB_count = np.nansum(C[0:k, n - k : n])
        inter_count = AB_count + np.nansum(C[n - k : n, 0:k])
        inter = inter_sum / inter_count
        AB = AB_sum / AB_count
        AA_ratios[k] = AA / AB
        BB_ratios[k] = BB / AB
        AA_oe[k] = AA
        BB_oe[k] = BB
        AB_oe[k] = AB
        ratios[k] = intra / inter
    return AA_oe, BB_oe, AB_oe, AA_ratios, BB_ratios, ratios
