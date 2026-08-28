#### Python 3.6.4 :: Anaconda, Inc.
import os
import logging
import warnings
import numpy as np
import pandas as pd

from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, vstack, triu
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


logger = logging.getLogger(__name__)


def S_construction(packed_arguments, resolution, dist):
    """
    Construct a feature matrix from binned short-format chromatin contact maps
    and perform dimensionality reduction using SVD or NMF.

    The input files are expected to contain binned short-format contact maps.
    Each row represents an interaction between two genomic bins. Input files
    may be tab-separated or space-separated.

    Expected input columns
    ----------------------
    Column 0 : chromosome of the first interacting bin
    Column 1 : genomic coordinate of the first bin
    Column 2 : chromosome of the second interacting bin
    Column 3 : genomic coordinate of the second bin
    Column 4 : interaction/contact value

    Example
    -------
    chr1    10000    chr1    20000    15

    Parameters
    ----------
    packed_arguments : tuple
        Tuple containing:
            mat      : list of binned short-format contact map files
            chrom    : chromosome to process
            c_length : chromosome length in number of bins
            method   : dimensionality-reduction method ('svd' or 'nmf')
            num_f    : number of latent features/components
            diag     : whether to include diagonal contacts

    resolution : int
        Bin size (resolution) of the input contact maps in base pairs.
        Genomic coordinates are divided by this value to obtain matrix
        bin indices. For example, for a 10-kb contact map, resolution
        should be 10000.

    dist : int or float
        Maximum genomic distance, in base pairs, between interacting bins
        to include in the analysis.

    Returns
    -------
    Sr_tilde : numpy.ndarray
        Reduced feature representation obtained using SVD or NMF.
    """

    # Unpack arguments.
    mat, chrom, c_length, method, num_f, diag = packed_arguments

    logger.info('Constructing feature matrix for chromosome {} from {} contact map(s).'.format(chrom, len(mat)))

    # Store the processed interaction vector from each contact map.
    Cr = []

    # Get indices from the upper triangular portion of the chromosome
    # contact matrix.
    ind = np.triu_indices(c_length)

    # Keep interactions between bins that fall within the specified
    # genomic distance.
    distance_mask = np.abs(ind[1] - ind[0]) < (dist / resolution)
    ind = (ind[0][distance_mask], ind[1][distance_mask])

    # Exclude self-interactions on the matrix diagonal when requested.
    if not diag:
        non_diag_mask = ind[0] != ind[1]
        ind = (
            ind[0][non_diag_mask],
            ind[1][non_diag_mask]
        )

    # Process each binned short-format contact map.
    for m in mat:

        # logger.info('Reading contact map: {}'.format(m))

        # Read the binned short-format contact map.
        # Both tab-separated and space-separated files are supported.
        c = pd.read_csv(
            m,
            delim_whitespace=True,
            header=None,
            index_col=None
        )

        # Keep rows in which the first interacting bin belongs to the
        # chromosome of interest.
        c = c.loc[c[0] == chrom]

        # Keep intra-chromosomal contacts only.
        # In short format, columns 0 and 2 contain the chromosome labels
        # for the two interacting bins.
        intra = c.loc[c[0] == c[2]].reset_index(drop=True)

        # Convert genomic coordinates from the short-format contact map
        # into chromosome contact-matrix bin indices.
        row_idx = (intra[1] / resolution).astype(int)
        col_idx = (intra[3] / resolution).astype(int)

        # Build the chromosome contact matrix using the interaction values
        # stored in column 4 of the short-format input file.
        temp = csr_matrix(
            (intra[4], (row_idx, col_idx)),
            shape=(c_length, c_length)
        ).toarray()

        # Convert the contact matrix to a symmetric matrix while avoiding
        # double-counting diagonal values.
        temp = temp + temp.T - np.diag(np.diag(temp))

        # Extract the selected contacts and apply a log(x + 1)
        # transformation.
        Cr.append(np.log(temp[ind] + 1))

    # Stack the processed contact vectors from all input contact maps.
    Sr = np.vstack(Cr)

    logger.info('Feature matrix constructed for chromosome {}.'.format(chrom))

    # Perform dimensionality reduction.
    if method == 'svd':
        logger.info('Running SVD with {} features.'.format(num_f))
        Sr_tilde, sigma, Ur = svds(Sr, k=num_f)

    elif method == 'nmf':
        logger.info('Running NMF with {} features.'.format(num_f))
        Sr_tilde = NMF(
            n_components=num_f,
            init='random'
        ).fit_transform(Sr)

    else:
        raise ValueError(
            "Unknown method: '{}'. Expected 'svd' or 'nmf'.".format(method)
        )

    logger.info('Feature reduction completed for chromosome {}.'.format(chrom))

    return Sr_tilde


def correction(S_tilde, k, metadata, sigma=15):
    """
    Correct dataset/batch-specific effects in the low-dimensional
    representation using mutual nearest neighbors (MNNs).

    The dataset containing the largest number of cells is used as the
    reference dataset. Each remaining dataset is aligned to the reference
    using a dataset-specific correction vector calculated from mutual
    nearest-neighbor pairs.

    Parameters
    ----------
    S_tilde : pandas.DataFrame
        Low-dimensional representation of the cells. Rows correspond to
        cells and columns correspond to latent features/components.

    k : int
        Number of nearest neighbors used to identify mutual nearest
        neighbors between the reference and target datasets.

    metadata : pandas.DataFrame
        Cell-level metadata containing a 'dataset/batch' column.

    sigma : float, optional
        Bandwidth of the Gaussian kernel used to weight correction vectors.
        Default is 15.

    Returns
    -------
    S_hat : pandas.DataFrame
        Batch-corrected low-dimensional representation.
    """

    # Identify available datasets/batches.
    avail_sets = np.unique(metadata['dataset/batch'])

    logger.info('Starting dataset/batch correction for {} dataset(s)/batch(es).'.format(len(avail_sets)))

    # Store dataset-specific cell counts, embeddings, and correction vectors.
    num_cells, S_tilde_l, m = {}, {}, {}

    if len(avail_sets) > 1:

        # Separate cells according to dataset/batch.
        for l in avail_sets:
            S_tilde_l[l] = S_tilde.loc[
                metadata.index[metadata['dataset/batch'] == l]
            ]
            num_cells[l] = len(S_tilde_l[l])

        # Use the dataset with the largest number of cells as the reference.
        refer_set = max(num_cells, key=num_cells.get)
        logger.info('Reference dataset/batch: {} ({} cells).'.format(refer_set, num_cells[refer_set]))

        # Treat all remaining datasets as target datasets.
        target_sets = avail_sets.tolist()
        target_sets.remove(refer_set)

        # Initialize the integrated representation with the reference dataset.
        S_hat = S_tilde_l[refer_set]

        for l in target_sets:

            logger.info('Correcting dataset/batch: {} ({} cells).'.format(l, num_cells[l]))

            # Compute pairwise distances between reference and target cells.
            pdist = pairwise_distances(
                S_tilde_l[refer_set],
                S_tilde_l[l]
            )

            # Identify k nearest neighbors in both directions.
            knn_i = pdist.argsort(axis=1)[:, :k]
            knn_j = pdist.argsort(axis=0)[:k, :]

            # Identify mutual nearest-neighbor pairs.
            Mi, Mj = [], []

            for i in range(knn_i.shape[0]):
                mnn = np.intersect1d(
                    knn_i[i],
                    np.where(knn_j == i)[1]
                )

                Mi.extend([i] * len(mnn))
                Mj.extend(mnn.tolist())

            logger.info('Found {} mutual nearest-neighbor pairs for dataset/batch {}.'.format(len(Mi), l))

            # If no mutual nearest neighbors are found, no correction is
            # applied to this target dataset.
            if len(Mi) == 0:
                logger.warning(
                    'No mutual nearest neighbors found for dataset/batch {}. '
                    'No correction applied.'.format(l)
                )

                S_hat = pd.concat([S_hat, S_tilde_l[l]])
                continue

            # Construct dataset-specific correction vectors from MNN pairs.
            M_bar = (
                S_tilde_l[refer_set].iloc[Mi].values
                -
                S_tilde_l[l].iloc[Mj].values
            )
            # M_bar = (
            #     S_tilde_l[refer_set]
            #     .loc[S_tilde_l[refer_set].index[Mi]]
            #     .values
            #     -
            #     S_tilde_l[l]
            #     .loc[S_tilde_l[l].index[Mj]]
            #     .values
            # )
            
            # Calculate Gaussian weights for the MNN correction vectors.
            gaus_kernel = np.exp(
                -np.linalg.norm(M_bar, axis=1) ** 2 /
                (2 * sigma ** 2)
            )

            gaus_weights = gaus_kernel / np.sum(gaus_kernel)

            # Calculate the dataset-specific correction vector.
            m[l] = np.sum(
                np.multiply(
                    M_bar,
                    gaus_weights.reshape(len(gaus_weights), 1)
                ),
                axis=0
            )

            # Apply the correction vector to the target dataset.
            S_tilde_l[l] = S_tilde_l[l] + m[l]

            # Add the corrected target dataset to the integrated representation.
            S_hat = pd.concat([S_hat, S_tilde_l[l]])
            logger.info('Correction completed for dataset/batch {}.'.format(l))

        logger.info('Dataset/batch correction completed.')

    else:
        logger.info('No integration/correction needed.')
        S_hat = S_tilde

    return S_hat


def read_chrom_size(chrom):
    """
    Read chromosome-size information from one or multiple files.

    If multiple chromosome-size files are provided, the files are combined
    and the maximum chromosome size across files is used for each chromosome.

    Parameters
    ----------
    chrom : str
        Path to one chromosome-size file or a space-separated string
        containing paths to multiple chromosome-size files.

    Returns
    -------
    chrom_size : pandas.DataFrame
        DataFrame containing chromosome names as the index and chromosome
        sizes in column 1.
    """

    # Split input into individual chromosome-size files.
    chrom_files = chrom.split(' ')

    logger.info('Reading chromosome-size information from {} file(s).'.format(len(chrom_files)))

    if len(chrom_files) > 1:

        # Read the first chromosome-size file.
        chrom_add = pd.read_csv(
            chrom_files[0],
            sep='\t',
            header=None,
            index_col=0
        )

        # Add chromosome sizes from the remaining files as new columns.
        for i in range(len(chrom_files) - 1):
            chrom_add = pd.concat(
                [
                    chrom_add,
                    pd.read_csv(
                        chrom_files[i + 1],
                        sep='\t',
                        header=None,
                        index_col=0
                    )
                ],
                axis=1
            )

        logger.info('Combining chromosome sizes across input files.')

        # Use the maximum chromosome size across all input files.
        chrom_size = pd.DataFrame(
            {1: chrom_add.max(axis=1).values},
            index=chrom_add.index
        )

    else:
        # If only one file is provided, use it directly.
        chrom_size = pd.read_csv(
            chrom_files[0],
            sep='\t',
            header=None,
            index_col=0
        )

    logger.info('Chromosome-size information loaded.')

    return chrom_size

