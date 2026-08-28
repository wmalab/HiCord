#### Python 3.6.4 :: Anaconda, Inc.

import os
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp

from sklearn.decomposition import PCA

from utils import *


logger = logging.getLogger(__name__)


def construct_feature_matrix(input_path, chrom_size, resolution, dist,
                             method, num_f, diag=False, parallel=False):
    """
    Construct the feature matrix across all chromosomes.

    Contact maps are collected from dataset subdirectories under the input
    directory. For each chromosome, S_construction() generates a reduced
    feature representation, and chromosome-specific representations are
    concatenated to form the final feature matrix.
    """

    logger.info('Begin constructing feature matrix for all data.')

    datasets = [
        f.path for f in os.scandir(input_path)
        if f.is_dir()
    ]

    maps = [
        f.path
        for d in datasets
        for f in os.scandir(d)
        if not f.name.startswith('.')
    ]

    logger.info('Found {} contact maps.'.format(len(maps)))

    Sr_tilde = []

    if not parallel:
        for chrom in chrom_size.index:

            logger.info(
                'Working on chromosome ---------------------- {}'.format(chrom)
            )

            c_length = int(
                np.ceil(chrom_size.loc[chrom].values[0] / resolution)
            )

            out = S_construction(
                (
                    maps,
                    chrom,
                    c_length,
                    method,
                    num_f,
                    diag
                ),
                resolution,
                dist
            )

            Sr_tilde.append(out)

    else:
        processes = mp.cpu_count()

        logger.info(
            'Parallel processing enabled using {} processes.'.format(processes)
        )

        pool = mp.Pool(processes=processes)

        arguments = [
            (
                (
                    maps,
                    chrom,
                    int(np.ceil(chrom_size.loc[chrom].values[0] / resolution)),
                    method,
                    num_f,
                    diag
                ),
                resolution,
                dist
            )
            for chrom in chrom_size.index
        ]

        Sr_tilde = pool.starmap(S_construction, arguments)

        pool.close()
        pool.join()

    S_tilde = pd.DataFrame(
        np.hstack(Sr_tilde),
        index=maps
    )

    logger.info(
        'Feature matrix construction completed: {} cells x {} features.'.format(
            S_tilde.shape[0],
            S_tilde.shape[1]
        )
    )

    return S_tilde


def integrate_feature_matrix(out_path, method, num_f, k, diag=False, sigma=15):
    """
    Perform dataset/batch correction on the previously constructed feature
    matrix and save the integrated representation.
    """

    logger.info('Begin data integration.')

    if diag:
        feature_file = (
            out_path +
            '/S_tilde_' + method +
            '_diag_f' + str(num_f) +
            '.txt'
        )

        output_file = (
            out_path +
            '/S_hat_' + method +
            '_diag_f' + str(num_f) +
            '_k' + str(k) +
            '.txt'
        )

    else:
        feature_file = (
            out_path +
            '/S_tilde_' + method +
            '_f' + str(num_f) +
            '.txt'
        )

        output_file = (
            out_path +
            '/S_hat_' + method +
            '_f' + str(num_f) +
            '_k' + str(k) +
            '.txt'
        )

    logger.info('Reading feature matrix: {}'.format(feature_file))

    S_tilde = pd.read_csv(
        feature_file,
        index_col=0,
        header=None
    )

    logger.info(
        'Loaded feature matrix with {} cells and {} features.'.format(
            S_tilde.shape[0],
            S_tilde.shape[1]
        )
    )

    meta = pd.DataFrame(index=S_tilde.index)

    meta['dataset/batch'] = [
        idx.split('/')[-2]
        for idx in meta.index
    ]

    logger.info(
        'Detected {} dataset(s)/batch(es).'.format(
            meta['dataset/batch'].nunique()
        )
    )

    S_hat = correction(
        S_tilde,
        k,
        meta,
        sigma=sigma
    )

    logger.info('Saving integrated feature matrix: {}'.format(output_file))

    S_hat.to_csv(
        output_file,
        index=True,
        header=False
    )

    logger.info('Data integration completed.')

    return S_hat


def construct_latent_embedding(out_path, method, num_f, k, pc, diag=False):
    """
    Construct a latent embedding from the integrated feature matrix using PCA.
    """

    logger.info('Begin data latent embedding.')

    if diag:
        integrated_file = (
            out_path +
            '/S_hat_' + method +
            '_diag_f' + str(num_f) +
            '_k' + str(k) +
            '.txt'
        )

        latent_file = (
            out_path +
            '/Latent_' + method +
            '_diag_f' + str(num_f) +
            '_k' + str(k) +
            '_pc' + str(pc) +
            '.txt'
        )

    else:
        integrated_file = (
            out_path +
            '/S_hat_' + method +
            '_f' + str(num_f) +
            '_k' + str(k) +
            '.txt'
        )

        latent_file = (
            out_path +
            '/Latent_' + method +
            '_f' + str(num_f) +
            '_k' + str(k) +
            '_pc' + str(pc) +
            '.txt'
        )

    logger.info(
        'Reading integrated feature matrix: {}'.format(integrated_file)
    )

    S_hat = pd.read_csv(
        integrated_file,
        index_col=0,
        header=None
    )

    logger.info(
        'Loaded integrated matrix with {} cells and {} features.'.format(
            S_hat.shape[0],
            S_hat.shape[1]
        )
    )

    logger.info(
        'Running PCA with {} principal components.'.format(pc)
    )

    latent = PCA(
        n_components=pc,
        random_state=0
    ).fit_transform(S_hat)

    latent = pd.DataFrame(
        latent,
        index=S_hat.index
    )

    logger.info(
        'Latent embedding generated: {} cells x {} components.'.format(
            latent.shape[0],
            latent.shape[1]
        )
    )

    logger.info(
        'Saving latent embedding: {}'.format(latent_file)
    )

    latent.to_csv(
        latent_file,
        index=True,
        header=False
    )

    logger.info('Data latent embedding completed.')

    return latent
