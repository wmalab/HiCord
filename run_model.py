#!/usr/bin/env python

"""
Run the model pipeline.

Pipeline:
    1. Read chromosome sizes
    2. Construct chromosome-level feature matrices
    3. Integrate/correct datasets or batches
    4. Construct the final latent embedding using PCA
"""

import os
import argparse
import logging

from utils import *
from model import *


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger()

logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_args():
    """
    Parse command-line arguments for the model pipeline.
    """

    parser = argparse.ArgumentParser(
        description='Construct and integrate contact-map features and '
                    'generate a latent embedding.'
    )

    # Input/output
    parser.add_argument(
        '--input',
        required=True,
        help='Input directory containing dataset directories and contact maps.'
    )

    parser.add_argument(
        '--out',
        required=True,
        help='Output directory.'
    )

    parser.add_argument(
        '--chrom',
        required=True,
        help='Chromosome-size file or space-separated chromosome-size files.'
    )

    # Contact-map parameters
    parser.add_argument(
        '--resolution',
        type=int,
        required=True,
        help='Contact-map bin size in base pairs.'
    )

    parser.add_argument(
        '--dist',
        type=int,
        default = 20000000,
        help='Maximum genomic interaction distance in base pairs.'
    )

    parser.add_argument(
        '--diag',
        action='store_true',
        help='Include diagonal contacts.'
    )

    # Feature construction
    parser.add_argument(
        '--method',
        choices=['svd', 'nmf'],
        default='svd',
        help='Feature reduction method. Default: svd.'
    )

    parser.add_argument(
        '--f',
        type=int,
        default=50,
        help='Number of features/components per chromosome.'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Process chromosomes in parallel.'
    )

    # Integration
    parser.add_argument(
        '--k',
        type=int,
        default=30,
        help='Number of nearest neighbors used for MNN correction.'
    )

    parser.add_argument(
        '--sigma',
        type=float,
        default=15,
        help='Gaussian kernel bandwidth used for batch correction.'
    )

    # Final PCA embedding
    parser.add_argument(
        '--pc',
        type=int,
        default=100,
        help='Number of principal components in the final latent embedding.'
    )

    # Pipeline controls
    parser.add_argument(
        '--feature_matrix',
        action='store_true',
        help='Skip feature-matrix construction and use an existing matrix.'
    )

    parser.add_argument(
        '--integrated',
        action='store_true',
        help='Skip integration and use an existing integrated matrix.'
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    logger.info('Starting model pipeline.')

    # Create output directory if needed.
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # -------------------------------------------------------------
    # Read chromosome sizes
    # -------------------------------------------------------------

    logger.info('Reading chromosome sizes.')

    chrom_size = read_chrom_size(args.chrom)

    logger.info(
        'Loaded chromosome information for {} chromosomes.'.format(
            len(chrom_size)
        )
    )

    # -------------------------------------------------------------
    # Step 1: Construct feature matrix
    # -------------------------------------------------------------

    if not args.feature_matrix:

        logger.info('Step 1: Constructing feature matrix.')

        S_tilde = construct_feature_matrix(
            input_path=args.input,
            chrom_size=chrom_size,
            resolution=args.resolution,
            dist=args.dist,
            method=args.method,
            num_f=args.f,
            diag=args.diag,
            parallel=args.parallel
        )

        # Save the feature matrix.
        if args.diag:
            feature_file = (
                args.out +
                '/S_tilde_' + args.method +
                '_diag_f' + str(args.f) +
                '.txt'
            )
        else:
            feature_file = (
                args.out +
                '/S_tilde_' + args.method +
                '_f' + str(args.f) +
                '.txt'
            )

        logger.info(
            'Saving feature matrix: {}'.format(feature_file)
        )

        S_tilde.to_csv(
            feature_file,
            index=True,
            header=False
        )

    else:
        logger.info(
            'Skipping feature-matrix construction; using existing file.'
        )

    # -------------------------------------------------------------
    # Step 2: Dataset/batch integration
    # -------------------------------------------------------------

    if not args.integrated:

        logger.info('Step 2: Performing dataset/batch integration.')

        S_hat = integrate_feature_matrix(
            out_path=args.out,
            method=args.method,
            num_f=args.f,
            k=args.k,
            diag=args.diag,
            sigma=args.sigma
        )

    else:
        logger.info(
            'Skipping integration; using existing integrated matrix.'
        )

    # -------------------------------------------------------------
    # Step 3: Construct final latent embedding
    # -------------------------------------------------------------

    logger.info('Step 3: Constructing latent embedding.')

    latent = construct_latent_embedding(
        out_path=args.out,
        method=args.method,
        num_f=args.f,
        k=args.k,
        pc=args.pc,
        diag=args.diag
    )

    logger.info(
        'Pipeline completed successfully. '
        'Final latent matrix: {} cells x {} components.'.format(
            latent.shape[0],
            latent.shape[1]
        )
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == '__main__':
    main()
