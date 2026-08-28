# HiCord: scHi-C Datasets/Batches Integration Using Mutual Neareast Neighbors

Code accompanying the manuscript:

**HiCord: scHi-C Datasets/Batches Integration Using Mutual Neareast Neighbors**  
Rui Ma, Huiling Liu, Wenxiu Ma  
[Journal / Preprint, Year]

This repository contains the Python implementation used to construct low-dimensional representations from **binned short-format chromatin contact maps**, perform dataset/batch integration, and generate the final latent embedding used in the analysis.

## Overview

The analysis consists of three main steps:

1. **Feature matrix construction**  
   Binned short-format contact maps are processed chromosome by chromosome. Contact values within the specified genomic distance are extracted and transformed, followed by dimensionality reduction using SVD or NMF.

2. **Dataset/batch integration**  
   Low-dimensional representations from multiple datasets or batches are integrated using mutual nearest neighbors (MNNs). The dataset containing the largest number of cells is used as the reference dataset.

3. **Latent embedding**  
   PCA is applied to the integrated feature matrix to generate the final latent representation.


### `run_model.py`

Main command-line entry point for running the analysis pipeline.

### `model.py`

Contains the major steps of the analysis pipeline:

- `construct_feature_matrix()` — constructs chromosome-specific features and combines them into the complete feature matrix.
- `integrate_feature_matrix()` — performs dataset/batch integration.
- `construct_latent_embedding()` — generates the final PCA latent embedding.

### `utils.py`

Contains supporting functions, including:

- `S_construction()` — constructs chromosome-level features from binned short-format contact maps.
- `correction()` — performs MNN-based dataset/batch correction.
- `read_chrom_size()` — reads and combines chromosome-size information.

## Requirements

The code was developed using **Python 3.6.4**.

The primary Python dependencies are:

```text
numpy==1.19.5
pandas==1.1.5
scipy==1.5.4
scikit-learn==0.24.2
umap-learn==0.5.2
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

A Conda environment is recommended when reproducing the original analysis.

## Input contact-map format

The model takes **binned short-format chromatin contact maps** as input.

Each row represents the interaction between two genomic bins:

```text
chromosome_1    position_1    chromosome_2    position_2    contact_value
```

For example:

```text
chr1    10000    chr1    20000    15
chr1    10000    chr1    30000    8
chr1    20000    chr1    30000    12
```

Input contact maps may be tab-separated or whitespace-separated.

The genomic coordinates should correspond to the specified contact-map resolution.

## Input directory structure

Contact maps from different datasets or batches should be placed in separate subdirectories:

```text
input/
├── dataset_1/
│   ├── cell_001.txt
│   ├── cell_002.txt
│   └── cell_003.txt
│
└── dataset_2/
    ├── cell_004.txt
    ├── cell_005.txt
    └── cell_006.txt
```

The parent directory name is used as the `dataset/batch` identifier during integration. For example, the files under `dataset_1/` are treated as belonging to one dataset/batch.

Hidden files beginning with `.` are ignored when collecting contact maps.

## Chromosome-size file

A chromosome-size file must also be provided.

Example:

```text
chr1    248956422
chr2    242193529
chr3    198295559
```

If multiple chromosome-size files are supplied, the maximum chromosome size across the supplied files is used for each chromosome.

## Running the model

The complete pipeline is run through `run_model.py`.

A general example is:

```bash
python run_model.py \
    --input test_data/input \
    --chrom test_data/chrom_sizes.txt \
    --out results \
    --resolution 1000000 \
    --dist 20000000 \
    --method svd \
    --f 50 \
    --k 30 \
    --sigma 15 \
    --pc 100
```

Adjust the parameters according to the resolution and analysis settings used for the dataset.

## Major parameters

`--input`  
Directory containing dataset/batch subdirectories and binned short-format contact maps.

`--chrom`  
Chromosome-size file. Multiple chromosome-size files may also be supplied.

`--out`  
Directory in which intermediate and final results are written.

`--resolution`  
Contact-map bin size in base pairs. For example, use `10000` for 10-kb contact maps.

`--dist`  
Maximum genomic distance, in base pairs, between interacting bins included during feature construction.

`--method`  
Dimensionality-reduction method used during chromosome-level feature construction. Supported options are:

```text
svd
nmf
```

`--f`  
Number of latent features/components generated for each chromosome.

`--k`  
Number of nearest neighbors used to identify mutual nearest neighbors during dataset/batch integration.

`--sigma`  
Bandwidth of the Gaussian kernel used to weight MNN-based correction vectors. The default value is `15`.

`--pc`  
Number of principal components retained in the final latent embedding.

`--diag`  
Include diagonal contacts during feature construction.

`--parallel`  
Process chromosomes in parallel.

## Output

The pipeline generates three main output files:

* **`S_tilde_*.txt`** — chromosome-level feature matrix.
* **`S_hat_*.txt`** — integrated/batch-corrected feature matrix.
* **`Latent_*.txt`** — final PCA latent embedding.

Output filenames include the analysis parameters, such as the feature construction method, number of features, MNN `k`, number of principal components, and whether diagonal contacts were included.

All output files are saved in the directory specified by `--out`.

## Reproducibility

For reproducibility, users should record:

- Python and package versions
- contact-map resolution
- maximum genomic interaction distance
- SVD/NMF method
- number of chromosome-level features
- MNN `k`
- Gaussian kernel `sigma`
- number of final PCA components
- whether diagonal contacts were included

The package versions used in the original analysis are provided in `requirements.txt`.


## Citation

If you use this code, please cite:

> Rui Ma, Huiling Liu, Wenxiu Ma. “HiCord: scHi-C Datasets/Batches Integration Using Mutual Neareast Neighbors.” [Journal], [Year]. [DOI]

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.