# PRISM: Patient-specific Representation Integration with a Shared Graph Model for Cancer Driver Gene Prioritization

PRISM is a graph neural network framework for cancer driver gene prioritization that integrates a shared protein–protein interaction (PPI) network with patient-specific gene interaction graphs derived from single-cell RNA sequencing (scRNA-seq) data. By combining auxiliary multi-cancer pretraining with target-specific fine-tuning, PRISM learns transferable biological representations that improve the prioritization of cancer driver genes across multiple cancer types.

---

## Model Framework

![PRISM framework](PRISM_architecture.png)

PRISM employs a shared graph encoder to jointly learn representations from a reference CPDB protein–protein interaction network and patient-specific gene interaction graphs constructed from scRNA-seq data. The encoder is pretrained on multiple non-target cancer types to learn transferable biological knowledge and is subsequently fine-tuned on the target cancer for driver gene prioritization.

## Requirements

- Python 3.9+
- PyTorch
- PyTorch Geometric
- NumPy
- pandas
- scikit-learn

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Data

The experiments use publicly available datasets from the following resources:

- **ConsensusPathDB (CPDB):** reference protein–protein interaction network.
- **The Cancer Genome Atlas (TCGA):** multi-omics features and cancer driver gene labels.
- **Gene Expression Omnibus (GEO):** public scRNA-seq datasets used to construct patient-specific gene interaction graphs with CellPhoneDB.

All datasets are organized under the `Data/` directory.

```text
Data/
├── PPI/
├── features/
├── labels/
└── scRNA/
```

### Data Availability

Due to the large size of the patient-specific scRNA-seq graph data, this subset is hosted externally and provided as an anonymous download link for the review process:

- **scRNA patient graphs:** [Anonymous Google Drive link](https://drive.google.com/drive/folders/1p3dvOkieMnGmfts1b78CY4WJ9Y8gxXL_?usp=sharing)

Please place the downloaded contents into `Data/scRNA/` before running the pipeline.

---

## Repository Structure

```text
ICTA2026/
├── README.md
├── requirements.txt
├── pretrain.py
├── train.py
├── model.py
├── utils.py
└── Data/
    ├── PPI/
    ├── features/
    ├── labels/
    └── scRNA/
```

---

## Usage

Install the required packages:

```bash
pip install -r requirements.txt
```

Run auxiliary pretraining:

```bash
python pretrain.py
```

Run target-specific fine-tuning:

```bash
python train.py
```

---

## License

This repository is intended for academic research purposes.
