# GGPONC - The German Clinical Guideline Corpus for Oncology

![GGPONC Annotations in INCepTION](assets/annotation.png)

This repository collects resources related to [GGPONC](https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-english/).

It covers:
- [(Nested) Clinical Named Entity Recogition](#clinical-named-entity-recognition)
- [UMLS Entity Linking with xMEN](#umls-entity-linking-with-xmen)
- [Resolution of Coordination Ellipses](#resolution-of-coordination-ellipses)
- [Molecular Named Entities (Genes / Proteins, Variants)](#molecular-named-entities)

see also:

| Repository | Description |
| ---- | ---- |
| [ggponc_annotation](https://github.com/hpi-dhc/ggponc_annotation) | GGPONC 2.0 Results and Gold Standard Annotations |
| [ggponc_preprocessing](https://github.com/hpi-dhc/ggponc_preprocessing) | Pre-Processing Pipeline (Tokenization, POS Tagging) and GGPONC 1.0 Results |
| [ggponc_ellipses](https://github.com/hpi-dhc/ggponc_ellipses) | Resolving Elliptical Compounds in German Medical Text |
| [ggponc_molecular](https://github.com/hpi-dhc/ggponc_molecular) | GGTWEAK - Gene Tagging with Weak Supervision for German Clinical Text |

## Preparation

1. Get access to GGPONC following the instructions on the [project homepage](https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-english/) and place the contents of the 2.0 release (`v2.0_2022_03_24` and `v2.0_agreement`) in the `data` folder
2. Install Python dependencies `pip install -r requirements.txt` `

## Clinical Named Entity Recognition

### Data Loading

A BigBIO-compatible data loader for loading the latest gold-standard annotations (GGPONC 2.0) to train NER models are available through the Hugging Face Hub: https://huggingface.co/datasets/bigbio/ggponc2

```python

from datasets import load_dataset
dataset = load_dataset('bigbio/ggponc2', data_dir='data/v2.0_2022_03_24', name='ggponc2_fine_long_bigbio_kb')
```

### Nested NER with spaCy Spancat

A trained spaCy model for nested NER is available on Hugging Face: https://huggingface.co/phlobo/de_ggponc_medbertde

```bash
huggingface-cli download phlobo/de_ggponc_medbertde de_ggponc_medbertde-any-py3-none-any.whl --local-dir .
pip install -q de_ggponc_medbertde-any-py3-none-any.whl
```

See: [01_GGPONC_Nested_NER](01_GGPONC_Nested_NER.ipynb)

### Flat NER

Training and evaluation of the (flat) NER models described in [Borchert et al. (2022)](https://aclanthology.org/2022.lrec-1.389/) is covered in the [GGPONC 2.0 repository](https://github.com/hpi-dhc/ggponc_annotation/blob/master/notebooks/02_NER_Baselines.ipynb).

## UMLS Entity Linking with xMEN

We use the [xMEN](https://github.com/hpi-dhc/xmen/) toolkit with a pre-trained re-ranker to normalized identified entity mentions to UMLS codes.

See: [02_GGPONC_UMLS_Linking](02_GGPONC_UMLS_Linking.ipynb)

## Resolution of Coordination Ellipses

Application of our encoder-decoder model for resolving elliptical coordinated compound nound phrases (ECCNPs), e.g. `Chemo- und Strahlentherapie` -> `Chemotherapie und Strahlentherapie`

To load the model, put the contents of `ellipses_2023_01_30` from the GGPONC releases into the data folder.

See: [03_ECCNP_Analysis.ipynb](03_ECCNP_Analysis.ipynb)

## Molecular Named Entities

Training and evaluation of a nested NER model for gene / protein and variant mentions. The dataset (`molecular_2024_04_03`) is not yet published, but available upon request. Place the release in `data` to run the notebook.

See: [04_Molecular.ipynb](04_Molecular.ipynb)
