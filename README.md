# 🧬 GenoTB-Care

## WGS-Based Tuberculosis Drug-Resistance Prediction

GenoTB-Care is a research prototype for analyzing *Mycobacterium tuberculosis* whole-genome sequencing (WGS) data and demonstrating genomic drug-resistance prediction using machine learning.

The system combines TB-Profiler for genomic variant and resistance interpretation with a custom feature-engineering and machine-learning layer.

\---

## 🚀 Project Pipeline

```text
TB WGS FASTQ
     │
     ▼
┌───────────────┐
│  TB-Profiler  │
└───────┬───────┘
        │
        ▼
Genomic Variants
        │
        ▼
Resistance-associated Mutations
        │
        ▼
┌─────────────────────┐
│ Feature Engineering │
└──────────┬──────────┘
           │
           ▼
     Feature Table
           │
           ▼
    Random Forest ML
           │
           ▼
 Resistance Prediction
           │
           ▼
 Streamlit Dashboard 

\## Data



Public Mycobacterium tuberculosis WGS datasets were used for

prototype development.



The raw FASTQ files are not included in this repository because

of their large size.



Users can obtain the corresponding public sequencing data using

the accession IDs documented in this project.

