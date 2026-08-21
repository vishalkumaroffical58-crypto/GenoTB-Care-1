# 🧬 GenoTB-Care

## WGS-Based Tuberculosis Drug-Resistance Prediction

GenoTB-Care is a research prototype for analyzing *Mycobacterium tuberculosis* whole-genome sequencing (WGS) data and demonstrating genomic drug-resistance prediction using machine learning.

The system combines TB-Profiler for genomic variant and resistance interpretation with a custom feature-engineering and machine-learning layer.

---

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
