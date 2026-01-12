# Debrecen Diabetic Retinopathy

Benchmark dataset used by the BioWorld OS POC.

- **Source:** UCI Machine Learning Repository — "Diabetic Retinopathy Debrecen
  Data Set"
- **File:** `messidor_features.arff` (ARFF format)
- **Rows:** 1,151
- **Features:** 19 (Messidor image features derived from fundus images)
- **Target:** `Class` — `0`/`1`, presence of signs of diabetic retinopathy
  (`1` = signs present). Stored as byte strings in the ARFF file; the loader
  (`bioworld.mltasks.load_dataframe`) decodes them.

## Get the data

The ARFF is bundled in this repo for a self-contained demo. The original can
also be downloaded from UCI:

https://archive.ics.uci.edu/dataset/3/diabetic+retinopathy+debrecen
