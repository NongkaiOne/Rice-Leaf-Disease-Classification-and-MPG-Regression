# Dataset provenance

- Dataset: **Rice Leaf Diseases Detection**
- Publisher/account: **loki4514** on Kaggle (do not infer the person's real name)
- Source: https://www.kaggle.com/datasets/loki4514/rice-leaf-diseases-detection
- Kaggle public metadata API: https://www.kaggle.com/api/v1/datasets/list?search=rice%20leaf%20diseases%20detection
- Metadata checked: **2026-09-30**; matching `ref` is `loki4514/rice-leaf-diseases-detection`; `licenseName` is **Apache 2.0**.
- License text: https://www.apache.org/licenses/LICENSE-2.0 (standard text included as `DATASET-LICENSE-APACHE-2.0.txt`)
- The local dataset was supplied by the user; this project does not claim authorship of the images.
- Keep attribution, the Apache 2.0 license, and any upstream copyright/NOTICE files with redistributed data. No upstream NOTICE was found in the supplied folders. The Kaggle license declaration is recorded here; individual original image provenance has not been independently verified.

## Data used

`Rice_Leaf_Diease/Rice_Leaf_Diease/train/<class>/*` and `test/<class>/*` now contain **25,445 images** before filtering: **20,462 train** and **4,983 test**. Counts must be read from `rice_leaf_app/artifacts/class_counts.csv` after running the audit; that generated report is authoritative if the files change.

Labels come from class folder names. Spaces are normalized to underscores and names to lowercase, mapping `Neck_Blast` to `neck_blast`, `Rice Hispa` to `rice_hispa`, `Sheath Blight` to `sheath_blight`, and `Tungro` to `tungro`. All 10 classes are explicitly defined in `rice_leaf_app/config.py`. Neck Blast includes the panicle/neck class, so the project is not strictly restricted to leaf-only imagery.

The original train/test folders are preserved. The audit excludes unreadable images, identical decoded RGB pixels, and identical pixels with conflicting labels. When an exact duplicate appears in both splits, its training copy is excluded; original files are never deleted. `dataset_manifest.csv` records every scanned image, label, pixel hash, inclusion status, and reason. `split_manifest.csv` records the deterministic development split (80% fit / 20% validation of eligible train; seed 42).

The separate `Rice_Leaf_AUG` folder has **11,790 images and 9 classes**, with no Tungro folder. It is not used or merged into training. Source-image IDs are unavailable, so exact deduplication cannot guarantee the absence of near duplicates, augmented relatives, or the same plant across splits. This limits interpretation of the test score. A future field-level evaluation should split by original plant/field before augmentation.

## Additional user-supplied images from more

Imported **7,040 source files**, retaining **7,000 distinct byte-content copies** in the existing class folders. There are **2,684 new decoded-pixel groups** beyond the original dataset. New groups are split approximately 80/20 per class with seed 42. Exact matches retain the existing split (test takes precedence if the old dataset already contained a cross-split duplicate). No existing dataset file was overwritten. Source/destination paths, labels, split assignments, and byte/pixel checksums are recorded in `rice_leaf_app/artifacts/more_import_manifest.csv`.

The user supplied these additions in Healthy, Leaf Blast and Sheath Blight folders. The user identified [Rice Leaf Disease: An Images Dataset — alamshihab075, Kaggle](https://www.kaggle.com/datasets/alamshihab075/rice-leaf-disease-an-images-dataset) as their source. Kaggle public metadata was checked on 2026-09-30 and declares **MIT** for `alamshihab075/rice-leaf-disease-an-images-dataset`. This is separate from the original dataset's Apache 2.0 license; `DATASET-LICENSE-APACHE-2.0.txt` applies only to the original source. Preserve the upstream MIT license and copyright notice when redistributing the additional images; do not replace them with the Apache text. Images named `aug_*` still lack reliable original-image IDs, so near-duplicate/augmentation leakage remains a limitation.

The source folder `more` is **deleted after successful training and checksum verification**. Test composition expanded, so new and previous aggregate scores are not directly comparable.

## Publishing the dataset

The primary dataset is retained locally and is not ignored by `.gitignore`. It has **not yet been uploaded to GitHub**. Check repository storage limits before uploading the full dataset; use Git LFS if required and verify that the teacher can download the actual image objects, not only pointer files. Include `DATASET.md` and the upstream license alongside the data. The unused AUG data and unrelated digit example are excluded from the submission by `.gitignore`.
