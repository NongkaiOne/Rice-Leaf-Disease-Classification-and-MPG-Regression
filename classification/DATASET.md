# Dataset — 5 retained classes

Kept **16,533 image files**, 6.19 GB of file content (including duplicate copies); original train/test folders remain. Only Leaf Blast, Bacterial Leaf Blight, Sheath Blight, Brown Spot and Healthy remain. Leaf Blast maps to rice_blast; spaces/case in Sheath Blight map to sheath_blight. All other class folders and the unused Rice_Leaf_AUG were deleted at the user's request. Historical Git commits/LFS objects are not rewritten.

After audit, active manifest uses fit 8,186 / validation 1,368 / test 2,228. The 329 quarantine records are excluded from training. Pixel hashes and inclusion reasons: data_audit/dataset_manifest.csv. Frozen group-aware split: experiments/cnn5/manifest.csv (StratifiedGroupKFold 7 folds, seed 42). Similarity groups are proxies, not verified leaf/plant IDs; test was used in earlier studies and is not a new holdout.

## Sources and redistribution notices

- [Rice Leaf Diseases Detection — loki4514](https://www.kaggle.com/datasets/loki4514/rice-leaf-diseases-detection): Kaggle metadata checked 2026-09-30 declared Apache 2.0. Standard text retained in DATASET-LICENSE-APACHE-2.0.txt. No upstream NOTICE was found in the supplied folders; the project does not claim authorship of these images.
- [Rice Leaf Disease: An Images Dataset — alamshihab075](https://www.kaggle.com/datasets/alamshihab075/rice-leaf-disease-an-images-dataset): user identified this as the source of added Healthy/Leaf Blast/Sheath Blight images. Metadata checked 2026-09-30 declared MIT; record in data_audit/more_source_metadata.json. This source is separate from Apache 2.0. Individual upstream image ownership has not been independently established; an upstream MIT copyright notice was not supplied and must not be fabricated or replaced with the Apache notice.
- data_audit/more_import_manifest.csv retains source/destination paths and byte/pixel hashes for the imported images. Labels are derived from the source class folders. Some augmented images lack original-image IDs, which limits leakage guarantees.

## Download

Images are tracked with Git LFS. Run git lfs install, clone the public repository, then git lfs pull. A GitHub page showing a pointer is not a downloaded image; real objects must be retrievable. Current access checks and any outstanding checks are recorded in PURPOSE_CHECKLIST.md/STATUS.md.

Inventory: data_audit/retained_inventory.json. Removal log: data_audit/cleanup_completed.json. Models and app do not load unused class data.
