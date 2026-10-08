# Classification on Render

Public URL: https://rice-leaf-classification.onrender.com/

The updated app contains DenseNet121 and SVM, five classes, model selection and an Evaluation comparison. See the latest observed state below and verify predictions after deployment.

Use the repository's root render.yaml, or Web Service settings: branch main, root directory classification, Dockerfile ./Dockerfile, Docker context ., health check /. The Dockerfile installs requirements.txt and copies app.py plus rice_leaf_app. No training runs during deployment. GIT_LFS_SKIP_SMUDGE=1 avoids downloading training images for the web service.

Required runtime assets: models/rice_cnn.onnx + rice_cnn.json + rice_svm.joblib, artifacts_cnn5/, examples_cnn5/, the inference modules and app.py. The CNN runs on CPU through ONNX Runtime. The dataset and PyTorch training checkpoint are for reproducibility, not web startup.

After a successful deployment, open the public URL without login, check that the model selector offers CNN (DenseNet121) and SVM, upload an example for each model, and confirm Evaluation has five classes. Test predictions, not only HTTP health. A sleeping free service may need time to start; availability during grading still requires an actual external check.

## Latest observed state — 2026-10-09

After publishing code commit c35294b1 to main, the public /config endpoint returned HTTP 200 but still showed the previous 10-class SVM interface. GitHub already contains the five-class CNN/SVM version. A new deployment is still needed or has not become live yet. In Render, select the Classification service and deploy the latest commit of main (Manual Deploy → Deploy latest commit if automatic deployment is disabled), then run `python scripts/smoke_cnn_api.py https://rice-leaf-classification.onrender.com/` to verify both models publicly. This task does not claim that the new CNN is already live.
