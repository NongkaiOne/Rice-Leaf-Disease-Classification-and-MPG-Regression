# Deploy Classification on Render

Use the repository's root `render.yaml` Blueprint. It creates a free Docker web service with `classification/` as its root directory and `/` as its health check.

[Deploy to Render](https://render.com/deploy?repo=https://github.com/NongkaiOne/Rice-Leaf-Disease-Classification-and-MPG-Regression)

Sign in to Render, open the link, review the service, and deploy. The final public URL is assigned by Render; add the verified URL to both README files after deployment succeeds.

For manual creation: select Web Service, this repository, branch `main`, root directory `classification`, Docker runtime, Dockerfile `./Dockerfile`, context `.`, and the Free plan. Set the environment variables from `render.yaml`.

The app reads Render's `PORT` and binds to `0.0.0.0`. It loads the saved model and bundled example images; training images are not required to serve predictions. The Docker context excludes the training dataset. No training runs during deployment.

Free services sleep after inactivity and may take time to start again. For the instructor's review period, allow startup time or choose an always-on paid plan yourself if needed.

After the deployment becomes live, open the public URL without signing in and test image classification plus class comparison. A successful build alone does not verify predictions.

References: [Render Blueprints](https://render.com/docs/blueprint-spec), [Free services](https://render.com/docs/free).
