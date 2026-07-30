# Results

Local result ZIP packages are stored under `zip_packages/` and `packages/`.

These archives are ignored by Git by default because they are generated artifacts. If a result file is required for reproducibility, prefer one of these options:

- Add a small CSV/JSON summary instead of the full archive.
- Publish the archive as a GitHub Release asset.
- Store large artifacts in Drive, Hugging Face Hub, or another external artifact store and link them from the README.

## External Artifact Not Copied

- `C:\Users\Surya Narayanaa\Downloads\labse_finetuning_important_results.zip` - approximately 7 GB. Keep this outside Git and publish through an artifact store if needed.
