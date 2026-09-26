# Create a private Hugging Face model repo for checkpoints.

# After: huggingface-cli login
bash scripts/create_hf_repo.sh

# Override name if needed:
# HF_CHECKPOINT_REPO=your-user/notre-checkpoints bash scripts/create_hf_repo.sh

# Keep the repo private. Gate B ended the project on 2026-09-25.
# There is no public weight release.
