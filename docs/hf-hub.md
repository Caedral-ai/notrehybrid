# Create a private Hugging Face model repo for checkpoints.

# After: huggingface-cli login
bash scripts/create_hf_repo.sh

# Override name if needed:
# HF_CHECKPOINT_REPO=your-user/notre-checkpoints bash scripts/create_hf_repo.sh

# Public weights only if Gate C passes. Until then keep the repo private.
