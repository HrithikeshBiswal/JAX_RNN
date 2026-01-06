# JAX_RNN
Built a hyper optimised JAX inference engine
## Hybrid Architecture Workflow

This project utilizes a **Hybrid Research-Production Pipeline**:

1. Training (PyTorch): Models are trained in PyTorch for flexibility and ease of debugging.
   We use a "Clone Factory" strategy (`src/train.py`) to train 40 independent instances with bootstrapped data subsets.
    
2.  Serialization (Bridge):
    Weights are extracted from PyTorch checkpoints and stacked into NumPy/JAX arrays.
    Result: A single `.npz` file containing the weights for the entire ensemble (Shape: `[40, Hidden_Dim, ...]`).

3.  Inference (JAX/Flax):
   The inference engine (`src/inference.py`) loads the stacked weights.
   `jax.vmap` is used to broadcast the forward pass across the ensemble dimension, executing all 40 models in a single compiled call.
