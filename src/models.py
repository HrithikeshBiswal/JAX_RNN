import jax
import jax.numpy as jnp
import optax
import numpy as np
import os
import joblib
from functools import partial
from models import predict_cond, predict_film, predict_hyper

# --- CONFIGURATION ---
SEED = 42
BATCH_SIZE = 512
EPOCHS = 5
LR = 0.0005
N_MODELS = 40  # The "Clone Army" size
SEQ_LEN = 20
FEAT_DIM = 32
SIG_DIM = 64

# --- UTILS ---
def create_train_state(rng, learning_rate):
    """
    Initializes the optimizer state for the entire ensemble.
    """
    optimizer = optax.adam(learning_rate)
    return optimizer

def init_ensemble_weights(rng, num_models):
    """
    Initializes diverse weights for 40 models at once.
    """
    keys = jax.random.split(rng, num_models)
    
    # Helper to generate random params for one model
    def init_single(key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        return {
            # GRU Weights (Input-Hidden, Hidden-Hidden)
            'gru_weight_ih_l0': jax.random.normal(k1, (3 * 96, 32 + 64)) * 0.01, # +64 for Context
            'gru_weight_hh_l0': jax.random.orthogonal(k2, (3 * 96, 96)),
            'gru_bias_ih_l0': jnp.zeros((3 * 96,)),
            'gru_bias_hh_l0': jnp.zeros((3 * 96,)),
            
            # Layer 2
            'gru_weight_ih_l1': jax.random.normal(k3, (3 * 96, 96)) * 0.01,
            'gru_weight_hh_l1': jax.random.orthogonal(k4, (3 * 96, 96)),
            'gru_bias_ih_l1': jnp.zeros((3 * 96,)),
            'gru_bias_hh_l1': jnp.zeros((3 * 96,)),
            
            # FC Head
            'fc_weight': jax.random.normal(k5, (32, 96)) * 0.01,
            'fc_bias': jnp.zeros((32,)),
            
            # FiLM / Hyper params would go here similarly...
        }

    # VMAP: Run initialization 40 times in parallel
    print(f"Initializing {num_models} models in parallel...")
    ensemble_params = jax.vmap(init_single)(keys)
    return ensemble_params

# --- TRAINING STEP (VECTORIZED) ---
@partial(jax.jit, static_argnames=['optimizer'])
def train_step(params, opt_state, x_seq, x_ctx, x_sig, y_true, optimizer):
    """
    The Core Magic: Trains 40 models simultaneously on the same batch.
    """
    
    def loss_fn(p, x_s, x_c, x_si, y):
    
        pred = predict_cond(x_s, x_c, x_si, p, rnn_lags=SEQ_LEN)
        return jnp.mean((pred - y) ** 2)


    grad_fn = jax.vmap(jax.value_and_grad(loss_fn), in_axes=(0, None, None, None, None))
    
    losses, grads = grad_fn(params, x_seq, x_ctx, x_sig, y_true)
    
    # Update optimizer
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, jnp.mean(losses)

# --- MAIN LOOP ---
def main():
    print(f"--- JAX Clone Army Training (Models: {N_MODELS}) ---")
    
    # 1. Mock Data Generation (For Repo Demo)
    
    print("Generating synthetic HFT data batches...")
    rng = jax.random.PRNGKey(SEED)
    rng, d_key = jax.random.split(rng)
    
    dummy_seq = jax.random.normal(d_key, (BATCH_SIZE, SEQ_LEN, FEAT_DIM))
    dummy_ctx = jax.random.normal(d_key, (BATCH_SIZE, FEAT_DIM))
    dummy_sig = jax.random.normal(d_key, (BATCH_SIZE, SIG_DIM))
    dummy_y = jax.random.normal(d_key, (BATCH_SIZE, FEAT_DIM))
    
    # 2. Initialize
    optimizer = create_train_state(rng, LR)
    params = init_ensemble_weights(rng, N_MODELS)
    opt_state = optimizer.init(params)
    
    print(f"Ensemble Param Shape (Input Layer): {params['gru_weight_ih_l0'].shape}")
    print("Notice (40, ...) -> This is the vectorized dimension.\n")

    # 3. Training Loop
    for epoch in range(EPOCHS):
        rng, step_key = jax.random.split(rng)
        
        # In a real loop iterate over DataLoader here
        params, opt_state, loss = train_step(
            params, opt_state, dummy_seq, dummy_ctx, dummy_sig, dummy_y, optimizer
        )
        
        print(f"Epoch {epoch+1} | Ensemble MSE Loss: {loss:.6f}")

    # 4. Save Weights
    print("\nSaving Ensemble Weights...")
    if not os.path.exists('weights'): os.makedirs('weights')
    np.savez_compressed('weights/jax_weights_Cond_CLONES.npz', **params)
    print("Saved to weights/jax_weights_Cond_CLONES.npz")

if __name__ == "__main__":
    main()
