import jax
import jax.numpy as jnp

# ==========================================
# PART 1: THE RAW GRU CELL (Matrix Math)
# ==========================================
def gru_cell(x, h, w_ih, b_ih, w_hh, b_hh):
    """
    Manual implementation of a GRU cell.
    No hidden libraries, just pure linear algebra.
    """
    gi = jnp.dot(w_ih, x) + b_ih
    gh = jnp.dot(w_hh, h) + b_hh
    i_r, i_z, i_n = jnp.split(gi, 3)
    h_r, h_z, h_n = jnp.split(gh, 3)

    reset = jax.nn.sigmoid(i_r + h_r)
    input_gate = jax.nn.sigmoid(i_z + h_z)
    new_gate = jnp.tanh(i_n + (reset * h_n))
    return new_gate + input_gate * (h - new_gate)

def run_gru_layer(inputs, w_ih, b_ih, w_hh, b_hh, return_seq=False):
    """Scans the GRU cell over a time-series sequence."""
    h_size = w_hh.shape[1]
    h0 = jnp.zeros(h_size)

    def step(h, x):
        new_h = gru_cell(x, h, w_ih, b_ih, w_hh, b_hh)
        return new_h, new_h

    last_h, seq_h = jax.lax.scan(step, h0, inputs)
    return seq_h if return_seq else last_h

# ==========================================
# PART 2: THE ARCHITECTURES
# ==========================================

def predict_cond(x_seq, x_ctx, x_sig, w, rnn_lags=20):
    """Architecture 1: Conditional Inputs"""
    ctx_tiled = jnp.tile(x_ctx, (rnn_lags, 1))
    gru_in = jnp.concatenate([x_seq, ctx_tiled], axis=1)

    h1 = run_gru_layer(gru_in, w['gru_weight_ih_l0'], w['gru_bias_ih_l0'], w['gru_weight_hh_l0'], w['gru_bias_hh_l0'], True)
    h2 = run_gru_layer(h1, w['gru_weight_ih_l1'], w['gru_bias_ih_l1'], w['gru_weight_hh_l1'], w['gru_bias_hh_l1'], False)

    return jnp.dot(w['fc_weight'], h2) + w['fc_bias']

def predict_film(x_seq, x_ctx, x_sig, w, rnn_lags=20):
    """Architecture 2: FiLM (Feature-wise Linear Modulation)"""
    ctx_tiled = jnp.tile(x_ctx, (rnn_lags, 1))
    gru_in = jnp.concatenate([x_seq, ctx_tiled], axis=1)

    h1 = run_gru_layer(gru_in, w['gru_weight_ih_l0'], w['gru_bias_ih_l0'], w['gru_weight_hh_l0'], w['gru_bias_hh_l0'], True)
    h2 = run_gru_layer(h1, w['gru_weight_ih_l1'], w['gru_bias_ih_l1'], w['gru_weight_hh_l1'], w['gru_bias_hh_l1'], False)

    # Modulation Logic
    f1 = jax.nn.relu(jnp.dot(w['film_0_weight'], x_sig) + w['film_0_bias'])
    f_out = jnp.dot(w['film_2_weight'], f1) + w['film_2_bias']
    gamma, beta = jnp.split(f_out, 2)

    h_mod = (gamma * h2) + beta
    return jnp.dot(w['fc_weight'], h_mod) + w['fc_bias']

def predict_hyper(x_seq, x_ctx, x_sig, w, rnn_lags=20):
    """Architecture 3: Hypernetworks (Weights generated dynamically)"""
    ctx_tiled = jnp.tile(x_ctx, (rnn_lags, 1))
    gru_in = jnp.concatenate([x_seq, ctx_tiled], axis=1)

    h1 = run_gru_layer(gru_in, w['gru_weight_ih_l0'], w['gru_bias_ih_l0'], w['gru_weight_hh_l0'], w['gru_bias_hh_l0'], True)
    h2 = run_gru_layer(h1, w['gru_weight_ih_l1'], w['gru_bias_ih_l1'], w['gru_weight_hh_l1'], w['gru_bias_hh_l1'], False)

    # Weight Generation
    p1 = jax.nn.relu(jnp.dot(w['hyper_0_weight'], x_sig) + w['hyper_0_bias'])
    p_out = jnp.dot(w['hyper_2_weight'], p1) + w['hyper_2_bias']

    w_gen = p_out[:3072].reshape(96, 32)
    b_gen = p_out[3072:]

    return jnp.dot(h2, w_gen) + b_gen
