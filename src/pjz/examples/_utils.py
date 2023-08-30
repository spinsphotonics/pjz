from datetime import datetime
import os
import pickle

import jax


def optimize(
    numsteps,
    init_state,
    loss_fn,
    params_fn,
    update_fn,
    path=None,
    progress_fn=None,
    loss_threshold=None,
):
  """Run an optimization, saving state to disk."""
  if path is not None:
    timestamp = datetime.strftime(datetime.utcnow(), "%Y-%m-%d@%H%M%S")
    os.makedirs(os.path.join(path, timestamp), exist_ok=True)

  def loss(theta, i):
    return loss_fn(i, theta)

  def step(i, state):
    (value, aux), grads = (
        jax.value_and_grad(loss, has_aux=True)(params_fn(state), i))
    state = update_fn(i, grads, state)
    return state, value, aux

  state = init_state
  aux_hist = []
  for i in range(numsteps):
    state, value, aux = step(i, state)
    aux_hist.append(aux)

    if path is not None:
      filename = os.path.join(path, timestamp, f"{i:05d}@{value:1.4e}")
      with open(filename, "wb") as f:
        pickle.dump(state, f)

    if progress_fn is not None:
      progress_fn(i, params_fn(state), aux_hist)

    if loss_threshold is not None and value <= loss_threshold:
      break

  return state
