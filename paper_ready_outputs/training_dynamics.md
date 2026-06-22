# Training Dynamics

This analysis uses saved `history.csv` files from the best completed tuner trial for each family.

- No TensorBoard event logs were available, so the reconstruction is limited to the saved CSV histories.
- `best_epoch` follows the saved selection metric priority `val_auc -> val_binary_accuracy -> val_loss`.
- `overfitting_onset_epoch` is a heuristic first epoch with two consecutive validation-loss increases while training loss still decreased.