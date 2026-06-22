# Deployment Discussion

- Saved summary artifacts show a clear trade-off between discrimination and deployability.
- The smallest deep architectures (`cnn_attention`, `cnn_focus_hybrid`, `focus_dnn`) were materially lighter than transfer / transformer families, while the transfer and transformer models retained higher file sizes and/or latency.
- Any deployment argument should therefore report both balanced discrimination and the saved CPU latency / file-size evidence rather than AUC alone.
- No new latency benchmark was run for this package; only existing saved latency artifacts were used.