# Leaderboard

Ranking: pooled test AUC desc, F1 desc, latency_ms_mean asc, params_count asc.

| rank | family | model_name | auc | f1 | acc | params_count | latency_ms_mean | latency_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| 1 | cnn_attention | cnn_attention | 0.9959 | 0.8187 | 0.9583 | 156569.0 | 6.6236 | 6.9986 |
| 2 | cnn | cnn | 0.9935 | 0.9054 | 0.9812 | 3729489.0 | 15.0237 | 18.8447 |
| 3 | cnn_focus_hybrid | cnn_focus_hybrid | 0.9776 | 0.8289 | 0.9650 | 102161.0 | NA | NA |
| 4 | transfer | transfer:MobileNet | 0.9748 | 0.7200 | 0.9341 | 3360193.0 | 9.3706 | 9.8054 |
| 5 | hybrid_vit | hybrid_vit | 0.9622 | 0.7089 | 0.9381 | 2390145.0 | 48.7440 | 111.0111 |
| 6 | focus_dnn | focus_dnn | 0.9080 | 0.4141 | 0.8439 | 33793.0 | NA | NA |
| 7 | classical_ml | classical_ml:gradient_boosting | 0.8568 | 0.6218 | 0.9394 | nan | NA | NA |
| 8 | vit | vit | 0.7399 | 0.3968 | 0.8977 | 462721.0 | 42.2628 | 48.6899 |