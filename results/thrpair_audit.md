# Per-pair threshold audit (source of truth)

target=gemma L=2, max_overlap=0.1, exclude=['Formal', 'HonestyShort', 'TensePresent']

| column | sub-group | n | median p | mean p | 95% CI | median residual |
|---|---|---:|---:|---:|---|---:|
| Model | gemma-2-9b | 318 | 2.301 | 2.423 | [2.285, 2.619] | 1.14% |
| Model | qwen3-1.7b | 310 | 2.226 | 2.270 | [2.163, 2.388] | 1.28% |
| Model | llama-3.1-8b | 318 | 2.502 | 2.632 | [2.388, 2.897] | 1.10% |
| Model | mistral-7b-v0.3 | 307 | 2.353 | 2.390 | [2.252, 2.546] | 0.91% |
| Model | aya-expanse-8b | 314 | 2.357 | 2.427 | [2.276, 2.607] | 0.95% |
| Model | yi-1.5-9b | 236 | 2.235 | 2.271 | [2.156, 2.394] | 1.28% |
| Perturbation layer | L=2 | 318 | 2.301 | 2.423 | [2.285, 2.619] | 1.14% |
| Perturbation layer | L=5 | 316 | 2.381 | 2.480 | [2.345, 2.659] | 1.12% |
| Perturbation layer | L=10 | 314 | 2.383 | 2.450 | [2.285, 2.660] | 1.25% |
| Perturbation layer | L=20 | 301 | 2.286 | 2.387 | [2.229, 2.605] | 1.34% |
| Measurement layer | penult | 318 | 2.301 | 2.423 | [2.285, 2.619] | 1.14% |
| Measurement layer | penult-5 | 318 | 2.207 | 2.321 | [2.207, 2.495] | 1.04% |
| Measurement layer | penult-10 | 318 | 2.218 | 2.295 | [2.199, 2.450] | 0.86% |
| Fit metric | $L^2$ | 318 | 2.301 | 2.423 | [2.285, 2.619] | 1.14% |
| Fit metric | $1-\cos$ | 318 | 2.205 | 2.362 | [2.208, 2.591] | 0.99% |
| Fit metric | KL | 318 | 2.196 | 2.361 | [2.198, 2.587] | 1.66% |
| Response threshold | 0.5xT | 318 | 2.302 | 2.427 | [2.284, 2.620] | 1.14% |
| Response threshold | 1.0xT | 318 | 2.301 | 2.423 | [2.285, 2.619] | 1.14% |
| Response threshold | 2.0xT | 318 | 2.130 | 2.250 | [2.125, 2.429] | 1.14% |
| Perturbation method | norm-matched | 318 | 2.301 | 2.423 | [2.285, 2.619] | 1.14% |
| Perturbation method | additive | 318 | 2.259 | 2.403 | [2.220, 2.643] | 1.56% |
| Anchor source | FineWeb-edu | 318 | 2.301 | 2.423 | [2.285, 2.619] | 1.14% |
| Anchor source | Wikipedia (en) | 318 | 2.153 | 2.245 | [2.138, 2.377] | 1.11% |
| Anchor source | Wikipedia (zh) | 318 | 2.215 | 2.253 | [2.127, 2.387] | 1.18% |
| Anchor source | Code | 318 | 2.107 | 2.244 | [2.107, 2.376] | 1.32% |
| Token position | pos $-1$ | 318 | 2.301 | 2.423 | [2.285, 2.619] | 1.14% |
| Token position | pos $-2$ | 318 | 2.293 | 2.400 | [2.270, 2.611] | 1.19% |
| Token position | pos $-3$ | 318 | 2.349 | 2.499 | [2.334, 2.743] | 1.00% |
| Direction family | Contrastive | 318 | 2.301 | 2.423 | [2.285, 2.619] | 1.14% |
| Direction family | MELBO | 507 | 2.162 | 2.212 | [2.013, 2.420] | 1.88% |
| Direction family | SAE | 395 | 2.188 | 2.281 | [2.139, 2.466] | 1.04% |
| Direction family | PCA | 528 | 1.968 | 2.026 | [1.936, 2.120] | 1.20% |
| Direction family | Random-diff | 710 | 2.018 | 2.050 | [2.001, 2.098] | 0.85% |
| Direction family | Random | 528 | 2.022 | 2.033 | [1.996, 2.072] | 0.73% |
| Misalignment | theta=0 | 160 | 2.379 | 2.495 | -- | 1.07% |
| Misalignment | theta=15 | 160 | 2.357 | 2.465 | -- | 1.00% |
| Misalignment | theta=30 | 160 | 2.294 | 2.381 | -- | 0.94% |
| Misalignment | theta=45 | 160 | 2.231 | 2.287 | -- | 0.89% |
| Misalignment | theta=60 | 160 | 2.082 | 2.121 | -- | 0.88% |
| Misalignment | theta=75 | 160 | 2.012 | 2.029 | -- | 0.78% |
| Misalignment | theta=90 | 160 | 2.004 | 2.026 | -- | 0.71% |
