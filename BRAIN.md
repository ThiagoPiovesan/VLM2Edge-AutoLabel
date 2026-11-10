# BRAIN — VLM2Edge-AutoLabel

## Visão Geral
Pipeline de destilação **VLM → YOLO → TensorRT**: um VLM local (`Qwen3-VL-4B-Instruct`) rotula imagens automaticamente via fila resiliente (Redis + DLQ), o conjunto rotulado treina um modelo YOLO, e o modelo final é exportado para TensorRT.

## Hardware & Restrições Locais
- **GPU**: NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM; ~4,8 GB livres).
- **Ambiente**: WSL2 (Ubuntu 24.04) + Docker CE + NVIDIA Container Toolkit + `uv`.
- **Regra de execução**: VLM e treino YOLO nunca rodam simultaneamente na GPU.

## Decisões de Arquitetura (2026-10-21)
1. **Variante do Modelo**: `Qwen3-VL-4B-Instruct` (descartado `Thinking` pelo alto consumo de tokens/KV cache sem ganho em coordenadas).
2. **Quantização**:
   - **vLLM**: `cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit` (~4,24 GiB totais na VRAM somando LLM 4-bit + Vision Tower 16-bit + `lm_head`; requer `--max-model-len 1536 --kv-cache-dtype fp8 --gpu-memory-utilization 0.82 --mm-processor-kwargs '{"max_pixels": 501760}' --enforce-eager`).
   - **Alternativa leve (GGUF)**: `Qwen/Qwen3-VL-4B-Instruct-GGUF` (`Q4_K_M` + `mmproj Q8_0`) via `llama-server` (~3,1 GB na VRAM).
   - **Descartado FP8 local**: ~4,5 GB só de pesos estoura os 4,8 GB livres quando soma encoder visual + KV cache.
3. **Dataset Inicial**: **TACO** (Trash Annotations in Context — objetos maiores e mais nítidos que VisDrone/SKU-110K, formato COCO nativo).
4. **Avaliação em 2 níveis**:
   - **Nível 1 (S1, sem treino)**: VLM vs Ground Truth. `metrics.py` (IoU, matching guloso, Precisão/Recall/F1) implementado **à mão pelo Thiago**.
     - **Resultado Nível 1 (981 imgs TACO, IoU >= 0.5)**: `TP=1147 | FP=193 | FN=1808` → **Precision = 85,60% | Recall = 38,82% | F1 = 53,41%** (subiu de 46,15% com 150 imgs para 53,41% com 981 imgs).
   - **Nível 2 (S2, com treino)**: YOLO treinado com auto-label vs YOLO treinado com GT, avaliados no mesmo split `val` humano (`src/vlm2edge/datasets/splits.py` + `scripts/train_yolo.py` com tracking no MLflow em `http://localhost:5000`).
     - **Baseline `yolo-gt` (50 épocas concluídas)**: `Precision = 67,60% | Recall = 42,25% | mAP50 = 48,15% | mAP50-95 = 33,95%`.
     - **Destilado `yolo-vlm` (50 épocas concluídas)**: `Precision = 66,63% | Recall = 37,50% | mAP50 = 40,27% | mAP50-95 = 27,51%` → **Retém 83,6% do mAP50 e 98,6% da Precisão do treino humano com zero anotação manual!**
5. **Export & Benchmark (S2)**:
   - Export para ONNX e TensorRT FP16/INT8 via `scripts/export_yolo.py`.
   - Cálculo de `mean_ms`, `p50_ms`, `p95_ms` e `fps` em `src/vlm2edge/eval/benchmark.py` (implementado **à mão pelo Thiago**).
   - **Resultados de Latência/FPS + mAP no split `val` (196 imgs, RTX 4050 GPU vs Core i5-13420H CPU)**:
     - `best.pt` (CPU): `Mean = 80,78 ms | p50 = 81,91 ms | p95 = 97,95 ms | 12,4 FPS`
     - `best.onnx` (CPU ONNX Runtime): `Mean = 63,57 ms | p50 = 62,52 ms | p95 = 73,78 ms | 15,7 FPS | mAP50 = 38,87% | mAP50-95 = 27,41%`
     - `best.pt` (GPU PyTorch): `Mean = 21,84 ms | p50 = 20,00 ms | p95 = 39,72 ms | 45,8 FPS | mAP50 = 39,36% | mAP50-95 = 27,53%`
     - `best_fp16.engine` (GPU TensorRT FP16, 59,4 MB, 65 MiB VRAM): `Mean = 8,20 ms | p50 = 7,96 ms | p95 = 10,92 ms | 122,0 FPS | mAP50 = 38,84% | mAP50-95 = 27,40%` (**98,7% do mAP50 do `.pt` com 2,66x mais FPS na GPU e 9,8x mais FPS que CPU**)
     - `best_int8.engine` (GPU TensorRT INT8, 31,1 MB, 36 MiB VRAM): `Mean = 8,49–11,22 ms | p50 = 7,77–11,10 ms | p95 = 12,59–15,32 ms | 89–117,8 FPS | mAP50 = 38,17% | mAP50-95 = 26,59%` (**97,0% do mAP50 do `.pt` com -47% de disco e -45% de VRAM que FP16**)
6. **Documentação & Fechamento S3 (2026-11-04)**:
   - `README.md` completo com métrica-manchete, diagrama de arquitetura Mermaid em 4 estágios, 4 tabelas experimentais (Nível 1, Nível 2, Custo/Tempo por 1.000 imgs, Benchmark/Trade-off FP16 vs INT8), decisões de engenharia, limitações e nota sobre licença AGPL-3.0 da Ultralytics.
5. **Inspeção Visual (FiftyOne)**: `scripts/inspect_fiftyone.py` (`make fiftyone`) sobrepõe `ground_truth` e `vlm_pred` na porta 5151 e ordena por divergência (`box_diff`).
