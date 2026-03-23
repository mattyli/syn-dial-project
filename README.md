# syn-dial-project

This repo is for my synthetic dialogue project.

## Approach
- Need to test at least 3 different models (2 AR and 2 Diffusion?), they should be similar sizes (8B)
- Models
    - AR = Qwen 3 8B, LLaMA 3 8B
    - Diffusion = LLada 8B, Dream ...?

- generate synthetic transcripts, then optimize prompts on those synthetic transcripts and evalute performance of those prompts on downstream tasks.

## Logistics
### Weights Path
Use instruct models
For the AR models, used the predownloaded weights that are shared on the cluster:
```
~/../../model-weights/Qwen2.5-0.5B-Instruct
```

For the Diffusion Models
```
/home/mattli/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07


```

### Running Inference

Input must be a JSONL file where each line has a `"prompt"` key, e.g.:
```json
{"prompt": "Generate a short customer service dialogue about a lost package."}
```

### Submit to Slurm

```bash
sbatch --export=ALL,INPUT=data/prompts.jsonl,OUTPUT=data/responses.jsonl run_inference.sh
```

Override `INPUT`/`OUTPUT` as needed. Logs are written to `logs/`.

### Run interactively

```bash
python3 run_inference.py --input data/prompts.jsonl --output data/responses.jsonl
```

### Running Inference With DLLM

```
python3 -u sample.py --model_name_or_path "/home/mattli/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
```
