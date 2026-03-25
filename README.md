# syn-dial-project

This repo is for my synthetic dialogue project for ELEC 825. THis will primarily focus on diffusion models.

Need to move downloaded files from user storage ($HOME) to project ($PROJECT)

## Approach
- Need to test at least 3 different models (2 AR and 2 Diffusion?), they should be similar sizes (8B)
- Models
    - AR = Qwen 3 8B, LLaMA 3 8B
    - Diffusion = LLada 8B, Dream ...?

- generate synthetic transcripts, then optimize prompts on those synthetic transcripts and evalute performance of those prompts on downstream tasks.
- OR
    - generate synthetic transcripts, then finetune a downstream LLM (either AR or DIFF) and then evaluate its performance on ACI Bench (like they do in MedSynth)

### Autoregressive Models

### Diffusion Models
- MedSynth firsts constructs notes, then optimizes those, before creating a dialogue based on those notes.
- They use the SOAP (Subjective, Objective, Assessment, Plan) format, commonly used to guide medical documentation.
    - Could try the infilling technique here since it's kinda a natural fit.
- Models to use:
    1. https://huggingface.co/inclusionAI/LLaDA2.1-mini
    2. https://huggingface.co/Dream-org/Dream-v0-Instruct-7B
    3. https://huggingface.co/tencent/WeDLM-7B-Instruct
## Logistics
### Weights Path
Use instruct models
For the AR models, used the predownloaded weights that are shared on the cluster:
```
~/../../model-weights/Qwen2.5-0.5B-Instruct
```

For the Diffusion Models use git LFS to download the model weights from HF. (Don't use LLaDA-8B, use LLaDA 2.1)
```
/home/mattli/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07


```

### Running Inference With DLLM

```
python3 -u sample.py --model_name_or_path "/home/mattli/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
```

## Experiment Setup
### 1. Straight Prompting

