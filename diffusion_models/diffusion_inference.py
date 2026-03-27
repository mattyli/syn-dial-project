from dataclasses import dataclass

import transformers

import dllm


@dataclass
class ScriptArguments:
    model_name_or_path: str = "/home/mattli/projects/aip-zhu2048/mattli/hf_cache/hub/models--inclusionAI--LLaDA2.1-mini/snapshots/f21be037104f6e044e1a86b6d8864a6b85cc868e"
    seed: int = 42
    visualize: bool = True

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.pipelines.llada2.LLaDA2SamplerConfig):
    steps_per_block: int = 32
    max_new_tokens: int = 128
    block_size: int = 32
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    threshold: float = 0.95


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

# Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.pipelines.llada2.LLaDA2Sampler(model=model, tokenizer=tokenizer)
terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def pad_to_same_length(inputs: list[list[int]]) -> list[list[int]]:
    """Left-pad a list of token-id lists so all have the same length."""
    max_len = max(len(seq) for seq in inputs)
    return [[pad_id] * (max_len - len(seq)) + seq for seq in inputs]


# --- Example 1: Batch sampling ---
print("\n" + "=" * 80)
print("TEST: llada.sample()".center(80))
print("=" * 80)

messages = [
    [{"role": "user", "content": "Lily runs 12 km/h for 4 hours. How far does she get in 8 hours?"}],
    [{"role": "user", "content": "Please write an educational python function."}],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)
inputs = pad_to_same_length(inputs)

outputs = sampler.sample(inputs, sampler_config, return_dict=True)
sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

for iter, s in enumerate(sequences):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print(s.strip() if s.strip() else "<empty>")
print("\n" + "=" * 80 + "\n")

if script_args.visualize and outputs.histories is not None:
    terminal_visualizer.visualize(outputs.histories, rich=True)

# --- Example 2: Batch fill-in-the-blanks ---
print("\n" + "=" * 80)
print("TEST: llada.infilling()".center(80))
print("=" * 80)

masked_messages = [
    [
        {"role": "user", "content": tokenizer.mask_token * 20},
        {
            "role": "assistant",
            "content": "Sorry, I do not have answer to this question.",
        },
    ],
    [
        {"role": "user", "content": "You are forty year old male who chain smokes regularly, presenting with significant chest pain and shortness of breath. You are engaging in conversation with your doctor.."},
        {
            "role": "assistant",
            "content": "Hey" + tokenizer.mask_token * 256  + "What",
        },
    ],
]

inputs = tokenizer.apply_chat_template(
    masked_messages,
    add_generation_prompt=False,
    tokenize=True,
)
inputs = pad_to_same_length(inputs)

outputs = sampler.infill(inputs, sampler_config, return_dict=True)
seq_ids_list = [outputs.sequences[i, :len(inputs[i])].tolist() for i in range(len(inputs))]
sequences = dllm.utils.infill_trim(tokenizer, seq_ids_list, inputs)

for iter, (i, s) in enumerate(zip(inputs, sequences)):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print("[Masked]:\n" + tokenizer.decode(i))
    print("-" * 80)
    print("[Filled]:\n" + (s.strip() if s.strip() else "<empty>"))
    print("-" * 80)
    print("[Full]:\n" + tokenizer.decode(outputs.sequences.tolist()[iter]))
print("\n" + "=" * 80 + "\n")

if script_args.visualize and outputs.histories is not None:
    terminal_visualizer.visualize(outputs.histories, rich=True)

# --- Example 3: SOAP note infilling ---
# Scenario: 58-year-old male, hypertension + T2DM, presenting with exertional chest pain.
# The assistant turn contains a SOAP note template with masked content sections;
# the model fills in clinically appropriate text given the patient context.
print("\n" + "=" * 80)
print("TEST: SOAP note infilling".center(80))
print("=" * 80)

MASK = tokenizer.mask_token
soap_scenario = (
    "Patient: James Carter, 58-year-old male. PMH: hypertension (lisinopril 10 mg daily), "
    "type 2 diabetes (metformin 1000 mg BID). Presents to outpatient clinic with a 3-day "
    "history of exertional chest pressure radiating to the left arm, associated with mild "
    "diaphoresis and dyspnea on exertion. Denies rest pain, syncope, or palpitations. "
    "Vitals: BP 158/94 mmHg, HR 88 bpm, RR 16, SpO2 98% RA, Temp 36.8°C. "
    "Exam: mild S4 gallop, no rubs or murmurs; lungs clear. "
    "ECG: normal sinus rhythm with no ST changes. Troponin I pending. "
    "Please write a SOAP note for this encounter."
)

soap_template = (
    "SOAP Note — James Carter, 58 M\n\n"
    "S (Subjective):\n" + MASK * 80 + "\n\n"
    "O (Objective):\n" + MASK * 60 + "\n\n"
    "A (Assessment):\n" + MASK * 40 + "\n\n"
    "P (Plan):\n" + MASK * 80
)

soap_messages = [
    [
        {"role": "user", "content": soap_scenario},
        {"role": "assistant", "content": soap_template},
    ]
]

soap_inputs = tokenizer.apply_chat_template(
    soap_messages,
    add_generation_prompt=False,
    tokenize=True,
)
soap_inputs = pad_to_same_length(soap_inputs)

soap_outputs = sampler.infill(soap_inputs, sampler_config, return_dict=True)
soap_seq_ids = [soap_outputs.sequences[i, :len(soap_inputs[i])].tolist() for i in range(len(soap_inputs))]
soap_sequences = dllm.utils.infill_trim(tokenizer, soap_seq_ids, soap_inputs)

for iter, (i, s) in enumerate(zip(soap_inputs, soap_sequences)):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print("[Masked template]:\n" + tokenizer.decode(i))
    print("-" * 80)
    print("[Filled sections]:\n" + (s.strip() if s.strip() else "<empty>"))
    print("-" * 80)
    print("[Full note]:\n" + tokenizer.decode(soap_outputs.sequences.tolist()[iter]))
print("\n" + "=" * 80 + "\n")

if script_args.visualize and soap_outputs.histories is not None:
    terminal_visualizer.visualize(soap_outputs.histories, rich=True)
