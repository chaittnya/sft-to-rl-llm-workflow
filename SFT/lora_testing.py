from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel

# device_map="auto" puts the base model on the GPU (NVIDIA RTX 2050) if one
# is available, instead of defaulting to the CPU.
base_model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "./final_lora"
)

tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct"
)

prompt = "What is reinforcement learning?"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

print(
    tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
)