from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct"
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
)

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