from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_path = "./final_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"
)

prompt = """### Instruction:
Translate to French

### Input:
Good Morning

### Response:
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=50
)

print(
    tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )
)