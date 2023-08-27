from transformers import AutoTokenizer, LlamaForCausalLM
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm

# initialize the model

model_path = "Phind-CodeLlama-34B-v1-GPTQ"
model = LlamaForCausalLM.from_pretrained(model_path, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = model.eval()

# HumanEval helper

def generate_one_completion(prompt: str):
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Generate
    generate_ids = model.generate(inputs.input_ids.to("cuda"), max_new_tokens=256, do_sample=True, top_p=0.75, top_k=40, temperature=0.1)
    completion = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    completion = completion.replace(prompt, "").split("\n\n\n")[0]

    return completion

# perform HumanEval
problems = read_problems()

num_samples_per_task = 1
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in tqdm(problems)
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)

# run `evaluate_functional_correctness samples.jsonl` in your HumanEval code sandbox

