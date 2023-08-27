# Locally hosted: 60% HumanEval

Just -7% behind the full model!  This repo shares how to reproduce the results using a consumer GPU in under 20 minutes.

Check out the blog post: https://www.phind.com/blog/code-llama-beats-gpt4

Uses: https://huggingface.co/TheBloke/Phind-CodeLlama-34B-v1-GPTQ


## How-To

```bash
git clone https://github.com/catid/phind
cd phind

conda create -n phind python=3.10
conda activate phind

pip install -r requirements.txt

# Download the `gptq-4bit-32g-actorder_True` branch of TheBloke/Phind-CodeLlama-34B-v1-GPTQ
# This performs much better than the main branch on the benchmark
git clone --single-branch --branch gptq-4bit-32g-actorder_True https://huggingface.co/TheBloke/Phind-CodeLlama-34B-v1-GPTQ

python test.py

evaluate_functional_correctness samples.jsonl
```

```bash
(phind) ➜  phind git:(master) ✗ evaluate_functional_correctness samples.jsonl
Reading samples...
164it [00:00, 46568.67it/s]
Running test suites...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 164/164 [00:03<00:00, 50.17it/s]
Writing results to samples.jsonl_results.jsonl...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 164/164 [00:00<00:00, 154820.13it/s]
{'pass@1': 0.6036585365853658}
```
