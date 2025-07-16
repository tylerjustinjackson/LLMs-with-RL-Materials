# Reinforcement Learning with Large Language Models (LLMs) — Learning Experience

Welcome to the **Reinforcement Learning with Large Language Models Learning Experience**! This codebase is designed to help you learn about the intersection of Reinforcement Learning (RL) and Large Language Models (LLMs) using Python projects.

- Lead Instructor: Tyler Jackson

- Technical Support Lead: Joseph Jabour
---

## Setup

### 1. **Install Ollama and the Llama 3.2 Model**

This codebase relies on [Ollama](https://ollama.com) to run LLMs locally. **You must:**

1. Go to [ollama.com](https://ollama.com) and download Ollama for your device (macOS, Windows, or Linux).
2. Follow the installation instructions on the Ollama website.
3. Once installed, open your terminal and run:

   ```sh
   ollama pull llama3.2
   ```

   This will download the Llama 3.2 model required for the projects.

---

### 2. **Set Up Your Python Environment**

We recommend using a virtual environment for Python. To install all dependencies, run:

```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Project Structure

This repository is organized into several subprojects:

### `(1)intro-codebase/`
- **Purpose:** Demonstrates basic usage of the Ollama Python API to connect to a local LLM, pull the `llama3.2` model, and generate responses.
- **Try it:** Run `python main.py` in this directory to see a simple prompt/response example.

### `(2)bandit-algorithms/`
- **Purpose:** Implements a Multi-Armed Bandit (MAB) algorithm to optimize LLM prompts. Uses UCB1 to select the best prompt variant based on simulated reward (token overlap with a ground truth answer).
- **Try it:** Run `python main.py` to watch the bandit learn which prompt elicits the best LLM response.

### `(3)RLHF/`
- **Purpose:** Simulates Reinforcement Learning from Human Feedback (RLHF). Generates multiple candidate responses, uses an LLM-based "critic" to rate them, and (optionally) lets a human user provide feedback. Produces a detailed report (`rlhf_report.txt`).
- **Try it:** Run `python main.py` and follow the prompts. Check the generated `rlhf_report.txt` for results.

### `(4)finetuning/`
- **Purpose:** Simulates parameter-efficient fine-tuning (PEFT/LoRA) of LLMs using Hugging Face Transformers. Includes synthetic dataset creation and (optionally) integration with Ollama Modelfiles.
- **Try it:** Run `python main.py` to walk through the fine-tuning simulation. (Requires a GPU for practical use.)

### `(5)RL-loops/`
- **Purpose:** Demonstrates an RL-style iterative self-improvement loop for LLMs. The LLM refines its output using "tools" (instructions) and receives rewards based on output quality.
- **Try it:** Run `python main.py` to see the optimization loop in action.

---

## Utilities

- **`import_debug.py`**: Utility script used in each subproject to ensure required Python packages are installed at runtime.
- **`rlhf_report.txt`**: Example output from the RLHF module, showing candidate responses, critic/user scores, and the best response selected.

---

## Notes & Troubleshooting

- **Ollama must be running** for any LLM-based scripts to work. Start Ollama from your Applications folder or by running `ollama serve` in your terminal.
- If you see errors about missing models, make sure you have run `ollama pull llama3.2`.
- Some modules (especially fine-tuning) require a GPU for practical speed.
- If you encounter missing package errors, the scripts will attempt to install them automatically using `import_debug.py`.

---

## Documentation
- [Ollama Documentation](https://ollama.com/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft/index)
- [Multi-Armed Bandit Algorithms](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [RLHF Overview](https://huggingface.co/blog/rlhf)

---

