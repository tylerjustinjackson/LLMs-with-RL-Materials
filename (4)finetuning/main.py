import ollama
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import sys
from import_debug import install_package


def setup_fine_tuning_environment():
    """
    Installs required packages for fine-tuning simulation.
    """
    print("\n--- Setting up fine-tuning environment ---")
    required_packages = [
        "ollama",
        "datasets",
        "peft",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "flash_attn",
    ]
    for package in required_packages:
        install_package(package)

    # Check for GPU availability
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    elif torch.backends.mps.is_available():  # For Apple Silicon Macs
        print("MPS is available! Using Apple Silicon GPU.")
        device = "mps"
    else:
        print("No GPU detected. Fine-tuning will be very slow on CPU.")
        device = "cpu"
    return device


def create_synthetic_dataset():
    """
    Creates a small, synthetic dataset for instruction tuning.
    We'll train the model to respond in a specific "helpful, technical assistant" persona.
    """
    print("\n--- Creating synthetic dataset ---")
    data = [
        {
            "instruction": "Explain the concept of fine-tuning in LLMs.",
            "output": "Fine-tuning in Large Language Models is the process of further training a pre-trained model on a smaller, specific dataset to adapt its knowledge and behavior to a particular task or domain. This typically involves adjusting a subset of the model's parameters using techniques like LoRA to make the process more efficient.",
        },
        {
            "instruction": "What is PEFT?",
            "output": "PEFT stands for Parameter-Efficient Fine-Tuning. It's a collection of techniques (like LoRA, Adapters) that allow you to fine-tune large models without modifying all their parameters, significantly reducing computational resources and memory requirements.",
        },
        {
            "instruction": "How does LoRA work?",
            "output": "LoRA (Low-Rank Adaptation) works by injecting small, trainable matrices into the transformer layers of a pre-trained model. During fine-tuning, only these low-rank matrices are updated, keeping the original model weights frozen. This greatly reduces the number of parameters that need to be trained, making it faster and less memory-intensive.",
        },
        {
            "instruction": "Tell me about the importance of Modelfiles in Ollama.",
            "output": "Modelfiles in Ollama are configuration files that define how a model behaves. They allow you to specify the base model, system prompts, inference parameters (like temperature, context length), and even custom instructions, effectively creating a personalized version of an existing model without full re-training.",
        },
        {
            "instruction": "What's the difference between pre-training and fine-tuning?",
            "output": "Pre-training is the initial phase where an LLM learns general language patterns from massive amounts of diverse text data. Fine-tuning, on the other hand, is a subsequent phase where the pre-trained model is adapted to a specific task or domain using a smaller, task-specific dataset, often with techniques like PEFT.",
        },
        {
            "instruction": "Summarize the benefits of using Ollama.",
            "output": "Ollama simplifies running and managing large language models locally. Its benefits include ease of use, ability to run various models, customization via Modelfiles, and a convenient API for integration into applications.",
        },
    ]

    # Format for Causal LM: combine instruction and output into a single 'text' field
    # We'll use a simple instruction-response template for clarity
    formatted_data = []
    for item in data:
        # A common format for instruction tuning
        formatted_data.append(
            {
                "text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            }
        )

    dataset = Dataset.from_list(formatted_data)
    print(f"Dataset created with {len(dataset)} examples.")
    print("Example data point:")
    print(dataset[0]["text"])
    return dataset


def simulate_fine_tuning(dataset, device):
    """
    Simulates fine-tuning a small LLM using Hugging Face Transformers and PEFT (LoRA).
    This part requires a good GPU for practical use.
    """
    print("\n--- Starting simulated fine-tuning (LoRA) ---")

    # Choose a small, instruction-tuned model from Hugging Face
    # Using a small Llama model or similar for demonstration
    # Replace with a model you have resources for, e.g., "meta-llama/Llama-2-7b-hf" or "google/gemma-2b"
    # Ensure you have accepted the model's terms on Hugging Face if it's a gated model.
    model_name = "google/gemma-2b"  # A relatively small model

    # Load tokenizer and model
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding token if not present (common for some models like Gemma)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # Or specify a different one if preferred

    # Load model in 4-bit for memory efficiency using bitsandbytes
    # Ensure bitsandbytes is installed and compatible with your CUDA/GPU setup
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
            load_in_4bit=True,  # Enable 4-bit quantization
        )
        print("Model loaded with 4-bit quantization.")
    except Exception as e:
        print(
            f"Error loading model with 4-bit quantization (likely no GPU/bitsandbytes issue): {e}"
        )
        print(
            "Attempting to load model without 4-bit quantization (may require more VRAM or run on CPU)."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
        )

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # LoRA attention dimension
        lora_alpha=16,  # Alpha parameter for LoRA scaling
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Modules to apply LoRA to
        bias="none",  # Bias type in LoRA layers
        task_type=TaskType.CAUSAL_LM,  # Task type for language modeling
        lora_dropout=0.05,  # Dropout probability for LoRA layers
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize the dataset
    def tokenize_function(examples):
        # We need to ensure the labels are correctly set for causal language modeling
        # For instruction tuning, the model learns to predict the next token based on the prompt+response
        tokenized_output = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,  # Max context length
            padding="max_length",  # Pad to max_length
            return_tensors="pt",
        )
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    print(f"Tokenized dataset size: {len(tokenized_dataset)}")
    print("Example tokenized data point (input_ids snippet):")
    print(tokenized_dataset[0]["input_ids"][:20])  # Print first 20 tokens

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=1,  # Adjust based on GPU memory. Lower if OOM.
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 batches to simulate larger batch size
        learning_rate=2e-4,  # Learning rate for fine-tuning
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",  # Save checkpoint every epoch
        push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
        report_to="none",  # Disable reporting to W&B etc.
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    print("\n--- Training started ---")
    trainer.train()
    print("--- Training complete ---")

    # Save the fine-tuned LoRA adapters
    lora_output_dir = "./fine_tuned_lora_adapters"
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    print(f"Fine-tuned LoRA adapters and tokenizer saved to: {lora_output_dir}")

    # For demonstration, we'll return a path that *would* contain the merged model
    # In a real scenario, you'd need to merge the base model with these adapters
    # and then convert to GGUF using tools like llama.cpp's convert.py or unsloth's export.
    return lora_output_dir, model_name


def create_ollama_modelfile_and_model(fine_tuned_model_path, base_model_name):
    """
    Simulates the creation of an Ollama Modelfile and a custom model.
    In a real scenario, 'fine_tuned_model_path' would refer to a GGUF file.
    For this simulation, we'll create a Modelfile that *pretends* to load
    a model that has the fine-tuned persona.
    """
    print("\n--- Simulating Ollama Modelfile creation ---")

    # Define the name for our custom Ollama model
    ollama_custom_model_name = "my-finetuned-technical-assistant"
    modelfile_content = f"""
FROM {base_model_name}
# Note: In a real scenario, you'd use FROM /path/to/your/merged_model.gguf
# For simulation, we'll use the base model and rely on system prompt for 'finetuned' behavior.

PARAMETER temperature 0.2
PARAMETER num_ctx 4096
SYSTEM '''
You are a highly specialized and helpful technical assistant.
Your responses are concise, accurate, and focus on providing in-depth explanations
related to Large Language Models, fine-tuning, PEFT, and Ollama.
Always maintain a professional and informative tone.
'''
"""
    modelfile_path = "Modelfile.custom"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"Modelfile created at: {modelfile_path}")
    print("Modelfile content:")
    print(modelfile_content)

    print(
        f"\n--- Simulating Ollama custom model creation: '{ollama_custom_model_name}' ---"
    )
    print(
        "This step would typically involve: 'ollama create my-finetuned-model -f Modelfile.custom'"
    )
    print("For this simulation, we'll assume the model is 'created' and ready to use.")

    # In a real scenario, you'd execute:
    # try:
    #     subprocess.run(["ollama", "create", ollama_custom_model_name, "-f", modelfile_path], check=True)
    #     print(f"Ollama custom model '{ollama_custom_model_name}' created successfully.")
    # except subprocess.CalledProcessError as e:
    #     print(f"Error creating Ollama model: {e}")
    #     print("Ensure Ollama server is running and base model is pulled.")
    #     sys.exit(1)

    return ollama_custom_model_name


def interact_with_ollama_model(model_name):
    """
    Connects to Ollama and interacts with the specified model,
    demonstrating the simulated fine-tuned behavior.
    """
    print(f"\n--- Interacting with Ollama model: '{model_name}' ---")
    print(
        "Note: The 'fine-tuning' effect here is primarily driven by the SYSTEM prompt in the Modelfile simulation."
    )

    try:
        # Check if the model is available (important if ollama create was run externally)
        ollama_list = ollama.list()
        if not any(
            item["name"].startswith(model_name) for item in ollama_list["models"]
        ):
            print(
                f"Model '{model_name}' not found in Ollama list. Attempting to pull or assuming it will be created implicitly."
            )
            # If `ollama create` was skipped for simulation, direct generation might still work
            # if the base model exists and the Modelfile was conceptually applied.

        # Test prompts to see the "fine-tuned" behavior
        prompts = [
            "What are the benefits of using LoRA for fine-tuning?",
            "Explain how Ollama helps with local LLM deployment.",
            "Tell me a short story about a knight.",  # This should show the effect of the technical persona
            "Define 'catastrophic forgetting' in the context of LLMs.",
        ]

        for i, prompt in enumerate(prompts):
            print(f"\n--- Prompt {i+1}: {prompt} ---")
            # Set a low temperature for more deterministic, "fine-tuned" like output
            response = ollama.generate(
                model=model_name, prompt=prompt, options={"temperature": 0.1}
            )
            print(response["response"])
            print("----------------------------------")

    except ollama.ResponseError as e:
        print(f"Ollama API Error: {e}")
        print(
            f"Please ensure the Ollama server is running and the model '{model_name}' is available/created."
        )
        print(
            "You might need to run 'ollama pull <base_model_name>' and then 'ollama create <your_custom_model> -f Modelfile.custom' manually."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """
    Connects to Ollama and checks if llama3.2 model is accessible.
    This function is primarily for demonstrating initial Ollama connectivity.
    It relies on `install_package` from `import_debug.py`.
    """
    print("\n--- Initial Ollama Connectivity Check ---")
    try:
        # Check if the model is available. This might implicitly pull if not present.
        print("Sending a test prompt to 'llama3.2'...")
        response = ollama.generate(
            model="llama3.2",
            prompt="Hello!",
            options={"temperature": 0.0, "num_predict": 10},
        )

        print("\n--- Ollama Test Response ---")
        print(response["response"].strip())
        print("----------------------------")
        print("Ollama 'llama3.2' model seems accessible.")

    except ollama.ResponseError as e:
        print(f"Ollama API Error during initial check: {e}")
        print(
            "Please ensure the Ollama server is running and the 'llama3.2' model is pulled."
        )
        print("You might need to run 'ollama pull llama3.2' in your terminal.")
        # Exit if the initial Ollama check fails, as the rest of the script relies on it.
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during initial Ollama check: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # We call this first to ensure Ollama server is running and a base model (llama3.2) is ready.
    # This function uses the `install_package` from `import_debug`.
    main()

    # Step 1: Set up the environment for fine-tuning and create dataset
    device = setup_fine_tuning_environment()
    synthetic_dataset = create_synthetic_dataset()

    # Step 2: Simulate fine-tuning (this is the computationally intensive part)
    fine_tuned_lora_output_dir, base_model_used = simulate_fine_tuning(
        synthetic_dataset, device
    )

    # Step 3: Simulate Ollama Modelfile and custom model creation
    ollama_custom_model_name = create_ollama_modelfile_and_model(
        fine_tuned_lora_output_dir, base_model_used
    )

    # Step 4: Interact with the (simulated) fine-tuned Ollama model
    print("\nAttempting to interact with the simulated fine-tuned model...")
    interact_with_ollama_model(ollama_custom_model_name)

    print("\n--- Complex Fine-tuning Example (Simulated) Complete ---")
    print("To truly run a fine-tuned model with Ollama, you would typically:")
    print(
        "1. Fine-tune your chosen model (e.g., Llama 3 8B) with your data and PEFT (LoRA)."
    )
    print(
        "2. Merge the LoRA adapters with the base model to get a full model checkpoint."
    )
    print(
        "3. Convert this merged model into the GGUF format (often using llama.cpp's convert.py or tools like unsloth's export_to_gguf)."
    )
    print(
        "4. Create an Ollama Modelfile that uses 'FROM /path/to/your/merged_model.gguf'."
    )
    print(
        "5. Run 'ollama create your-custom-model-name -f your-modelfile-path' in your terminal."
    )
    print(
        "6. Then, use 'ollama.generate(model='your-custom-model-name', ...)' in Python."
    )
