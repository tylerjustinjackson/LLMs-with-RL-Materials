import ollama
import sys
import subprocess
import os
from import_debug import install_package


def create_ollama_modelfile_and_model(base_model_name):
    """
    Creates an Ollama Modelfile and attempts to create a custom model.
    This simulates fine-tuning by setting a strong system prompt.
    """
    print("\n--- Creating Ollama Modelfile ---")

    ollama_custom_model_name = "my-ollama-technical-assistant"
    # CORRECTED Modelfile content format: Use escaped newlines (\\) for multi-line SYSTEM prompt
    modelfile_content = f"""
FROM {base_model_name}

PARAMETER temperature 0.2
PARAMETER num_ctx 4096
SYSTEM \"\"\"You are a highly specialized and helpful technical assistant.\\
Your responses are concise, accurate, and focus on providing in-depth explanations\\
related to Large Language Models, fine-tuning concepts (even if simulated), PEFT, and Ollama.\\
Always maintain a professional and informative tone.\"\"\"
"""
    modelfile_path = "Modelfile.custom"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"Modelfile created at: {modelfile_path}")
    print("Modelfile content:")
    print(modelfile_content)

    print(
        f"\n--- Attempting to create Ollama custom model: '{ollama_custom_model_name}' ---"
    )
    try:
        print(
            f"Running command: ollama create {ollama_custom_model_name} -f {modelfile_path}"
        )

        result = subprocess.run(
            ["ollama", "create", ollama_custom_model_name, "-f", modelfile_path],
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(
                f"Ollama custom model '{ollama_custom_model_name}' created successfully."
            )
            if result.stdout:
                print(f"Stdout: {result.stdout.strip()}")
        else:
            if "already exists" in result.stderr or "already exists" in result.stdout:
                print(
                    f"Ollama model '{ollama_custom_model_name}' already exists. Skipping creation."
                )
            else:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    output=result.stdout,
                    stderr=result.stderr,
                )

    except subprocess.CalledProcessError as e:
        print(f"Error creating Ollama model: {e}")
        print(f"Command failed with exit code {e.returncode}")
        print(f"Stdout: {e.stdout.strip()}")
        print(f"Stderr: {e.stderr.strip()}")
        print(
            "Ensure Ollama server is running and the base model is pulled (e.g., 'ollama pull llama3.2')."
        )
        sys.exit(1)
    except ollama.ResponseError as e:
        print(f"Ollama API Error (is Ollama server running?): {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during Ollama model creation: {e}")
        print(f"Type of error: {type(e)}")
        print(f"Error message: {e}")
        sys.exit(1)

    return FIXME


def interact_with_ollama_model(model_name):
    """
    Connects to Ollama and interacts with the specified model,
    demonstrating the custom behavior defined in the Modelfile.
    """
    print(f"\n--- Interacting with Ollama model: '{model_name}' ---")
    print(
        "Note: The 'customization' effect here is primarily driven by the SYSTEM prompt in the Modelfile."
    )

    try:
        prompts = [
            "What are the benefits of using LoRA for fine-tuning?",
            "Explain how Ollama helps with local LLM deployment.",
            "Tell me a short story about a knight.",
            "Define 'catastrophic forgetting' in the context of LLMs.",
        ]

        for i, prompt in enumerate(prompts):
            print(f"\n--- Prompt {i+1}: {prompt} ---")
            response = ollama.generate(
                model=model_name, prompt=prompt, options={"temperature": FIXME}
            )
            print(response["response"])
            print("----------------------------------")

    except ollama.ResponseError as e:
        print(f"Ollama API Error: {e}")
        print(
            f"Please ensure the Ollama server is running and the model '{model_name}' is available/created."
        )
        print(
            "You might need to run 'ollama create <your_custom_model> -f Modelfile.custom' manually if the script failed to do so."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """
    Ensures Ollama is accessible and then creates/interacts with a custom Ollama model.
    """
    install_package("ollama")

    print("\n--- Initial Ollama Connectivity Check ---")
    base_model = "llama3.2"

    try:
        print(f"Attempting to pull '{base_model}' to ensure it's available...")
        for chunk in ollama.pull(base_model, stream=True):
            if "total" in chunk and "completed" in chunk:
                percent = (chunk["completed"] / chunk["total"]) * 100
                print(
                    f"\rDownloading {base_model}: {percent:.2f}% complete...",
                    end="",
                    flush=True,
                )
            elif "status" in chunk:
                print(f"\rStatus: {chunk['status']}...", end="", flush=True)
        print(f"\nModel '{base_model}' is ready or was already present.")

        print(f"Sending a test prompt to '{base_model}'...")
        response = ollama.generate(
            model=base_model,
            prompt="Hi",
            options={"temperature": 0.0, "num_predict": 10},
        )

        print("\n--- Ollama Test Response ---")
        print(response["response"].strip())
        print("----------------------------")
        print(f"Ollama '{base_model}' model seems accessible.")

    except ollama.ResponseError as e:
        print(f"Ollama API Error during initial check/pull: {e}")
        print(f"Please ensure the Ollama server is running and accessible.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during initial Ollama check: {e}")
        sys.exit(1)

    # Step 1: Create Ollama Modelfile and custom model
    ollama_custom_model_name = create_ollama_modelfile_and_model(base_model)

    # Step 2: Interact with the custom Ollama model
    print("\nAttempting to interact with the custom model...")
    interact_with_ollama_model(ollama_custom_model_name)

    print("\n--- Ollama Custom Model Example Complete ---")
    print("This example demonstrates how to customize an Ollama model's behavior")
    print("using a Modelfile and a system prompt, without traditional fine-tuning.")


if __name__ == "__main__":
    main()
