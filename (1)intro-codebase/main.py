# Import necessary libraries
import ollama
from import_debug import install_package


def main():
    """
    Connects to Ollama, pulls the llama3.2 model, and generates a response.
    """
    # Ensure the ollama package is installed
    install_package("ollama")

    try:
        print("\nAttempting to pull the 'llama3.2' model from Ollama...")
        # This command will pull the model if it's not already available locally.
        # It's good practice to ensure the model is present before trying to use it.
        # Note: Depending on the Ollama server version and network, this might take some time.
        # You might need to run `ollama pull llama3.2` manually in your terminal
        # if this programmatic pull causes issues or if you prefer to manage models manually.
        # The `ollama` python client doesn't have a direct `pull` method that blocks
        # until completion in the same way `ollama run` does, so we'll rely on
        # the `generate` call to implicitly handle it or assume it's pre-pulled.
        # For a more robust solution, you might consider using subprocess to run
        # `ollama pull llama3.2` directly if you encounter "model not found" errors.

        # Let's try a simple generate call. Ollama will often attempt to pull
        # if the model isn't found when a generate request is made.
        print("Sending a test prompt to 'llama3.2'...")
        response = ollama.generate(model=FIXME, prompt="Why is the sky blue?")

        print("\n--- Model Response ---")
        print(response["response"])
        print("----------------------")

        # You can also stream responses for longer outputs
        print("\n--- Streaming Response Example ---")
        stream = ollama.generate(
            model=FIXME,
            prompt="Tell me a short story about a brave knight.",
            stream=FIXME,
            options={"temperature": FIXME},
        )
        for chunk in stream:
            print(chunk["response"], end="", flush=True)
        print("\n----------------------------------")

    except ollama.ResponseError as e:
        print(f"Ollama API Error: {e}")
        print(
            "Please ensure the Ollama server is running and the 'llama3.2' model is available."
        )
        print("You might need to run 'ollama pull llama3.2' in your terminal.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    FIXME()
