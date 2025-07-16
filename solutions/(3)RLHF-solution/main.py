import ollama
import re
import time
import os
from import_debug import install_package


def get_llm_response(
    model_name: str, prompt_text: str, temperature: float = 0.7, stream: bool = False
) -> str:
    """
    Generates a response from the specified Ollama model.
    """
    try:
        print(
            f"Generating response from '{model_name}' (temperature: {temperature})..."
        )
        if stream:
            full_response = ""
            stream_chunks = ollama.generate(
                model=model_name,
                prompt=prompt_text,
                stream=True,
                options={"temperature": temperature},
            )
            for chunk in stream_chunks:
                full_response += chunk["response"]
                # print(chunk['response'], end='', flush=True) # Uncomment to see streaming output
            return full_response
        else:
            response = ollama.generate(
                model=model_name,
                prompt=prompt_text,
                options={"temperature": temperature},
            )
            return response["response"]
    except ollama.ResponseError as e:
        print(f"Ollama API Error for model '{model_name}': {e}")
        print(
            f"Please ensure the Ollama server is running and the '{model_name}' model is available."
        )
        print(f"You might need to run 'ollama pull {model_name}' in your terminal.")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred during LLM response generation: {e}")
        return ""


def generate_candidate_responses(
    policy_model: str, user_prompt: str, num_responses: int = 3
) -> list[str]:
    """
    Generates multiple diverse candidate responses using the policy model.
    By setting a non-zero temperature, we encourage varied outputs.
    """
    print(f"\n--- Generating {num_responses} Candidate Responses ---")
    candidate_responses = []
    for i in range(num_responses):
        print(f"Generating candidate {i+1}/{num_responses}...")
        # Use a slightly different temperature or just repeat to get diversity
        response = get_llm_response(
            policy_model, user_prompt, temperature=0.7 + (i * 0.1), stream=False
        )
        if response:
            candidate_responses.append(response.strip())
        time.sleep(0.5)  # Small delay to avoid overwhelming Ollama server
    return candidate_responses


def get_reward_from_critic(
    critic_model: str, original_prompt: str, candidate_response: str
) -> float:
    """
    Uses the critic model to evaluate a candidate response and assign a reward score.
    The critic is prompted to give a numerical rating.
    """
    critic_prompt = f"""
    You are an AI assistant evaluating the quality of responses.
    Given the original prompt: "{original_prompt}"

    And the candidate response:
    ---
    {candidate_response}
    ---

    Rate the candidate response on a scale of 1 to 5, where:
    1 = Very Poor (irrelevant, harmful, or completely wrong)
    2 = Poor (partially relevant, some inaccuracies)
    3 = Fair (mostly relevant, minor issues)
    4 = Good (relevant, accurate, helpful)
    5 = Excellent (highly relevant, accurate, comprehensive, well-written)

    Provide only the numerical rating.
    Rating:
    """
    print(f"\n--- Critic Evaluating Response ---")
    # print(f"Original Prompt: {original_prompt[:50]}...") # Too verbose for each eval
    # print(f"Candidate Response: {candidate_response[:100]}...") # Too verbose for each eval

    critic_output = get_llm_response(
        critic_model, critic_prompt, temperature=0.0, stream=False
    )  # Use low temp for critic for consistent rating

    # Try to extract the numerical rating
    match = re.search(r"Rating:\s*(\d+)", critic_output)
    print("Critic Output", critic_output)
    if match:
        try:
            score = float(match.group(1))
            # Ensure score is within valid range (1-5)
            score = max(1.0, min(5.0, score))
            print(f"Critic Score: {score}")
            return score
        except ValueError:
            print(
                f"Could not parse score from critic output: '{critic_output}'. Defaulting to 1.0."
            )
            return 1.0
    else:
        print(
            f"No rating found in critic output: '{critic_output}'. Defaulting to 1.0."
        )
        return 1.0


def get_user_rating(response_text: str, critic_score: float, user_prompt: str) -> float:
    """
    Prompts the user to rate a response.
    """
    while True:
        print("\n--------------------------------------------------")
        # Print the AI critic's score and the response first
        print(f"The AI critic rated this response: {critic_score:.1f}/5")
        print("---")
        print("Question:", user_prompt, "\n\nResponse:\n")
        print(response_text)
        print("---")

        # Then, prompt the user for their rating
        user_input = input(
            "Please review the above response and provide your rating (1-5, or press Enter to accept critic's score): "
        ).strip()

        if not user_input:
            print(f"Accepted critic's score: {critic_score:.1f}")
            return critic_score
        try:
            user_score = float(user_input)
            if 1 <= user_score <= 5:
                print(f"Your rating: {user_score:.1f}")
                return user_score
            else:
                print("Invalid input. Please enter a number between 1 and 5.")
        except ValueError:
            print(
                "Invalid input. Please enter a number between 1 and 5, or press Enter."
            )


def generate_report(report_data: dict, filename: str = "rlhf_report.txt"):
    """
    Generates a text file report of the RLHF process.
    """
    print(f"\n--- Generating Report: {filename} ---")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("               RLHF Process Report                \n")
            f.write("==================================================\n\n")

            f.write(f"Date and Time: {time.ctime()}\n")
            f.write(f"Policy Model: {report_data['policy_model']}\n")
            f.write(f"Critic Model: {report_data['critic_model']}\n")
            f.write(f"Original User Prompt: \"{report_data['user_prompt']}\"\n\n")

            f.write("--------------------------------------------------\n")
            f.write("              Candidate Responses                 \n")
            f.write("--------------------------------------------------\n")
            for i, item in enumerate(report_data["scored_responses"]):
                f.write(f"\nCandidate Response #{i+1}:\n")
                f.write(f"  Critic Score: {item['critic_score']:.1f}/5\n")
                if item["user_score"] is not None:
                    f.write(f"  User Score: {item['user_score']:.1f}/5\n")
                f.write(
                    f"  Final Score Used for Selection: {item['final_score']:.1f}/5\n"
                )
                f.write("  Response:\n")
                f.write(f"    {item['response']}\n")
                f.write("-" * 50 + "\n")

            f.write("\n==================================================\n")
            f.write("             RLHF Chosen Best Response            \n")
            f.write("==================================================\n")
            if report_data["best_response_info"]:
                best_info = report_data["best_response_info"]
                f.write(
                    f"Chosen Response Final Score: {best_info['final_score']:.1f}/5\n"
                )
                f.write(
                    f"Chosen Response Critic Score: {best_info['critic_score']:.1f}/5\n"
                )
                if best_info["user_score"] is not None:
                    f.write(
                        f"Chosen Response User Score: {best_info['user_score']:.1f}/5\n"
                    )
                f.write("\nResponse Content:\n")
                f.write(best_info["response"] + "\n")
            else:
                f.write("No best response was chosen.\n")

            f.write("\n==================================================\n")
            f.write("                 End of Report                    \n")
            f.write("==================================================\n")
        print(f"Report successfully written to {filename}")
    except IOError as e:
        print(f"Error writing report to file {filename}: {e}")


def run_rlhf_example(policy_model: str = "llama3.2", critic_model: str = "llama3.2"):
    """
    Demonstrates a simplified RLHF workflow with user feedback and report generation.
    """
    # Ensure ollama package is installed
    install_package("ollama")

    user_prompt = "Explain the concept of quantum entanglement in simple terms."

    print(f"\n==================================================")
    print(f"Starting RLHF Demonstration with Ollama")
    print(f"Policy Model: {policy_model}")
    print(f"Critic Model: {critic_model}")
    print(f'User Prompt: "{user_prompt}"')
    print(f"==================================================\n")

    # Step 1: Generate multiple candidate responses from the policy model
    candidate_responses_raw = generate_candidate_responses(
        policy_model, user_prompt, num_responses=3
    )

    if not candidate_responses_raw:
        print("No candidate responses generated. Exiting RLHF example.")
        return

    scored_responses = []
    print("\n--- Evaluating Candidate Responses with Critic and User ---")
    for i, response_text in enumerate(candidate_responses_raw):
        print(f"\nProcessing Candidate Response #{i+1}:")

        # Get critic's score
        critic_score = get_reward_from_critic(critic_model, user_prompt, response_text)

        # Get user's score
        user_score = get_user_rating(response_text, critic_score, user_prompt)

        # Determine the final score for selection (user's score overrides critic's if provided)
        final_score = user_score if user_score is not None else critic_score

        scored_responses.append(
            {
                "response": response_text,
                "critic_score": critic_score,
                "user_score": (
                    user_score if user_score != critic_score else None
                ),  # Store user_score only if different
                "final_score": final_score,
            }
        )
        time.sleep(0.5)  # Small delay

    print("\n--- All Scored Responses (Critic and User) ---")
    for i, item in enumerate(scored_responses):
        print(
            f"Response #{i+1} (Critic: {item['critic_score']:.1f}, User: {item['user_score'] if item['user_score'] is not None else 'N/A'}, Final: {item['final_score']:.1f}):"
        )
        print(f"  {item['response'][:200]}...")  # Print first 200 chars
        print("-" * 20)

    # Step 3: Select the best response based on the highest final score
    best_response_info = None
    if scored_responses:
        best_response_info = max(scored_responses, key=lambda x: x["final_score"])
        print(f"\n==================================================")
        print(
            f"--- RLHF Chosen Best Response (Highest Final Score: {best_response_info['final_score']:.1f}) ---"
        )
        print(best_response_info["response"])
        print(f"==================================================\n")
    else:
        print("No responses were scored.")

    # Step 4: Generate the report
    report_data = {
        "policy_model": policy_model,
        "critic_model": critic_model,
        "user_prompt": user_prompt,
        "scored_responses": scored_responses,
        "best_response_info": best_response_info,
    }
    generate_report(report_data)

    print("\n--- Understanding the RLHF Process ---")
    print(
        "In a full RLHF system, the 'critic score' (reward) would be used by a Reinforcement Learning algorithm"
    )
    print("(e.g., PPO, DPO) to fine-tune the 'policy model' (llama3.2 in this case).")
    print(
        "The goal is to make the policy model generate responses that consistently achieve higher rewards."
    )
    print(
        "Human feedback is crucial for training the 'critic' or 'reward model' itself, teaching it what constitutes a 'good' response."
    )
    print(
        "This demonstration simulates the generation and evaluation steps, showing how feedback guides selection."
    )
    print(
        "Your manual ratings provide direct human feedback, influencing the 'best' choice in this simulation."
    )


if __name__ == "__main__":
    # You can specify different models if you have them pulled in Ollama
    # For this example, we use 'llama3.2' for both policy and critic.
    # If you have a smaller, faster model for critic, you might use that.
    run_rlhf_example(policy_model="llama3.2", critic_model="llama3.2")
