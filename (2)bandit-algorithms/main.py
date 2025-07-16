import ollama
import numpy as np
import math
import re  # For simple tokenization in evaluate_response
from import_debug import install_package


class MultiArmedBandit:
    """
    Implements the UCB1 Multi-Armed Bandit algorithm.
    """

    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.counts = np.zeros(
            num_arms
        )  # N_i(t) - number of times arm i has been pulled
        self.values = np.zeros(num_arms)  # S_i(t) - sum of rewards for arm i
        self.total_pulls = 0  # Total number of pulls across all arms

    def select_arm(self):
        """
        Selects an arm using the UCB1 algorithm.
        """
        # Play each arm once initially to get an estimate
        for arm in range(self.FIXME):
            if self.counts[FIXME] == 0:
                return arm

        # Calculate UCB1 value for each arm
        ucb_values = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            # Q_i(t) - estimated value of arm i
            average_reward = self.values[FIXME] / self.counts[FIXME]
            # UCB1 exploration term
            exploration_term = math.sqrt(
                2 * math.log(self.total_pulls) / self.counts[arm]
            )
            ucb_values[arm] = average_reward + exploration_term

        # Select the arm with the highest UCB1 value
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        """
        Updates the bandit's counts and values after pulling an arm and observing a reward.
        """
        self.counts[chosen_arm] += 1
        self.total_pulls += 1
        self.values[chosen_arm] += FIXME  # Sum of rewards


class LLMPromptOptimizer:
    """
    Optimizes LLM prompts using a Multi-Armed Bandit approach.
    """

    def __init__(self, ollama_model: str, prompt_variants: list, ground_truth: str):
        self.ollama_model = FIXME
        self.prompt_variants = FIXME
        self.ground_truth = FIXME
        self.num_prompts = len(prompt_variants)
        self.bandit = MultiArmedBandit(self.num_prompts)

        # Basic text processing for reward calculation
        self.ground_truth_tokens = self._tokenize(ground_truth.lower())

    def _tokenize(self, text):
        """Simple tokenization by splitting on non-alphanumeric characters."""
        return set(re.findall(r"\b\w+\b", text))

    def get_llm_response(self, prompt: str) -> str:
        """
        Connects to Ollama and generates a response for a given prompt.
        """
        try:
            # Set the temperature to 0.0 for deterministic output during evaluation
            response_data = ollama.generate(
                model=self.ollama_model, prompt=prompt, options={"temperature": FIXME}
            )
            return response_data["response"].strip()
        except ollama.ResponseError as e:
            print(f"Ollama API Error: {e}")
            print(
                "Please ensure the Ollama server is running and the model is available."
            )
            return ""
        except Exception as e:
            print(f"An unexpected error occurred during LLM response generation: {e}")
            return ""

    def evaluate_response(self, llm_response: str) -> float:
        """
        Simulated reward function: Calculates a score based on token overlap
        with a predefined ground truth. Higher overlap means higher reward.
        """
        if not FIXME:
            return 0.0

        response_tokens = self._tokenize(llm_response.lower())

        # Calculate overlap (Jaccard similarity style)
        intersection = len(self.ground_truth_tokens.intersection(response_tokens))
        union = len(self.ground_truth_tokens.union(response_tokens))

        if union == 0:  # Avoid division by zero if both are empty
            return 0.0

        # Normalize reward to be between 0 and 1
        reward = intersection / union
        return reward

    def run_optimization_step(self):
        """
        Runs a single step of the MAB optimization: selects a prompt,
        gets LLM response, calculates reward, and updates the bandit.
        """
        chosen_prompt_index = self.bandit.FIXME()
        chosen_prompt = self.prompt_variants[chosen_prompt_index]

        print(f"\n--- Trial {self.bandit.total_pulls + 1} ---")
        print(f'Chosen Prompt (Arm {chosen_prompt_index}): "{chosen_prompt}"')

        llm_response = self.FIXME(chosen_prompt)
        print(f'LLM Response: "{llm_response[:100]}..."')  # Print first 100 chars

        reward = self.FIXME(llm_response)
        self.bandit.update(chosen_prompt_index, reward)

        print(f"Reward: {reward:.4f}")
        return chosen_prompt_index, llm_response, reward

    def get_best_prompt(self):
        """
        Returns the prompt variant with the highest estimated average reward.
        """
        best_arm_index = np.argmax(
            self.bandit.values / (self.bandit.counts + 1e-9)
        )  # Add epsilon to avoid div by zero
        return self.prompt_variants[best_arm_index], best_arm_index


if __name__ == "__main__":

    # Ensure ollama is installed
    install_package("ollama")

    OLLAMA_MODEL = FIXME  # Ensure this model is pulled in Ollama

    # Define various prompt strategies (arms)
    prompt_variants = [
        "Explain the concept of photosynthesis simply.",
        "What is photosynthesis? Provide a concise explanation.",
        "Could you briefly describe photosynthesis?",
        "Tell me about photosynthesis in easy terms for a 10-year-old.",
        "Photosynthesis: elaborate on its core principles.",
    ]

    # Ground truth for reward calculation (simplified for this example)
    # In a real system, this would be a human-curated ideal answer or a robust evaluation method.
    ground_truth_answer = "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll pigments. It converts light energy into chemical energy."

    print("--- Starting LLM Multi-Armed Bandit Prompt Optimization ---")

    optimizer = LLMPromptOptimizer(
        ollama_model=OLLAMA_MODEL,
        prompt_variants=prompt_variants,
        ground_truth=ground_truth_answer,
    )

    num_trials = 50  # Number of times we "pull" an arm (try a prompt)

    for i in range(num_trials):
        optimizer.run_optimization_step()

    print("\n--- Optimization Complete ---")
    print(f"Total trials: {optimizer.bandit.total_pulls}")
    print("\nArm Statistics:")
    for i, prompt in enumerate(optimizer.prompt_variants):
        avg_reward = optimizer.bandit.values[i] / (
            optimizer.bandit.counts[i] + 1e-9
        )  # Avoid div by zero
        print(f"  Arm {i} (Prompt: '{prompt}'):")
        print(f"    Pulls: {int(optimizer.bandit.counts[i])}")
        print(f"    Average Reward: {avg_reward:.4f}")
        print(f"    Total Reward: {optimizer.bandit.values[i]:.4f}")

    best_prompt, best_index = optimizer.get_best_prompt()
    print(f"\nBest performing prompt (Arm {best_index}):")
    print(f"'{best_prompt}'")
    print(
        f"Estimated Average Reward: {optimizer.bandit.values[best_index] / (optimizer.bandit.counts[best_index] + 1e-9):.4f}"
    )

    # Optionally, run the best prompt one more time to see its output
    print("\n--- Final Test with Best Prompt ---")
    final_response = optimizer.get_llm_response(best_prompt)
    print(f"Prompt: '{best_prompt}'")
    print(f"Response: '{final_response}'")
    print(
        f"Final Reward (against ground truth): {optimizer.evaluate_response(final_response):.4f}"
    )
