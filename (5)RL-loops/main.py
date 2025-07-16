import ollama
import numpy as np
import matplotlib.pyplot as plt
import time  # For simulating delay or for future use
from import_debug import install_package


# Ensure required packages are installed
install_package("ollama")
install_package("numpy")
install_package("matplotlib")
# gymnasium is not used in this simplified iterative self-refinement approach,
# but can be integrated for more formal RL setups.


class LLMOptimizerEnvironment:
    """
    Simulates an environment for optimizing LLM output using an RL-like loop.
    The "agent" here is the LLM itself, iteratively refining its output.
    It takes actions by applying "tools" (refinement instructions) and
    receives rewards based on the quality of its generated text.
    """

    def __init__(self, model_name="llama3.2", initial_prompt="Why is the sky blue?"):
        """
        Initializes the LLM optimization environment.

        Args:
            model_name (str): The name of the Ollama model to use (e.g., 'llama3.2').
            initial_prompt (str): The starting prompt for the LLM.
        """
        self.model_name = model_name
        self.initial_prompt = initial_prompt
        self.current_output = ""
        self.history = []  # Stores (iteration, output, reward) tuples for analysis
        self.iteration = 0

        # Define "tools" as specific instructions for the LLM to refine its output.
        # These are the "actions" the LLM agent can take in this simplified RL loop.
        self.tools = {
            "concise": "Make this answer more concise and to the point.",
            "expand_scientific": "Elaborate on the scientific principles involved.",
            "simple_explanation": "Rephrase this for a general audience, using simpler terms.",
            "add_example": "Provide a simple, relatable example to illustrate the concept.",
        }
        self.tool_keys = list(
            self.tools.keys()
        )  # Get a list of tool names for easy iteration

        print(f"Initialized LLM Optimizer Environment with model: {self.model_name}")

    def _generate_llm_response(self, prompt_text, stream=False):
        """
        Helper function to interact with the Ollama LLM.

        Args:
            prompt_text (str): The prompt to send to the LLM.
            stream (bool): Whether to stream the response (True) or get it all at once (False).

        Returns:
            str: The generated response from the LLM, or an error message if an issue occurs.
        """
        try:
            if stream:
                full_response = ""
                # Use temperature 0.0 for more deterministic output, useful for optimization
                stream_gen = ollama.generate(
                    model=self.model_name,
                    prompt=prompt_text,
                    stream=True,
                    options={"temperature": FIXME},
                )
                for chunk in stream_gen:
                    full_response += chunk["response"]
                return full_response
            else:
                # Use temperature 0.0 for more deterministic output
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt_text,
                    options={"temperature": FIXME},
                )
                return response["response"]
        except ollama.ResponseError as e:
            print(f"Ollama API Error during generation: {e}")
            print(
                "Please ensure the Ollama server is running and the model is available."
            )
            return f"ERROR: {e}"
        except Exception as e:
            print(f"An unexpected error occurred during generation: {e}")
            return f"ERROR: {e}"

    def reset(self):
        """
        Resets the environment for a new optimization run.
        Generates the initial response and calculates its reward.

        Returns:
            tuple: (initial_output, initial_reward, done_status, info_dict)
        """
        print(f"\n--- Resetting Environment ---")
        self.iteration = 0
        self.history = []
        # Initial generation based on the original prompt
        print(f"Initial generation for prompt: '{self.initial_prompt}'")
        self.current_output = self._generate_llm_response(self.initial_prompt)
        initial_reward = self._calculate_reward(self.current_output)
        self.history.append((self.iteration, self.current_output, initial_reward))
        print(
            f"Initial Output:\n{self.current_output}\nInitial Reward: {initial_reward:.2f}"
        )
        # In this setup, 'done' is usually managed by the external loop's max_iterations
        return (
            self.current_output,
            initial_reward,
            False,
            {},
        )  # observation, reward, done, info

    def _calculate_reward(self, text):
        """
        Calculates a reward for the generated text based on predefined criteria.
        This is a simplified, rule-based reward function.
        *** IMPORTANT: You MUST customize this function based on your specific
        optimization goals for the LLM's output. ***

        For "Why is the sky blue?":
        - Rewards for mentioning key scientific terms (Rayleigh scattering, blue light, etc.).
        - Penalizes excessive length (encourages conciseness).
        - Penalizes generic or error messages.

        Args:
            text (str): The LLM generated text to evaluate.

        Returns:
            float: The calculated reward.
        """
        reward = 0.0
        text_lower = text.lower()

        # 1. Reward for conciseness (penalize long answers)
        word_count = len(text.split())
        if word_count < FIXME:
            reward += 0.5  # Bonus for being concise
        elif word_count > FIXME:
            reward -= 0.5  # Penalty for being too verbose
        else:
            reward += 0.1  # Small bonus for reasonable length

        # 2. Reward for mentioning key scientific terms for "Why is the sky blue?"
        if "rayleigh scattering" in text_lower:
            reward += FIXME
        if "blue light" in text_lower and "scatter" in text_lower:
            reward += FIXME
        if "wavelength" in text_lower:
            reward += FIXME
        if "atmosphere" in text_lower:
            reward += FIXME
        if "molecules" in text_lower or "particles" in text_lower:
            reward += FIXME

        # 3. Penalize generic or incorrect answers / Ollama errors
        if (
            "error" in text_lower
            or "problem" in text_lower
            or "not found" in text_lower
        ):
            reward -= FIXME  # Significant penalty for generation errors
        if "sun" not in text_lower and "light" not in text_lower:
            reward -= FIXME  # Basic components of the explanation missing

        # Normalize and clip reward to a reasonable range (e.g., -2.0 to 2.0)
        # This helps keep rewards consistent and prevents extreme values.
        return np.clip(reward, -2.0, 2.0)

    def step(self, tool_key):
        """
        Takes a "step" in the environment by applying a chosen tool to refine the output.
        This simulates the LLM "taking an action" to improve its state.

        Args:
            tool_key (str): The key of the tool to apply (e.g., "concise", "expand_scientific").

        Returns:
            tuple: (new_output, reward, done, info)
                - new_output (str): The LLM's response after applying the tool.
                - reward (float): The reward for the new_output.
                - done (bool): Always False in this continuous refinement loop.
                - info (dict): Additional information about the step.
        """
        if tool_key not in self.tools:
            print(f"Warning: Tool '{tool_key}' not recognized. Skipping step.")
            # Return current state with no change and a penalty reward
            return (
                self.current_output,
                self._calculate_reward(self.current_output) - 1.0,
                False,
                {"message": "Invalid tool"},
            )

        tool_instruction = self.tools[tool_key]
        print(f"\n--- Applying Tool: '{tool_key}' ---")
        print(f"Instruction: '{tool_instruction}'")

        # Construct the prompt to guide the LLM to use the "tool."
        # We explicitly ask the LLM to refine its *previous* output based on the tool instruction.
        refinement_prompt = (
            f"Given the following previous answer to the question '{self.initial_prompt}':\n\n"
            f"'{self.current_output}'\n\n"
            f"Please {tool_instruction} Provide the refined answer only, without any conversational preamble."
        )

        new_output = self._generate_llm_response(refinement_prompt)
        new_reward = self._calculate_reward(new_output)

        self.current_output = new_output  # Update the environment's current state
        self.iteration += 1
        self.history.append((self.iteration, self.current_output, new_reward))

        done = False  # In this iterative setup, the environment is never "done" unless an external limit is hit.
        info = {"tool_applied": tool_key, "instruction": tool_instruction}

        print(f"Refined Output:\n{self.current_output}\nNew Reward: {new_reward:.2f}")
        return self.current_output, new_reward, done, info


def run_rl_optimization_loop(env, max_iterations=5):
    """
    Runs an RL-like optimization loop for the LLM.
    This function implements a simple heuristic "policy" to select tools.

    Args:
        env (LLMOptimizerEnvironment): The environment instance.
        max_iterations (int): The maximum number of refinement steps to perform.

    Returns:
        tuple: (rewards_over_time, outputs_over_time, full_history)
            - rewards_over_time (list): List of rewards at each iteration.
            - outputs_over_time (list): List of outputs at each iteration.
            - full_history (list): Detailed history of (iteration, output, reward).
    """
    print("\n--- Starting RL Optimization Loop ---")
    rewards_over_time = []
    outputs_over_time = []

    # Reset the environment to get the initial state and reward
    observation, reward, done, info = env.reset()
    rewards_over_time.append(reward)
    outputs_over_time.append(observation)

    # heuristic policy for selecting the next tool, tries to adapt based on the current reward and output characteristics.
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1}/{max_iterations} ---")
        current_reward = env.history[-1][2]
        best_tool_for_this_step = None
        current_word_count = len(env.current_output.split())

        if (
            current_reward < FIXME
        ):  # If reward is relatively low, try to add more scientific detail
            best_tool_for_this_step = "expand_scientific"
        elif (
            current_reward >= FIXME and current_word_count > FIXME
        ):  # If reward is decent but verbose, try to be concise
            best_tool_for_this_step = "concise"
        elif (
            current_reward >= FIXME and "example" not in env.current_output.lower()
        ):  # If already good, try adding an example
            best_tool_for_this_step = "add_example"
        else:  # Fallback: cycle through tools if no specific condition met
            best_tool_for_this_step = env.tool_keys[i % len(env.tool_keys)]

        # Ensure a tool is always selected
        if best_tool_for_this_step is None:
            best_tool_for_this_step = env.tool_keys[i % len(env.tool_keys)]

        # Take a step in the environment with the selected tool
        new_observation, new_reward, done, info = env.step(best_tool_for_this_step)
        rewards_over_time.append(new_reward)
        outputs_over_time.append(new_observation)

        if (
            done
        ):  # Check if the environment signalled completion (unlikely in this setup)
            print("Environment finished early.")
            break

    print("\n--- Optimization Loop Finished ---")
    return rewards_over_time, outputs_over_time, env.history


def plot_rewards(rewards, title="Reward Over Optimization Iterations"):
    """
    Plots the rewards obtained over iterations using matplotlib.

    Args:
        rewards (list): A list of reward values for each iteration.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(rewards)), rewards, marker="o", linestyle="-", color="skyblue")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.xticks(range(len(rewards)))  # Ensure all iterations are marked on the x-axis
    plt.show()


if __name__ == "__main__":
    print("Starting LLM RL Optimization Example.")
    print("Please ensure Ollama is running and 'llama3.2' model is available.")
    print("You can pull the model by running: `ollama pull llama3.2` in your terminal.")

    # --- Example 1: Optimizing "Why is the sky blue?" for conciseness and scientific accuracy ---
    llm_env = LLMOptimizerEnvironment(
        model_name="llama3.2", initial_prompt="Why is the sky blue?"
    )
    # Run the optimization loop for a few iterations
    rewards_sky_blue, outputs_sky_blue, history_sky_blue = run_rl_optimization_loop(
        llm_env, max_iterations=FIXME
    )

    print("\n--- Final Outputs for 'Why is the sky blue?' Optimization ---")
    for i, (iter_num, output, reward) in enumerate(history_sky_blue):
        print(f"\n--- Iteration {iter_num} (Reward: {reward:.2f}) ---")
        print(output)

    # Plot the rewards to visualize the optimization process
    plot_rewards(rewards_sky_blue, "Reward for 'Why is the sky blue?' Optimization")
