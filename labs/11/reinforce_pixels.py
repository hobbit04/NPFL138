#!/usr/bin/env python3
import argparse

import gymnasium as gym
import numpy as np
import torch

import npfl138
npfl138.require_version("2526.11")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=5, type=int, help="Number of episodes per batch.")
parser.add_argument("--episodes", default=1500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="cart_pole_pixels.pt", type=str, help="Path to save/load model.")


class Agent:
    # Use an accelerator if available.
    device = npfl138.trainable_module.get_auto_device()

    def __init__(self, env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create a suitable model of the policy. Note that the shape
        # of the observations is available in `env.observation_space.shape`
        # and the number of actions in `env.action_space.n`.
        #
        # Apart from the policy network defined in `reinforce` assignment, you
        # also need a value network for computing the baseline, returning
        # a single output with no activation.
        #
        # Using Adam optimizer for both models is a good default.
        def make_layers(out_features):
            return [
                torch.nn.Conv2d(3, 16, kernel_size=8, stride=4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, kernel_size=4, stride=2),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(32 * 8 * 8, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, out_features),
            ]

        self._policy = torch.nn.Sequential(
            *make_layers(env.action_space.n),
            torch.nn.Softmax(dim=-1),
        ).to(self.device)

        self._baseline = torch.nn.Sequential(*make_layers(1)).to(self.device)

        self._policy_optimizer = torch.optim.Adam(self._policy.parameters(), lr=args.learning_rate)
        self._baseline_optimizer = torch.optim.Adam(self._baseline.parameters(), lr=args.learning_rate)

        self._policy_loss = torch.nn.NLLLoss(reduction="none")
        self._baseline_loss = torch.nn.MSELoss()
    
    def save(self, path: str) -> None:
        torch.save({
            "policy": self._policy.state_dict(),
            "baseline": self._baseline.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device)
        self._policy.load_state_dict(data["policy"])
        self._baseline.load_state_dict(data["baseline"])

    def _preprocess(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() == 3:
            states = states.unsqueeze(0)  
        return states.permute(0, 3, 1, 2) / 255.0

    # The `npfl138.rl_utils.typed_torch_function` automatically converts input arguments
    # to PyTorch tensors of given type, and converts the result to a NumPy array.
    @npfl138.rl_utils.typed_torch_function(device, torch.float32, torch.int64, torch.float32)
    def train(self, states: torch.Tensor, actions: torch.Tensor, returns: torch.Tensor) -> None:
        # TODO: Perform training.
        # You should:
        # - compute the predicted baseline using the baseline model,
        # - train the policy model, using `returns` - `predicted_baseline` as
        #   advantage estimate,
        # - train the baseline model to predict `returns`.
        #
        # Note that predicting returns in 0-500 range is challenging for the network, given
        # that the default initialization tries to keep variance -- it might be helpful for
        # the network if you predict returns in a smaller range.
        states = self._preprocess(states)  # (N, C, H, W)

        baseline = self._baseline(states).squeeze(-1)
        advantage = returns - baseline.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        self._policy_optimizer.zero_grad()
        probs = self._policy(states)
        log_probs = torch.log(probs + 1e-8)
        per_sample_loss = self._policy_loss(log_probs, actions)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        policy_loss = (per_sample_loss * advantage).mean() - 0.01 * entropy
        policy_loss.backward()
        self._policy_optimizer.step()

        self._baseline_optimizer.zero_grad()
        baseline = self._baseline(states).squeeze(-1)
        baseline_loss = self._baseline_loss(baseline, returns)
        baseline_loss.backward()
        self._baseline_optimizer.step()

    @npfl138.rl_utils.typed_torch_function(device, torch.float32)
    def predict(self, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            state = self._preprocess(state)
            return self._policy(state).squeeze(0)

def main(env: npfl138.rl_utils.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    # Construct the agent.
    agent = Agent(env, args)

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        # TODO: Load the agent
        agent.load(args.model_path)

        # Final evaluation
        while True:
            state, done = env.reset(options={"start_evaluation": True})[0], False
            while not done:
                # TODO: Choose a greedy action.
                probs = agent.predict(state)
                action = np.argmax(probs)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

    # TODO: Perform training
    for episode_block in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []

        for _ in range(args.batch_size):
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            while not done:
                probs = agent.predict(state)
                action = np.random.choice(len(probs), p=probs)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # Compute discounted returns
            G = 0
            returns = []
            for reward in reversed(rewards):
                G = reward + args.gamma * G
                returns.insert(0, G)

            batch_states += states
            batch_actions += actions
            batch_returns += returns

        agent.train(batch_states, batch_actions, batch_returns)

    # Save the trained model
    agent.save(args.model_path)



if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    main_env = npfl138.rl_utils.EvaluationEnv(
        gym.make("npfl138/CartPolePixels-v1"), main_args.seed, main_args.render_each)

    main(main_env, main_args)
