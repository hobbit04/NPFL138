import gym_cartpole
import argparse
import itertools
import os
import time

def tune():
    # Focused search space (12 combinations)
    lrs = [0.001, 0.005]
    batch_sizes = [32, 64]
    h1s = [64, 128, 256]
    
    # Static parameters
    h2 = 64
    epochs = 100
    eval_episodes = 20  # Reduced for speed

    best_score = -1
    best_params = None
    
    combinations = list(itertools.product(lrs, batch_sizes, h1s))
    total = len(combinations)
    print(f"Starting simple tuning (max 5m). Combinations: {total}")
    
    start_time = time.time()
    timeout = 5 * 60  # 5 minutes

    for i, (lr, batch_size, h1) in enumerate(combinations):
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"\n[Timeout] 5 minutes reached. Terminating.")
            break

        print(f"\n[{i+1}/{total}] {elapsed:.0f}s elapsed. Testing: lr={lr}, bs={batch_size}, h1={h1}")
        
        args = argparse.Namespace(
            evaluate=False,
            recodex=False,
            render=False,
            seed=42,
            threads=4,
            batch_size=batch_size,
            epochs=epochs,
            h1=h1,
            h2=h2,
            lr=lr,
            eval_episodes=eval_episodes,
            model="tuning_model.pt"
        )
        
        try:
            score = gym_cartpole.main(args)
            print(f"Score: {score}")
            
            if score > best_score:
                best_score = score
                best_params = {"lr": lr, "batch_size": batch_size, "h1": h1}
                print(f"*** NEW BEST ***")
                if os.path.exists("tuning_model.pt"):
                    os.replace("tuning_model.pt", "best_gym_cartpole_model.pt")
                if os.path.exists("tuning_model.pt.json"):
                    os.replace("tuning_model.pt.json", "best_gym_cartpole_model.pt.json")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*30)
    if best_params:
        print(f"Best Score (20 eps): {best_score}")
        print(f"Best Params: {best_params}")
    else:
        print("No results.")
    print("="*30)

if __name__ == "__main__":
    tune()
