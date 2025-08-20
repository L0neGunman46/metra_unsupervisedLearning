import json
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# --- Configuration ---
# List of (environment_id, seed) tuples for which to plot data.
# Each entry here will correspond to a row in your final plot.
# The script will search for JSON files whose *content* matches these env_id/seed pairs.
RUNS_TO_PLOT = [
    ("Ant-v5", 42),
    # ("HalfCheetah-v5", 42),
    # Add more entries here, e.g., for Humanoid-v5 if you train it:
    # ("Humanoid-v5", 42),
    # ("Ant-v5", 123), # If you have runs with different seeds
]

# Which training step's data to use?
# 'latest': will find the latest step saved for each (env_id, seed) combination.
# Or specify a specific number like 100000 (if you know the step).
TRAINING_STEP_TO_PLOT = 'latest'

# Directory where trajectory JSON files are stored (e.g., "results/trajectories")
TRAJECTORY_DIR = "results/trajectories"
# Where to save the final combined plot
OUTPUT_FILE = "results/plots/figure13_style_trajectories.png"

# Axis limits for consistent plotting. Adjust based on your training results.
# These are examples, based roughly on paper's Figure 13 for states, but might need tuning.
PLOT_AXIS_LIMITS = {
    "Ant-v5": {
        "xy_xlim": (-60, 60), "xy_ylim": (-60, 60),
        "latent_xlim": (-300, 300), "latent_ylim": (-300, 300),
    },
    "HalfCheetah-v5": {
        "xy_xlim": (-120, 120), "xy_ylim": (-10, 10),
        "latent_xlim": (-3, 3), "latent_ylim": (0, 3.5),
    },
    "Humanoid-v5": {
        "xy_xlim": (-15, 15), "xy_ylim": (-15, 15),
        "latent_xlim": (-5, 5), "latent_ylim": (-5, 5),
    },
}

def load_and_validate_trajectory_data(filepath, expected_env_id, expected_seed):
    """Loads a JSON file and validates its environment ID and seed."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check if the data inside the JSON matches the expected env_id and seed
        if data.get("environment") == expected_env_id and data.get("seed") == expected_seed:
            return data
        else:
            print(f"  File {os.path.basename(filepath)} content mismatch: "
                  f"Expected ({expected_env_id}, {expected_seed}), "
                  f"Found ({data.get('environment')}, {data.get('seed')}). Skipping.")
            return None
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def find_latest_matching_old_trajectory_file(env_id, seed):
    """
    Finds the latest 'skills_step_X.json' file whose *content*
    matches the given environment ID and seed.
    """
    all_old_style_files = glob.glob(os.path.join(TRAJECTORY_DIR, "skills_step_*.json"))
    
    latest_matching_file = None
    latest_step = -1

    for filepath in all_old_style_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check if this file's content matches the desired env_id and seed
            if data.get("environment") == env_id and data.get("seed") == seed:
                current_step = data.get("training_step", -1)
                if current_step > latest_step:
                    latest_step = current_step
                    latest_matching_file = filepath
        except (FileNotFoundError, json.JSONDecodeError):
            continue # Skip corrupted or non-existent files
            
    return latest_matching_file

def plot_single_run_panel(ax_xy, ax_latent, data, env_id, axis_limits):
    """Plots x-y and latent trajectories for a single run on given axes."""
    skills = data.get("skills", [])
    if not skills:
        ax_xy.text(0.5, 0.5, 'No skill data', ha='center', va='center', transform=ax_xy.transAxes)
        ax_latent.text(0.5, 0.5, 'No skill data', ha='center', va='center', transform=ax_latent.transAxes)
        return

    num_skills = len(skills)
    colors = plt.cm.jet(np.linspace(0, 1, num_skills)) # Use 'jet' colormap as in paper

    for i, skill_data in enumerate(skills):
        xy = np.array(skill_data.get("xy_trajectory", []))
        latent = np.array(skill_data.get("latent_trajectory", []))

        # Plot x-y trajectory
        if xy.size > 0:
            ax_xy.plot(xy[:, 0], xy[:, 1], color=colors[i], linewidth=1.0)
            ax_xy.scatter(xy[0, 0], xy[0, 1], color=colors[i], marker='o', s=10) # Start
            ax_xy.scatter(xy[-1, 0], xy[-1, 1], color=colors[i], marker='x', s=15) # End
        
        # Plot latent trajectory (only if latent dim >= 2)
        if latent.size > 0 and latent.shape[1] >= 2:
            ax_latent.plot(latent[:, 0], latent[:, 1], color=colors[i], linewidth=1.0)
            ax_latent.scatter(latent[0, 0], latent[0, 1], color=colors[i], marker='o', s=10) # Start
            ax_latent.scatter(latent[-1, 0], latent[-1, 1], color=colors[i], marker='x', s=15) # End

    # Apply limits
    if env_id in axis_limits:
        ax_xy.set_xlim(axis_limits[env_id]["xy_xlim"])
        ax_xy.set_ylim(axis_limits[env_id]["xy_ylim"])
        ax_latent.set_xlim(axis_limits[env_id]["latent_xlim"])
        ax_latent.set_ylim(axis_limits[env_id]["latent_ylim"])

    ax_xy.grid(True, linestyle='--', alpha=0.5)
    ax_latent.grid(True, linestyle='--', alpha=0.5)
    ax_xy.set_aspect('equal', adjustable='box')
    ax_latent.set_aspect('equal', adjustable='box')

def main():
    """Generates and saves the Figure 13-style trajectory plot."""
    num_rows = len(RUNS_TO_PLOT)
    num_cols = 2 # x-y trajectories, phi(s) trajectories

    # figsize is heuristic: 4 inches per plot, plus some padding
    fig, axes = plt.subplots(
        num_rows, 
        num_cols, 
        figsize=(num_cols * 4 + 1, num_rows * 4 + 1), 
        squeeze=False # Ensure axes is always 2D
    )
    
    fig.suptitle(f"METRA Latent Space Visualization", fontsize=16, y=1.02)
    print("Generating Figure 13-style visualization...")

    for row_idx, (env_id, seed) in enumerate(RUNS_TO_PLOT):
        filepath = None
        data = None

        if TRAINING_STEP_TO_PLOT == 'latest':
            # Find the latest file that *contains* the correct env_id and seed
            filepath = find_latest_matching_old_trajectory_file(env_id, seed)
            if filepath:
                data = load_and_validate_trajectory_data(filepath, env_id, seed)
            else:
                print(f"  No 'skills_step_*.json' found matching Env: {env_id}, Seed: {seed}. Skipping.")

        else: # Specific training step provided (e.g., 100000)
            filepath = os.path.join(
                TRAJECTORY_DIR, 
                f"skills_step_{TRAINING_STEP_TO_PLOT}.json" # Old naming convention
            )
            data = load_and_validate_trajectory_data(filepath, env_id, seed)
        
        ax_xy = axes[row_idx, 0]
        ax_latent = axes[row_idx, 1]

        if data:
            print(f"  Plotting from: {os.path.basename(filepath)}")
            plot_single_run_panel(ax_xy, ax_latent, data, env_id, PLOT_AXIS_LIMITS)
        else:
            ax_xy.text(0.5, 0.5, 'Data N/A', ha='center', va='center', transform=ax_xy.transAxes, fontsize=10)
            ax_latent.text(0.5, 0.5, 'Data N/A', ha='center', va='center', transform=ax_latent.transAxes, fontsize=10)

        # Set row labels (Environment ID and Seed)
        if row_idx == 0:
            ax_xy.set_title("x-y trajectories", fontsize=12)
            ax_latent.set_title("φ(s) trajectories", fontsize=12)
        
        ax_xy.set_ylabel(f"{env_id}\n(Seed {seed})", fontsize=10, weight='bold')
        
        # Hide internal x-y labels and titles for clarity in a grid
        ax_xy.set_xlabel('')
        ax_latent.set_xlabel('')
        ax_xy.set_ylabel('') # Redundant since we use row label
        ax_latent.set_ylabel('')


    plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.5)
    
    # Save the final plot
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Combined plot saved successfully to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()