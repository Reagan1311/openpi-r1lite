import dataclasses
import enum
import logging
import pathlib
import time
import datetime
import os
import cv2
import requests
import base64
import io
from PIL import Image

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

from examples.r1lite.robot_env import RobotEnv

logger = logging.getLogger(__name__)

# Specifying the task for robot control
PROMPTS = ''

# Robot control mode: 'eepose' for end-effector pose, 'joint' for joint angles
CONTROL_MODE = 'eepose'

# Directory to save periodic image logs
LOG_IMAGE_DIR = "./log_images_r1lite_ros2"

# Mapping from server-expected camera names to environment camera names
CAMERA_MAPPING = {
    "cam_head": "image_right",
    "cam_left_wrist": "image_left_wrist",
    "cam_right_wrist": "image_right_wrist",
}

# --- HELPER FUNCTIONS ---

def encode_image(img: np.ndarray) -> str:
    """Encodes a numpy array (BGR) into a base64 PNG string."""
    # Convert BGR (from OpenCV) to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def setup_logging_directory():
    """Creates the logging directory if it doesn't exist."""
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"Created log directory: {LOG_IMAGE_DIR}")

# ------------------------

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    R1LITE = 'r1lite'


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "192.168.31.160"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 8000
    # API key to use for the server.
    api_key: str | None = None
    # Number of steps to run the policy for.
    num_steps: int = 20
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None
    # Environment to run the policy in.
    env: EnvMode = EnvMode.ALOHA_SIM


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def main(args: Args) -> None:
    """Main function to connect to the robot, run the control loop, and handle shutdown."""
    setup_logging_directory()
    env = None
    
    try:
        # --- A. INITIALIZATION ---
        print("Initializing robot environment (ROS2)...")
        env = RobotEnv()
        time.sleep(2) # Wait for environment to stabilize
        print("Initialization complete!")

        # --- B. MAIN CONTROL LOOP ---
        print("Entering main control loop...")
        while True:
            # --- i. Get Observations ---
            print("\n" + "="*50)
            print("1. Gathering robot state and images...")

            frames, state = env.update_obs_window()
            if state is None or not frames:
                print("Warning: No state or image data received. Skipping this cycle.")
                time.sleep(1)
                continue
            eef_pose_state = state.get('eef_pose')
            # qpos_state = state.get('qpos')

            # --- ii. Prepare Data for Server ---
            print("2. Preparing data for inference server...")
            encoded_images = {}
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for server_name, env_name in CAMERA_MAPPING.items():
                img = frames.get(env_name)
                if img is not None and img.size > 0:
                    encoded_images[server_name] = encode_image(img)
                    log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}.png")
                    cv2.imwrite(log_path, img)
                else:
                    print(f"Warning: Image for '{server_name}' is not available.")
            
            # random observation for testing
            obs_fn = {
                EnvMode.ALOHA: _random_observation_aloha,
                EnvMode.ALOHA_SIM: _random_observation_aloha,
                EnvMode.DROID: _random_observation_droid,
                EnvMode.LIBERO: _random_observation_libero,
                EnvMode.R1LITE: _random_observation_r1lite,
            }[args.env]
            
            # obs_fn = {
            #     "state": eef_pose_state,
            #     "images": encoded_images,
            #     "prompt": PROMPTS,
            # }

            # --- iii. Send Request to Server ---
            print(f"3. Sending request to server {args.host}...")
            try:
                policy = _websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                    api_key=args.api_key,
                )
                logger.info(f"Server metadata: {policy.get_server_metadata()}")
            except requests.exceptions.RequestException as e:
                print(f"Error communicating with server: {e}. Retrying after 5s.")
                time.sleep(5)
                continue

            # Send a few observations to make sure the model is loaded.
            for _ in range(2):
                policy.infer(obs_fn())
                
            timing_recorder = TimingRecorder()

            for _ in tqdm.trange(args.num_steps, desc="Running policy"):
                inference_start = time.time()
                action = policy.infer(obs_fn())
                timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
                for key, value in action.get("server_timing", {}).items():
                    timing_recorder.record(f"server_{key}", value)
                for key, value in action.get("policy_timing", {}).items():
                    timing_recorder.record(f"policy_{key}", value)

            timing_recorder.print_all_stats()

            if args.timing_file is not None:
                timing_recorder.write_parquet(args.timing_file)
                
            # --- iv. Parse and Execute Actions ---
            print("4. Parsing and executing actions...")
            actions = action.get('actions')
            if not actions:
                print("No actions received from the model. Skipping.")
                continue

            for i, act in enumerate(np.array(actions)):
                action = np.array(act, dtype=np.float32)
                print(f"[Step {i+1}/{len(actions)}] Executing action: {np.round(action, 3)}")
                
                if CONTROL_MODE == 'eepose':
                    env.control_eef(action)
                elif CONTROL_MODE == 'joint':
                    env.control(action)
                    
                time.sleep(0.05) # Short pause between actions
            
            print("Action sequence execution complete.")
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # --- C. SHUTDOWN ---
        # if env:
        #     env.shutdown()
        print("Program finished.")


def _random_observation_r1lite() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_head": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
