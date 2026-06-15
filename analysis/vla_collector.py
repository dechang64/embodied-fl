# ── python/analysis/vla_collector.py ──
"""
VLA Data Collector for Embodied Intelligence
=============================================
Collects Vision-Language-Action episodes from various robot platforms
and converts them into a unified RLDS-compatible format.

Supported sources:
  - ROS2 bag files (rosbag2)
  - Isaac Sim recordings (HDF5)
  - Raw teleoperation logs (JSON/CSV)
  - Synthetic data generators (for testing)

Output format (per episode):
  {
    "episode_id": "ep_001",
    "steps": [
      {
        "observation": {
          "image": "<base64 or file path>",
          "depth": "<base64 or file path>",       // optional
          "robot_state": [q1, q2, ..., q7],       // joint angles (rad)
          "gripper": 0.8,                         // gripper aperture (0=closed, 1=open)
          "end_effector_pose": [x, y, z, rx, ry, rz],  // optional
        },
        "instruction": "pick up the red cup",
        "action": {
          "target_joint_pos": [q1, ..., q7],      // target joint angles
          "gripper_command": 0.0,                  // gripper target
          "delta_pose": [dx, dy, dz, drx, dry, drz],  // optional, relative
        },
        "reward": 0.0,
        "is_terminal": false,
        "is_success": false,
      },
      ...
    ],
    "metadata": {
      "robot_type": "franka_panda",
      "task_type": "grasping",
      "language": "zh",
      "duration_sec": 12.5,
      "num_steps": 50,
      "source": "ros2_bag",
    }
  }

Bridge to Rust:
  Collected episodes → serialized → gRPC UploadEpisode → Rust stores + indexes
"""

import json
import os
import time
import hashlib
import base64
import numpy as np
from pathlib import Path
from typing import Optional, Union, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime


# ── Data Structures ──

@dataclass
class Observation:
    """Single observation snapshot from robot sensors."""
    image: Optional[str] = None          # base64 or file path
    depth: Optional[str] = None          # base64 or file path
    robot_state: list[float] = field(default_factory=list)   # joint angles
    gripper: float = 0.5                 # 0=closed, 1=open
    end_effector_pose: list[float] = field(default_factory=list)  # [x,y,z,rx,ry,rz]

    def to_dict(self) -> dict:
        return asdict(self)

    def get_embedding_input(self) -> dict:
        """Return fields needed for embedding (image + state)."""
        return {
            "image": self.image,
            "robot_state": self.robot_state,
            "gripper": self.gripper,
        }


@dataclass
class Action:
    """Single action command for robot."""
    target_joint_pos: list[float] = field(default_factory=list)
    gripper_command: float = 0.5
    delta_pose: list[float] = field(default_factory=list)  # optional relative motion

    def to_dict(self) -> dict:
        return asdict(self)

    def to_flat(self) -> list[float]:
        """Flatten action into a single vector."""
        vec = list(self.target_joint_pos) + [self.gripper_command]
        if self.delta_pose:
            vec.extend(self.delta_pose)
        return vec


@dataclass
class Step:
    """Single timestep in an episode."""
    observation: Observation
    instruction: str = ""
    action: Optional[Action] = None
    reward: float = 0.0
    is_terminal: bool = False
    is_success: bool = False
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        d = {
            "observation": self.observation.to_dict(),
            "instruction": self.instruction,
            "action": self.action.to_dict() if self.action else None,
            "reward": self.reward,
            "is_terminal": self.is_terminal,
            "is_success": self.is_success,
        }
        if self.timestamp:
            d["timestamp"] = self.timestamp
        return d


@dataclass
class Episode:
    """Complete robot episode (sequence of steps)."""
    episode_id: str = ""
    steps: list[Step] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.episode_id:
            self.episode_id = f"ep_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }

    def to_rlds(self) -> dict:
        """Convert to RLDS-compatible format.

        RLDS structure:
          episode: {
            steps: {
              observation: {image, depth, robot_state, gripper},
              instruction: str,
              action: {target_joint_pos, gripper_command, delta_pose},
              reward: float,
              is_terminal: bool,
              is_first: bool,
              is_last: bool,
            }
          }
        """
        rlds_steps = []
        for i, step in enumerate(self.steps):
            rlds_step = step.to_dict()
            rlds_step["is_first"] = (i == 0)
            rlds_step["is_last"] = (i == len(self.steps) - 1)
            rlds_steps.append(rlds_step)

        return {
            "episode_id": self.episode_id,
            "steps": rlds_steps,
            "metadata": self.metadata,
        }

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def action_dim(self) -> int:
        """Infer action dimensionality from first step."""
        if self.steps and self.steps[0].action:
            return len(self.steps[0].action.to_flat())
        return 0

    @property
    def state_dim(self) -> int:
        """Infer robot state dimensionality from first step."""
        if self.steps:
            return len(self.steps[0].observation.robot_state)
        return 0

    def get_success_rate(self) -> float:
        """Return fraction of successful episodes (0 or 1 for single episode)."""
        return 1.0 if self.steps and self.steps[-1].is_success else 0.0

    def get_avg_reward(self) -> float:
        """Return average reward across steps."""
        if not self.steps:
            return 0.0
        return sum(s.reward for s in self.steps) / len(self.steps)


# ── Collectors ──

class BaseCollector:
    """Base class for VLA data collectors."""

    def __init__(self, robot_type: str = "unknown"):
        self.robot_type = robot_type
        self.episodes: list[Episode] = []

    def collect(self, source: Union[str, Path]) -> list[Episode]:
        """Collect episodes from a data source. Override in subclass."""
        raise NotImplementedError

    def save(self, output_dir: Union[str, Path], format: str = "json"):
        """Save collected episodes to disk.

        Args:
            output_dir: Directory to save episodes.
            format: "json" for individual files, "rlds" for RLDS bundle.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "json":
            for ep in self.episodes:
                path = output_dir / f"{ep.episode_id}.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(ep.to_dict(), f, ensure_ascii=False, indent=2)
        elif format == "rlds":
            # Save as single RLDS bundle
            bundle = {
                "metadata": {
                    "num_episodes": len(self.episodes),
                    "robot_type": self.robot_type,
                    "created_at": datetime.now().isoformat(),
                },
                "episodes": [ep.to_rlds() for ep in self.episodes],
            }
            path = output_dir / "rlds_bundle.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        return len(self.episodes)


class SyntheticCollector(BaseCollector):
    """Generate synthetic VLA episodes for testing and prototyping.

    Produces realistic-looking data with configurable dimensions,
    suitable for unit tests and integration tests.
    """

    def __init__(
        self,
        robot_type: str = "franka_panda",
        state_dim: int = 7,
        action_dim: int = 8,  # 7 joints + 1 gripper
        image_size: tuple[int, int] = (480, 640),
        seed: int = 42,
    ):
        super().__init__(robot_type)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_size = image_size
        self.seed = seed

    def collect(
        self,
        num_episodes: int = 10,
        steps_per_episode: int = 50,
        task_type: str = "grasping",
        instruction: str = "pick up the object",
    ) -> list[Episode]:
        """Generate synthetic episodes.

        Args:
            num_episodes: Number of episodes to generate.
            steps_per_episode: Steps per episode.
            task_type: Task type label.
            instruction: Language instruction for all episodes.

        Returns:
            List of Episode objects.
        """
        import numpy as np
        rng = np.random.RandomState(self.seed)

        self.episodes = []
        for ep_idx in range(num_episodes):
            steps = []
            # Simulate a trajectory: start random, converge to target
            start_state = rng.randn(self.state_dim).astype(np.float32) * 0.5
            target_state = rng.randn(self.state_dim).astype(np.float32) * 0.3

            for step_idx in range(steps_per_episode):
                t = step_idx / max(steps_per_episode - 1, 1)
                # Linear interpolation with noise
                state = (1 - t) * start_state + t * target_state + rng.randn(self.state_dim) * 0.02 * (1 - t)
                state = state.astype(np.float32)

                # Action: move toward target (state_dim dims) + gripper (1 dim)
                action_delta = (target_state - state) * 0.3
                noise = rng.randn(self.action_dim) * 0.01
                action_vec = np.zeros(self.action_dim, dtype=np.float32)
                action_vec[:self.state_dim] = action_delta + noise[:self.state_dim]
                if self.action_dim > self.state_dim:
                    action_vec[self.state_dim] = noise[self.state_dim]
                action_vec = np.clip(action_vec, -1.0, 1.0).astype(np.float32)

                gripper = 1.0 if t < 0.4 else (0.0 if t > 0.6 else 1.0 - (t - 0.4) / 0.2)

                obs = Observation(
                    robot_state=state.tolist(),
                    gripper=float(gripper),
                    # No real image in synthetic mode — use placeholder
                    image=None,
                )

                act = Action(
                    target_joint_pos=action_vec[:self.state_dim].tolist(),
                    gripper_command=float(action_vec[self.state_dim]) if self.action_dim > self.state_dim else gripper,
                )

                reward = float(-np.linalg.norm(state - target_state))

                step = Step(
                    observation=obs,
                    instruction=instruction,
                    action=act,
                    reward=reward,
                    is_terminal=(step_idx == steps_per_episode - 1),
                    is_success=(step_idx == steps_per_episode - 1 and reward > -0.1),
                    timestamp=time.time(),
                )
                steps.append(step)

            ep = Episode(
                episode_id=f"synth_ep_{ep_idx:04d}",
                steps=steps,
                metadata={
                    "robot_type": self.robot_type,
                    "task_type": task_type,
                    "language": "en",
                    "duration_sec": steps_per_episode * 0.1,
                    "num_steps": steps_per_episode,
                    "source": "synthetic",
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                },
            )
            self.episodes.append(ep)

        return self.episodes


class ROS2BagCollector(BaseCollector):
    """Collect VLA episodes from ROS2 bag files.

    Expected topics:
      /camera/color/image_raw  — RGB image (sensor_msgs/Image)
      /joint_states            — Joint angles (sensor_msgs/JointState)
      /gripper/status          — Gripper state (std_msgs/Float64)
      /action/command          — Action commands
      /instruction             — Language instruction (std_msgs/String)
    """

    def __init__(self, robot_type: str = "franka_panda"):
        super().__init__(robot_type)

    def collect(self, source: Union[str, Path]) -> list[Episode]:
        """Parse a ROS2 bag file directory.

        Args:
            source: Path to rosbag2 directory (contains metadata.yaml).

        Returns:
            List of Episode objects parsed from the bag.
        """
        bag_path = Path(source)
        if not bag_path.exists():
            raise FileNotFoundError(f"ROS2 bag not found: {bag_path}")

        # Try to import rosbag2_py
        try:
            from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        except ImportError:
            # Fallback: parse raw SQLite database directly
            return self._parse_sqlite_fallback(bag_path)

        return self._parse_with_rosbag2(bag_path)

    def _parse_with_rosbag2(self, bag_path: Path) -> list[Episode]:
        """Parse using rosbag2_py API."""
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

        storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
        converter_options = ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        # Group messages by topic
        topic_types = {}
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            topic_types[topic] = type(data).__name__

        # For now, return empty — full implementation requires
        # deserializing CDR-encoded ROS messages per topic type.
        # This is a placeholder for production integration.
        return []

    def _parse_sqlite_fallback(self, bag_path: Path) -> list[Episode]:
        """Fallback: directly read rosbag2 SQLite database.

        Rosbag2 stores data in metadata.yaml + a SQLite database.
        This method reads the DB without requiring rosbag2_py.
        """
        import sqlite3

        db_path = bag_path / "rosbag2.db"
        if not db_path.exists():
            # Try alternative names
            for name in ["rosbag2.db", "db3.sqlite"]:
                candidate = bag_path / name
                if candidate.exists():
                    db_path = candidate
                    break
            else:
                raise FileNotFoundError(f"No SQLite DB found in {bag_path}")

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get topics
        cursor.execute("SELECT id, name, type FROM topics")
        topics = {row[0]: {"name": row[1], "type": row[2]} for row in cursor.fetchall()}

        # Get messages
        cursor.execute("SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp")
        messages = cursor.fetchall()
        conn.close()

        # Group by topic
        topic_messages: dict[int, list] = {}
        for topic_id, timestamp, data in messages:
            if topic_id not in topic_messages:
                topic_messages[topic_id] = []
            topic_messages[topic_id].append({"timestamp": timestamp, "data": data})

        # Build episodes from message streams
        # This is a simplified parser — production version needs
        # CDR deserialization per message type.
        episode = Episode(
            episode_id=f"ros2_{bag_path.stem}",
            metadata={
                "robot_type": self.robot_type,
                "task_type": "unknown",
                "source": "ros2_bag",
                "bag_path": str(bag_path),
                "topics": list(topics.values()),
            },
        )
        self.episodes = [episode]
        return self.episodes


class IsaacSimCollector(BaseCollector):
    """Collect VLA episodes from Isaac Sim HDF5 recordings.

    Expected HDF5 structure:
      /episode_0/
        /obs/
          /rgb_cam/          (N, H, W, 3) uint8
          /depth_cam/        (N, H, W)    float32
          /joint_pos/        (N, D)        float32
          /gripper_qpos/     (N, 1)        float32
        /actions/
          /target_qpos/      (N, D)        float32
          /gripper_command/  (N, 1)        float32
        /instruction         str
        /success             bool
    """

    def __init__(self, robot_type: str = "franka_panda"):
        super().__init__(robot_type)

    def collect(self, source: Union[str, Path]) -> list[Episode]:
        """Parse Isaac Sim HDF5 recording.

        Args:
            source: Path to HDF5 file.

        Returns:
            List of Episode objects.
        """
        import h5py
        import numpy as np

        h5_path = Path(source)
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        self.episodes = []
        with h5py.File(h5_path, "r") as f:
            episode_keys = sorted(
                [k for k in f.keys() if k.startswith("episode_")],
                key=lambda x: int(x.split("_")[1]),
            )

            for ep_key in episode_keys:
                ep_group = f[ep_key]
                obs_group = ep_group["obs"]
                act_group = ep_group["actions"]

                # Read instruction
                instruction = ""
                if "instruction" in ep_group.attrs:
                    instruction = ep_group.attrs["instruction"]
                elif "instruction" in ep_group:
                    instruction = str(ep_group["instruction"][()])

                # Read success
                is_success = False
                if "success" in ep_group.attrs:
                    is_success = bool(ep_group.attrs["success"])
                elif "success" in ep_group:
                    is_success = bool(ep_group["success"][()])

                # Determine number of steps
                joint_pos = obs_group["joint_pos"][:]
                num_steps = len(joint_pos)

                steps = []
                for i in range(num_steps):
                    # Image (optional)
                    image_b64 = None
                    if "rgb_cam" in obs_group:
                        img = obs_group["rgb_cam"][i]
                        if img.ndim == 3 and img.shape[2] == 3:
                            _, buffer = cv2.imencode(".png", img)
                            image_b64 = base64.b64encode(buffer).decode("utf-8")

                    # Depth (optional)
                    depth_b64 = None
                    if "depth_cam" in obs_group:
                        depth = obs_group["depth_cam"][i]
                        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
                        _, buffer = cv2.imencode(".png", depth_norm)
                        depth_b64 = base64.b64encode(buffer).decode("utf-8")

                    # Robot state
                    state = obs_group["joint_pos"][i].astype(np.float32).tolist()

                    # Gripper
                    gripper = float(obs_group["gripper_qpos"][i]) if "gripper_qpos" in obs_group else 0.5

                    obs = Observation(
                        image=image_b64,
                        depth=depth_b64,
                        robot_state=state,
                        gripper=gripper,
                    )

                    # Action
                    target_pos = act_group["target_qpos"][i].astype(np.float32).tolist()
                    grip_cmd = float(act_group["gripper_command"][i]) if "gripper_command" in act_group else gripper

                    act = Action(
                        target_joint_pos=target_pos,
                        gripper_command=grip_cmd,
                    )

                    step = Step(
                        observation=obs,
                        instruction=instruction,
                        action=act,
                        reward=0.0,  # Isaac Sim may not provide per-step rewards
                        is_terminal=(i == num_steps - 1),
                        is_success=(i == num_steps - 1 and is_success),
                    )
                    steps.append(step)

                ep = Episode(
                    episode_id=ep_key,
                    steps=steps,
                    metadata={
                        "robot_type": self.robot_type,
                        "task_type": "unknown",
                        "source": "isaac_sim",
                        "hdf5_path": str(h5_path),
                        "num_steps": num_steps,
                    },
                )
                self.episodes.append(ep)

        return self.episodes


class JSONLogCollector(BaseCollector):
    """Collect VLA episodes from JSON/CSV teleoperation logs.

    Expected JSON format (per file = per episode):
    {
      "instruction": "pick up the red cup",
      "robot_type": "franka_panda",
      "steps": [
        {
          "image_path": "frame_0001.png",
          "joint_pos": [q1, ..., q7],
          "gripper": 0.8,
          "action": [a1, ..., a8],
          "reward": 0.0
        },
        ...
      ]
    }
    """

    def __init__(self, robot_type: str = "unknown"):
        super().__init__(robot_type)

    def collect(self, source: Union[str, Path]) -> list[Episode]:
        """Parse JSON log files.

        Args:
            source: Path to a single JSON file or directory of JSON files.

        Returns:
            List of Episode objects.
        """
        source_path = Path(source)
        files = []
        if source_path.is_file():
            files = [source_path]
        elif source_path.is_dir():
            files = sorted(source_path.glob("*.json"))
        else:
            raise FileNotFoundError(f"Source not found: {source_path}")

        self.episodes = []
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            instruction = data.get("instruction", "")
            robot_type = data.get("robot_type", self.robot_type)
            raw_steps = data.get("steps", [])

            steps = []
            for i, raw_step in enumerate(raw_steps):
                obs = Observation(
                    image=raw_step.get("image_path"),
                    robot_state=raw_step.get("joint_pos", []),
                    gripper=raw_step.get("gripper", 0.5),
                )

                raw_action = raw_step.get("action", [])
                if isinstance(raw_action, list) and len(raw_action) > 0:
                    act = Action(
                        target_joint_pos=raw_action[:-1] if len(raw_action) > 1 else [],
                        gripper_command=raw_action[-1] if len(raw_action) > 0 else 0.5,
                    )
                else:
                    act = None

                step = Step(
                    observation=obs,
                    instruction=instruction,
                    action=act,
                    reward=raw_step.get("reward", 0.0),
                    is_terminal=(i == len(raw_steps) - 1),
                    is_success=raw_step.get("success", False),
                )
                steps.append(step)

            ep = Episode(
                episode_id=fpath.stem,
                steps=steps,
                metadata={
                    "robot_type": robot_type,
                    "task_type": data.get("task_type", "unknown"),
                    "source": "json_log",
                    "file_path": str(fpath),
                },
            )
            self.episodes.append(ep)

        return self.episodes


# ── Utility Functions ──

def collect_from_source(
    source: Union[str, Path],
    source_type: str = "auto",
    robot_type: str = "franka_panda",
) -> list[Episode]:
    """Auto-detect source type and collect episodes.

    Args:
        source: Path to data source (file or directory).
        source_type: "auto", "ros2", "isaac_sim", "json", or "synthetic".
        robot_type: Robot type identifier.

    Returns:
        List of Episode objects.
    """
    source_path = Path(source)

    if source_type == "auto":
        if source_path.is_dir() and (source_path / "metadata.yaml").exists():
            source_type = "ros2"
        elif source_path.suffix == ".hdf5" or source_path.suffix == ".h5":
            source_type = "isaac_sim"
        elif source_path.is_file() and source_path.suffix == ".json":
            source_type = "json"
        elif source_path.is_dir():
            # Check for JSON files inside
            if list(source_path.glob("*.json")):
                source_type = "json"
            else:
                source_type = "synthetic"
        else:
            source_type = "synthetic"

    collectors = {
        "ros2": lambda: ROS2BagCollector(robot_type),
        "isaac_sim": lambda: IsaacSimCollector(robot_type),
        "json": lambda: JSONLogCollector(robot_type),
        "synthetic": lambda: SyntheticCollector(robot_type),
    }

    if source_type not in collectors:
        raise ValueError(f"Unknown source type: {source_type}. Choose from {list(collectors.keys())}")

    collector = collectors[source_type]()
    return collector.collect(source)


def merge_episodes(episodes_list: list[list[Episode]]) -> list[Episode]:
    """Merge multiple episode lists, deduplicating by episode_id."""
    seen = set()
    merged = []
    for episodes in episodes_list:
        for ep in episodes:
            if ep.episode_id not in seen:
                seen.add(ep.episode_id)
                merged.append(ep)
    return merged


def compute_episode_statistics(episodes: list[Episode]) -> dict:
    """Compute aggregate statistics over a list of episodes."""
    if not episodes:
        return {"num_episodes": 0}

    total_steps = sum(ep.num_steps for ep in episodes)
    successes = sum(1 for ep in episodes if ep.get_success_rate() > 0.5)
    avg_reward = np.mean([ep.get_avg_reward() for ep in episodes])
    action_dims = set(ep.action_dim for ep in episodes if ep.action_dim > 0)
    state_dims = set(ep.state_dim for ep in episodes if ep.state_dim > 0)

    return {
        "num_episodes": len(episodes),
        "total_steps": total_steps,
        "avg_steps_per_episode": total_steps / len(episodes),
        "success_rate": successes / len(episodes),
        "avg_reward": float(avg_reward),
        "action_dims": list(action_dims),
        "state_dims": list(state_dims),
    }
