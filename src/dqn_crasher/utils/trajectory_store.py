import json
import os

from dqn_crasher.buffers.experience_replay import Transition


class TrajectoryStore(object):
    def __init__(self, file_path: str = "trajectories/all_episodes.jsonl"):
        # Holds completed episodes: { episode_idx: [ {state,action,next_state,reward}, … ] }
        self.episodes = {}
        self.current_episode = None

        self.dirpath = os.path.dirname(file_path)
        if self.dirpath:
            os.makedirs(self.dirpath, exist_ok=True)

        self.original_file_path = file_path
        self.file_path = file_path
        self.metadata = None

    def reset_filepath(self, step):
        split = self.original_file_path.split("/")
        split[-2] += f"/{step}"
        file_path = os.path.join(*split)
        file_path = os.path.join(*split)
        if self.original_file_path.startswith("/"):
            file_path = "/" + file_path

        self.dirpath = os.path.dirname(file_path)
        if self.dirpath:
            os.makedirs(self.dirpath, exist_ok=True)

        self.save_metadata(self.metadata)

        self.file_path = file_path

    def start_episode(self, episode_idx: int):
        """Begin collecting a fresh list of transitions."""
        self.current_episode = episode_idx
        self.episodes[episode_idx] = []

    def add(self, transition: Transition, info):
        """
        Append a single Transition(state, action, next_state, reward) to the current episode.
        """
        if self.current_episode is None:
            raise RuntimeError("call start_episode() before add()")

        # Convert numpy arrays to lists, ints/floats to native types
        entry = {
            "state": transition.state.tolist(),
            "action": transition.action.tolist(),
            "next_state": None
            if transition.next_state is None
            else transition.next_state.tolist(),
            "reward": float(transition.reward),
            "ttc_x": info["ttc_x"],
            "ttc_y": info["ttc_y"],
            "collision_details": info.get("collision_details", None),
        }
        self.episodes[self.current_episode].append(entry)

    def end_episode(self):
        """
        Write the just‑finished episode as one JSON line,
        then clear it from memory.
        """
        ep = self.current_episode
        if ep is None:
            return

        record = {"episode": ep, "transitions": self.episodes.pop(ep)}

        # Append one JSON object per line
        with open(self.file_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

        self.current_episode = None

    def save_metadata(self, config: dict):
        if self.metadata is None:
            self.metadata = config
        metadata_file = os.path.join(self.dirpath, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(config, f)

    def load(self, file_path: str = None):
        """
        Load all episodes from a JSONL file into self.episodes.
        Any existing episodes in memory will be replaced.
        """
        path = file_path or self.file_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: {path}")

        loaded = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ep_idx = record.get("episode")
                transitions = record.get("transitions", [])
                loaded[ep_idx] = transitions

        self.episodes = loaded
        print(f"Loaded {len(loaded)} episodes from {path}")
        return self.episodes
