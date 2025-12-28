import contextlib
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch

from dqn_crasher.buffers.experience_replay import Transition
from dqn_crasher.utils.trajectory_store import TrajectoryStore
from dqn_crasher.agents.dqn_agent import DQN_Agent
from dqn_crasher.utils.model_pool import ModelPool, Sampling
from dqn_crasher.utils.utils import DeviceHelper
from dqn_crasher.utils.config import load_pkg_yaml

from dqn_crasher.policies import BasePolicy

from amago.agent import Agent
from amago.nets.tstep_encoders import FFTstepEncoder
from amago.nets.traj_encoders import TformerTrajEncoder
from amago.cli_utils import switch_traj_encoder
from amago.utils import retry_load_checkpoint, get_constant_schedule_with_warmup, stack_list_array_dicts
from amago.hindsight import Timestep, Trajectory

from accelerate import Accelerator, DistributedDataParallelKwargs

from gymnasium.spaces import Dict, Box, Discrete
import gymnasium as gym
import highway_env




class AMAGOPolicy(BasePolicy):
    def __init__(self, trajectory_store_dir, *args, **kwargs):

        # TODO: Parametrize device selection
        dev_config = { "device": 'cuda' }
        device = DeviceHelper.get(dev_config)
        print(f"Device: {device}")
        config = {}

        # TODO: Parametrize trajectory encoder
        traj_encoder_type = switch_traj_encoder(
            config, arch="transformer", memory_size=256, layers=3
        )

        # TODO: Derive rl2_space from environment and take argument to override
        rl2_space = Dict(
            {
                "obs": Dict({ "observation": Box(
                    shape=(5,5),
                    dtype=np.float32,
                    low=float("-inf"),
                    high=float("inf"),
                )}),
                "rl2": Box(
                    shape=(6,),
                    dtype=np.float32,
                    low=float("-inf"),
                    high=float("inf"),
                ),
            }
        )

        # TODO: Parametrize tstep_encoder_type and derive action space from environment
        policy_kwargs = {
            "tstep_encoder_type": FFTstepEncoder,
            "traj_encoder_type": traj_encoder_type,
            "obs_space": rl2_space["obs"],
            "rl2_space": rl2_space["rl2"],
            "action_space": Discrete(5),
            "max_seq_len": 400,
        }

        self.parallel_actors = 1
        self.sample_actions = False
        self.hidden_state = None

        print(f"Initializing policy with args: {policy_kwargs}")
        policy = Agent(**policy_kwargs)

        adamw_kwargs = dict(lr=1e-4, weight_decay=1e-3)
        optimizer = torch.optim.AdamW(policy.trainable_params, **adamw_kwargs)

        lr_schedule = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=500
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=1,
            device_placement=True,
            log_with="wandb",
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ],
            mixed_precision='no',
        )

        self.policy_aclr, self.optimizer, self.lr_schedule = self.accelerator.prepare(
            policy, optimizer, lr_schedule
        )

        # Not training but need to match state of checkpoint
        self.accelerator.register_for_checkpointing(self.lr_schedule) # Not doing this will result in an error "No Distributed Objects"
        # TODO: Parametrize ckpt_path
        ckpt_path = '/p/crash/amago-multi-car-lane-400/multi-car-lane-400-2000_crash-v0_trial_0/ckpts/training_states/multi-car-lane-400-2000_crash-v0_trial_0_epoch_1950'
        self.accelerator.load_state(ckpt_path)

        self.prev_action = None

        class_name = type(self).__module__ + "." + type(self).__name__
        train_file_path = os.path.join(
            trajectory_store_dir, "train", f"{class_name}.jsonl"
        )
        test_file_path = os.path.join(
            trajectory_store_dir, "test", f"{class_name}.jsonl"
        )

        self.store: TrajectoryStore = TrajectoryStore(file_path=train_file_path)
        self.test_store: TrajectoryStore = TrajectoryStore(file_path=test_file_path)

    @property
    def policy(self) -> Agent:
        """Returns the current Agent policy free from the accelerator wrapper."""
        return self.accelerator.unwrap_model(self.policy_aclr)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def car_id(self):
        return self._car_id

    def caster(self):
        """Get the context manager for mixed precision training."""
        return contextlib.suppress()

    def reset(self, obs = None, info = None, batched_envs: int = 1, test : bool = True):
        self.step_count = np.zeros((batched_envs,), dtype=np.int64)
        self.prev_action = np.zeros((batched_envs, 5), dtype=np.uint8)
        self.hidden_state = None
        if obs is None:
            return None, None
        timestep = self.make_timestep(obs, self.prev_action, None, None)
        return timestep, info
        

    def restore_state(self, state):
        pass

    def save_state(self):
        pass

    def set_state(self, ego_state, npc_state):
        pass

    def set_config(self, config: dict):
        pass

    def set_car_id(self, car_id: int):
        self._car_id = car_id

    def select_action(self, obs, rewards, terminateds, truncateds, infos, batched_envs: int = 1, car_id: int = 0):
        if self.car_id is None:
            self._car_id = car_id

        if self.hidden_state is None:
            # init new hidden state
            self.hidden_state = self.policy.traj_encoder.init_hidden_state(
                self.parallel_actors, self.device
            )

        self.step_count += 1
        done = np.logical_or(terminateds, truncateds)
        timestep = self.make_timestep(obs, self.prev_action, rewards, done)
        obs, rl2s, time_idxs = self.get_t([timestep.as_input()])

        with torch.no_grad():
            with self.caster():
                actions, self.hidden_state = self.policy.get_actions(
                    obs=obs,
                    rl2s=rl2s,
                    time_idxs=time_idxs,
                    sample=self.sample_actions,
                    hidden_state=self.hidden_state,
                )

        
        self.prev_action =self.make_action_rep(actions)

        return actions, self.prev_action.copy()

    def make_action_rep(self, action, batched_envs : int = 1) -> np.ndarray:
        # action as one-hot
        action_rep = np.zeros((batched_envs, 5), dtype=np.uint8)
        action_rep[np.arange(batched_envs), action[..., 0]] = 1

        return action_rep.astype(np.uint8)

    def update(self, transition: Transition, done, train):
        return transition.next_state

    def make_timestep(self, obs, prev_action, reward, terminal, batched_envs : int = 1):


        if isinstance(obs, tuple):
            obs = obs[self.car_id]
            obs = obs[:, -5:]

        if not isinstance(obs, dict):
            obs = {"observation": obs}

        if batched_envs == 1:
            # force batch dim
            obs = {k: v[np.newaxis, ...] for k, v in obs.items()}
            # reward = np.array([reward], dtype=np.float32)
            # terminated = np.array([terminated], dtype=bool)
            # truncated = np.array([truncated], dtype=bool)

            if reward is None:
                reward = np.zeros((batched_envs,), dtype=np.float32)
            else:
                reward = np.array([reward], dtype=np.float32)
            if terminal is None:
                terminal = np.zeros((batched_envs,), dtype=bool)
            else:
                terminal = np.array([terminal], dtype=bool)

        timestep = Timestep(
            obs = obs,
            prev_action = prev_action,
            reward = reward,
            terminal=terminal,
            time_idx=self.step_count.copy(),
            batched_envs = batched_envs
        )
        return timestep

    def get_t(self, timestep):
        # fetch `Timestep.make_sequence` from all envs
        _obs, _rl2s, _time_idxs = [], [], []
        for _o, _r, _t in timestep:
            _obs.append(_o)
            _rl2s.append(_r)
            _time_idxs.append(_t)
        # stack all the results
        _obs = stack_list_array_dicts(_obs, axis=0, cat=True)
        _rl2s = np.concatenate(_rl2s, axis=0)
        _time_idxs = np.concatenate(_time_idxs, axis=0)
        # ---> torch --> GPU --> dummy length dim
        _obs = {
            k: torch.from_numpy(v).to(self.device).unsqueeze(1)
            for k, v in _obs.items()
        }
        _rl2s = torch.from_numpy(_rl2s).to(self.device).unsqueeze(1)
        _time_idxs = torch.from_numpy(_time_idxs).to(self.device).unsqueeze(1)
        return _obs, _rl2s, _time_idxs


if __name__ == "__main__":
    amago_policy = AMAGOPolicy(trajectory_store_dir='.')
    amago_policy.set_car_id(0)
    amago_policy.policy.eval()

    gym_config = load_pkg_yaml("configs/env/multi_agent.yaml")
    env = gym.make('crash-v0', render_mode = 'rgb_array',  config=gym_config)
    env = gym.wrappers.RecordVideo(env, video_folder="./",
              episode_trigger=lambda e: True)
    obs, info = env.reset()
    timestep, info = amago_policy.reset(obs, info)
    # obs, rl2s, time_idxs = amago_policy.get_t([timestep.as_input()])

    done = False

    episodes = 0
    while not done and episodes < 10:
        actions = amago_policy.select_action(obs, None, None, None, None)
        action = actions.squeeze().cpu().numpy()
        obs, reward, term, trunc, info = env.step((action, 0))
        done = np.logical_or(term, trunc)
        env.render()

        if done:
            obs, info = env.reset()
            episodes += 1
            done = False


