import gymnasium as gym
import numpy as np

class RetryOnCrashWrapper(gym.Wrapper):
    def __init__(self, env, max_retries=None):
        """
        A wrapper that repeats the same scenario seed if the agent crashes.
        
        :param env: The environment to wrap
        :param max_retries: Optional integer. If the agent fails this many times 
                            on the same seed, force a new seed anyway to prevent infinite loops.
        """
        super().__init__(env)
        self.current_seed = np.random.randint(0, 2**32 - 1)
        self.last_episode_crashed = False
        self.retry_count = 0
        self.max_retries = max_retries

    def reset(self, *, seed=None, options=None):
        # Logic: 
        # 1. If we crashed last time, we want to RETRY (keep current_seed).
        # 2. If we succeeded (didn't crash), we want NEW SCENARIO (new current_seed).
        # 3. If we hit max_retries, we force NEW SCENARIO.
        
        should_retry = self.last_episode_crashed
        
        if self.max_retries is not None and self.retry_count >= self.max_retries:
            should_retry = False

        if should_retry:
            self.retry_count += 1
            # print(f"Crash detected! Retrying same seed: {self.current_seed} (Attempt {self.retry_count})")
        else:
            # Generate a fresh seed for a new scenario
            self.current_seed = np.random.randint(0, 2**32 - 1)
            self.retry_count = 0
            # print(f"New Scenario. Seed: {self.current_seed}")

        # IMPORTANT: We override the seed passed to the internal env
        return self.env.reset(seed=self.current_seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Monitor the info dict for the crash flag
        # Highway-env standard is info['crashed']
        if "crashed" in info:
            self.last_episode_crashed = info["crashed"]
        
        # Add seed info to the output for tracking/logging
        info["scenario_seed"] = self.current_seed
        info["num_retries"] = self.retry_count
        
        return obs, reward, terminated, truncated, info