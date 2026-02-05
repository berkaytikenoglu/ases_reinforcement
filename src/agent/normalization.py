
import numpy as np

class RunningMeanStd:
    """Tracks running mean and standard deviation of data stream."""
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 0 else 1
        
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class RewardScaler:
    """
    Stabilizes reward signals using compression and normalization.
    Pipeline:
    Raw -> Normalize (Running Mean/Std) -> Buffer
    Log Scaling is DISABLED by default because it destroys penalty ratios.
    """
    def __init__(self, use_log=False, gamma=0.99):
        self.rms = RunningMeanStd(shape=())
        self.gamma = gamma
        self.ret = 0  # Discounted return
        self.use_log = use_log
        self.epsilon = 1e-8
        
    def __call__(self, reward, reset=False):
        """
        Processes a raw reward.
        Args:
            reward (float): Raw reward from environment.
            reset (bool): If True, resets the discounted return (e.g. at episode end).
        Returns:
            float: Normalized reward ready for PPO buffer.
        """
        # 1. Compress (Log Scaling)
        # Handle large values like +100k or -50k
        # Sgn(r) * Log(1 + |r|) shrinks magnitude while preserving sign and order
        if self.use_log:
            compressed_reward = np.sign(reward) * np.log(1 + np.abs(reward))
        else:
            compressed_reward = reward # Or Tanh

        # 2. Update Running Mean/Std (using discounted return approximation)
        # PPO usually normalizes Advantages, but normalizing rewards helps Value function convergence
        # Here we just track statistics of the reward stream itself or returns?
        # Standard implementation (like stable-baselines3 VecNormalize) normalizes RETURNS, not just rewards.
        # But for simplicity in this loop, we can normalize the compressed reward itself to be roughly unit variance.
        
        if reset:
            self.ret = 0
            
        self.ret = self.ret * self.gamma + compressed_reward
        self.rms.update(np.array([self.ret]))
        
        # 3. Normalize
        # (r - mean) / std ? No, usually we just scale by std dev of returns
        # reward / sqrt(var + epsilon)
        
        normalized_reward = compressed_reward / np.sqrt(self.rms.var + self.epsilon)
        
        # Clip to prevent extreme outliers even after normalization (optional)
        normalized_reward = np.clip(normalized_reward, -10.0, 10.0)
        
        return normalized_reward

    def reset(self):
        self.ret = 0
