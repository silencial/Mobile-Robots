import numpy as np
from halton import halton


class DubinsSampler:
    def __init__(self, env):
        self.env = env
        self.xlimit = self.env.xlimit
        self.ylimit = self.env.ylimit

    def sample(self, num_samples):
        """
        Samples configurations.
        Each configuration is (x, y, angle).

        @param num_samples: Number of sample configurations to return
        @return 2D numpy array of size [num_samples x 3]
        """

        # TODO
        xy_samples = halton(2, num_samples)
        start = self.env.limit.T[0]
        end = self.env.limit.T[1]
        xy_samples = start + (end-start-1) * xy_samples

        angle_step = 4
        angles = np.linspace(0, 2 * np.pi, angle_step)
        samples = np.empty((4 * num_samples, 3))
        for i in range(angle_step):
            samples[i * num_samples:(i+1) * num_samples, 0:2] = xy_samples
            samples[i * num_samples:(i+1) * num_samples, 2] = angles[i]
        return samples
