import gym
import numpy as np


def sample_from_simplex(dim):
    samples = np.array([0.0] * (dim + 1))
    samples[-1] = 1.0
    samples[1:-1] = np.random.random(dim - 1)
    samples.sort()
    simplex_samples = samples[1:] - samples[:-1]
    return simplex_samples


class FakeRecSim(gym.Env):
    def __init__(self, num_candidates=10, slate_size=3, num_user_type=2, doc_dim=3):
        super(FakeRecSim, self).__init__()

        self.docs = []
        self.num_docs = num_candidates
        self.slate_size = slate_size
        self.num_utypes = num_user_type
        self.doc_dim = doc_dim

        ## Transition model parameters
        ##############################
        self.transition_threshold = 0.5
        self.transition_coeff = 0.9
        self.utype = None

        ## Engagement parameters
        ##############################
        self.min_eng = 1.0
        self.max_eng = 2.0
        self.eng_scale = 0.1

        ## State variables
        ##############################
        self.user_prefs = None
        self.time_budget = 60

        self.reset_task()
        self.reset()

    def generate_engagement(self, score):
        # TODO: fix this
        engagement_loc = (score * self.max_eng
                          + (1 - score) * self.min_eng)
        engagement_scale = self.eng_scale
        log_engagement = np.random.normal(loc=engagement_loc, scale=engagement_scale)
        # return np.exp(log_engagement) # TODO: why exp???
        return np.exp(log_engagement)

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        slate = self.docs[action]
        scores = np.array([(doc * self.user_prefs).sum() for doc in slate])
        item_idx_chosen = np.random.choice(slate.shape[0], (scores / scores.sum()))

        # Response
        engagement = self.generate_engagement(scores[item_idx_chosen])

        # Transition
        if self.utype == 1:
            if scores[item_idx_chosen] < self.transition_threshold:
                self.user_prefs = self.transition_coeff * self.user_prefs + (
                        1 - self.transition_coeff) * slate[item_idx_chosen]

        else:
            if scores[item_idx_chosen] > self.transition_threshold:
                self.user_prefs = self.transition_coeff * self.user_prefs + (
                        1 - self.transition_coeff) * slate[item_idx_chosen]

        self.time_budget -= 1

        done = True if self.time_budget <= 0 else False
        obs = self.docs  # TODO: is this correct?
        reward = engagement
        info = {'task': self.utype}

        return obs, reward, done, info

    @property
    def observation_space(self):
        return gym.spaces.Box(shape=(self.num_docs, self.doc_dim), dtype=np.float32, low=0.0, high=1.0)

    @property
    def action_space(self):
        return gym.spaces.MultiDiscrete(self.num_docs * np.ones((self.slate_size,)))

    def reset(self):
        """
        Reset the environment. This should *NOT* automatically reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """
        self.user_prefs = sample_from_simplex(self.doc_dim)
        self.docs = np.stack([sample_from_simplex(self.doc_dim) for _ in range(self.num_docs)], axis=0)
        return self.docs

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return self.utype

    def reset_task(self, task=None):
        """
        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment.
        """
        self.utype = task if task is not None else np.random.randint(1, 3)
