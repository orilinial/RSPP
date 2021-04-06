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

        self.env_type = 'Box'       # 'Categorical' or 'Box'
        self.oracle = False

        self.num_docs = num_candidates
        self.slate_size = slate_size
        self.num_utypes = num_user_type
        self.doc_dim = doc_dim
        self.docs = None

        ## Transition model parameters
        ##############################
        self.transition_threshold = 0.5
        self.transition_coeff = 0.9
        self.utype = None

        ## Engagement parameters
        ##############################
        self.min_eng = 1.0
        self.max_eng = 5.0
        self.eng_scale = 0.1

        ## State variables
        ##############################
        self.user_prefs = None
        self.time_budget = 60

        self._max_episode_steps = 60

        self.reset_task()
        self.reset()

    def generate_engagement(self, score):
        engagement_loc = (score * self.max_eng
                          + (1 - score) * self.min_eng)
        # engagement_scale = self.eng_scale
        # log_engagement = np.random.normal(loc=engagement_loc, scale=engagement_scale)
        # return np.exp(log_engagement)
        return engagement_loc

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        if self.env_type == 'Categorical':
            slate = self.docs[action.nonzero()]
        elif self.env_type == 'Box':
            probs = np.exp(action) / np.exp(action).sum()
            slate_idx = np.random.choice(self.docs.shape[0], size=self.slate_size, replace=False,
                                         p=probs)
            slate = self.docs[slate_idx]

        scores = np.array([(doc * self.user_prefs).sum() for doc in slate])
        user_idx_choice = np.random.choice(slate.shape[0], p=(scores / scores.sum()))

        # Response
        engagement = self.generate_engagement(scores[user_idx_choice])

        # Transition
        # if self.utype == 1:
        #     if scores[item_idx_chosen] < self.transition_threshold:
        #         self.user_prefs = self.transition_coeff * self.user_prefs + (
        #                 1 - self.transition_coeff) * slate[item_idx_chosen]
        #         self.user_prefs /= self.user_prefs.sum()
        #
        # else:
        #     if scores[item_idx_chosen] > self.transition_threshold:
        #         self.user_prefs = self.transition_coeff * self.user_prefs + (
        #                 1 - self.transition_coeff) * slate[item_idx_chosen]
        #         self.user_prefs /= self.user_prefs.sum()

        self.time_budget -= 1

        done = self.time_budget <= 0

        # Make observation
        slate_idx_one_hot = np.zeros(self.docs.shape[0])
        slate_idx_one_hot[slate_idx] = 1.0

        user_choice_one_hot = np.zeros(self.docs.shape[0])
        user_choice_one_hot[user_idx_choice] = 1.0

        obs = np.concatenate((self.docs.reshape(-1), slate_idx_one_hot, user_choice_one_hot))
        reward = engagement
        info = {'task': self.utype}

        if self.oracle:
            scores = np.array([(doc * self.user_prefs).sum() for doc in self.docs])
            reward = self.generate_engagement(np.min(scores))

        return obs, reward, done, info

    @property
    def observation_space(self):
        return gym.spaces.Box(shape=(self.num_docs * (self.doc_dim + 2),), dtype=np.float32, low=0.0, high=1.0)

    @property
    def action_space(self):
        if self.env_type == 'Categorical':
            return gym.spaces.MultiDiscrete(self.num_docs * np.ones((self.slate_size,)))
        elif self.env_type == 'Box':
            return gym.spaces.Box(shape=(self.num_docs,), low=-1.0, high=1.0)

    def reset(self):
        """
        Reset the environment. This should *NOT* automatically reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """
        self.time_budget = self._max_episode_steps
        # self.user_prefs = sample_from_simplex(self.doc_dim)
        if self.utype == 1:
            self.user_prefs = np.array([0.15, 0.15, 0.7])
        elif self.utype:
            self.user_prefs = np.array([0.7, 0.15, 0.15])

        # self.docs = np.stack([sample_from_simplex(self.doc_dim) for _ in range(self.num_docs)], axis=0)
        self.single_docs = np.array(
            [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [0., 0.5, 0.5], [0.5, 0.5, 0.], [0.5, 0.0, 0.5],
             [0.333, 0.333, 0.333], [0.2, 0.4, 0.4], [0.4, 0.2, 0.4], [0.4, 0.4, 0.2]])
        self.docs = self.single_docs
        return np.concatenate((self.docs.reshape(-1), np.zeros(self.docs.shape[0]), np.zeros(self.docs.shape[0])))

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
