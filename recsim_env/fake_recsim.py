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
        self.transition_coeff = 0.95
        self.utype = None

        ## Engagement parameters
        ##############################
        self.min_eng = 0.0
        self.max_eng = 1.0
        self.eng_scale = 0.1

        ## State variables
        ##############################
        self.user_prefs = None
        self.pref_collection = None

        self._max_episode_steps = 100
        self.task_dim = 1

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
            raise NotImplementedError
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

        # Transition 1
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

        # Transition 2
        # Condition for transition modification for both users
        if user_idx_choice != np.argmax(scores):        # user did NOT choose the best doc

            if self.utype == 1:     # user that likes to vary - goes towards the chosen doc
                influencing_doc = slate[user_idx_choice]
            else:                   # user that does not vary - goes towards the argmax doc
                influencing_doc = slate[np.argmax(scores)]

            # Users change their transition in the same way (but to different directions)
            self.user_prefs = self.transition_coeff * self.user_prefs + (1 - self.transition_coeff) * influencing_doc
            self.user_prefs /= self.user_prefs.sum()

        self.pref_collection[self._max_episode_steps - self.time_budget] = self.user_prefs
        self.time_budget -= 1

        # Make observation
        slate_idx_one_hot = np.zeros(self.docs.shape[0])
        slate_idx_one_hot[slate_idx] = 1.0

        done = self.time_budget <= 0
        user_choice_one_hot = np.zeros(self.docs.shape[0])
        user_choice_one_hot[user_idx_choice] = 1.0

        obs = np.concatenate((self.docs.reshape(-1), slate_idx_one_hot, user_choice_one_hot))
        reward = engagement
        info = {'task': self.utype, 'prefs': self.pref_collection}

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

        # USER PREFS INIT
        # self.user_prefs = sample_from_simplex(self.doc_dim)

        if self.utype == 1:
            self.user_prefs = np.array([0.15, 0.15, 0.7])
        else:
            self.user_prefs = np.array([0.7, 0.15, 0.15])

        self.pref_collection = np.zeros((self._max_episode_steps, 3))
        self.pref_collection[0] = self.user_prefs
        # self.user_prefs = np.array([0.15, 0.15, 0.7])

        # DOCS INIT
        # self.docs = np.stack([sample_from_simplex(self.doc_dim) for _ in range(self.num_docs)], axis=0)

        # self.docs = np.array(
        #     [[0., 0., 1.],           #0
        #      [0., 1., 0.],           #1
        #      [1., 0., 0.],           #2
        #      [0.333, 0.333, 0.333],         #3
        #      [0.333, 0.333, 0.333],         #4
        #      [0.333, 0.333, 0.333],        #5
        #      [0.333, 0.333, 0.333],  #6
        #      [0.333, 0.333, 0.333],        #7
        #      [0.333, 0.333, 0.333],        #8
        #      [0.333, 0.333, 0.333]])       #9
        self.docs = np.array(
            [[0., 0., 1.],           #0
             [0., 1., 0.],           #1
             [1., 0., 0.],           #2
             [0., 0.5, 0.5],         #3
             [0.5, 0.5, 0.],         #4
             [0.5, 0.0, 0.5],        #5
             [0.333, 0.333, 0.333],  #6
             [0.2, 0.4, 0.4],        #7
             [0.4, 0.2, 0.4],        #8
             [0.4, 0.4, 0.2]])       #9

        return np.concatenate((self.docs.reshape(-1), np.zeros(self.docs.shape[0]), np.zeros(self.docs.shape[0])))

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return np.array(self.utype)

    def reset_task(self, task=None):
        """
        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment.
        """
        self.utype = task if task is not None else np.random.randint(1, 3)
        # self.utype = 1
        # self.utype = 2
