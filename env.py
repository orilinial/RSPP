import numpy as np
from recsim import document
from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
# from recsim.simulator import environment
import environment
# from recsim.simulator import recsim_gym
import recsim_gym
from gym import spaces


def sample_from_simplex(dim):
    samples = np.array([0.0] * (dim + 1))
    samples[-1] = 1.0
    samples[1:-1] = np.random.random(dim - 1)
    samples.sort()
    simplex_samples = samples[1:] - samples[:-1]
    return simplex_samples


#  Document model
class Document(document.AbstractDocument):
    def __init__(self, doc_id, genre):
        self.genre = genre
        super(Document, self).__init__(doc_id)  # doc_id is an integer representing the unique ID of this document

    def create_observation(self):
        return self.genre

    @staticmethod
    def observation_space():
        return spaces.Box(shape=(3,), dtype=np.float32, low=0.0, high=1.0)

    def __str__(self):
        return "Document {} with genre {}.".format(self._doc_id, self.genre)


class DocumentSampler(document.AbstractDocumentSampler):
    def __init__(self, doc_ctor=Document, **kwargs):
        super(DocumentSampler, self).__init__(doc_ctor, **kwargs)
        self._doc_count = 0
        self.genre_dim = 3

    def sample_document(self):
        doc_features = {'doc_id': self._doc_count,
                        'genre': sample_from_simplex(self.genre_dim)}
        self._doc_count += 1
        return self._doc_ctor(**doc_features)


# User model
class UserState(user.AbstractUserState):
    def __init__(self, preferences, min_eng, max_eng, eng_scale, transition_threshold, transition_coeff, user_type,
                 time_budget, observation_noise_stddev=0.1):
        ## Transition model parameters
        ##############################
        self.transition_threshold = transition_threshold
        self.transition_coeff = transition_coeff
        self.user_type = user_type

        ## Engagement parameters
        ##############################
        self.min_eng = min_eng
        self.max_eng = max_eng
        self.eng_scale = eng_scale

        ## State variables
        ##############################
        self.preferences = preferences
        self.time_budget = time_budget

        # Noise
        self._observation_noise = observation_noise_stddev

    def create_observation(self):
        # Option 1: preferences are observable
        return self.preferences

        # Option 2: return only the most preferred genre
        # prefs = np.zeros(3)
        # prefs[np.argmax(self.preferences)] = 1.0
        # return prefs

        # Option 3: add some noise, this is not a simplex anymore
        # noise = np.random.random(self.preferences.shape[0]) * self._observation_noise
        # return self.preferences + noise

    @staticmethod
    def observation_space():
        return spaces.Box(shape=(3,), dtype=np.float32, low=-2.0, high=2.0)

    # scoring function for use in the choice model
    def score_document(self, doc_obs):
        return (doc_obs * self.preferences).sum()


class StaticUserSampler(user.AbstractUserSampler):
    _state_parameters = None

    def __init__(self,
                 user_ctor=UserState,
                 min_eng=1.0,
                 max_eng=2.0,
                 eng_scale=0.1,
                 transition_threshold=0.5,
                 transition_coeff=0.9,
                 time_budget=60,
                 **kwargs):
        self._state_parameters = {'time_budget': time_budget,
                                  'min_eng': min_eng,
                                  'max_eng': max_eng,
                                  'eng_scale': eng_scale,
                                  'transition_threshold': transition_threshold,
                                  'transition_coeff': transition_coeff}
        super(StaticUserSampler, self).__init__(user_ctor, **kwargs)

    def sample_user(self):
        starting_preferences = sample_from_simplex(3)
        self._state_parameters['preferences'] = starting_preferences
        self._state_parameters['user_type'] = np.random.randint(1, 3)
        return self._user_ctor(**self._state_parameters)


# Response model
class Response(user.AbstractResponse):
    # The maximum degree of engagement.
    MAX_ENGAGEMENT_MAGNITUDE = 100.0

    def __init__(self, clicked=False, engagement=0.0):
        self.clicked = clicked
        self.engagement = engagement

    def create_observation(self):
        return {'click': int(self.clicked), 'engagement': np.array(self.engagement)}

    @classmethod
    def response_space(cls):
        # `engagement` feature range is [0, MAX_ENGAGEMENT_MAGNITUDE]
        return spaces.Dict({
            'click':
                spaces.Discrete(2),
            'engagement':
                spaces.Box(
                    low=0.0,
                    high=cls.MAX_ENGAGEMENT_MAGNITUDE,
                    shape=tuple(),
                    dtype=np.float32)
        })


#####
# User model
class UserModel(user.AbstractUserModel):
    def __init__(self, slate_size, seed=0):
        super(UserModel, self).__init__(Response, StaticUserSampler(UserState, seed=seed), slate_size)
        self.choice_model = MultinomialLogitChoiceModel({})

    def simulate_response(self, slate_documents):
        # List of empty responses
        responses = [self._response_model_ctor() for _ in slate_documents]
        # Get click from of choice model.
        self.choice_model.score_documents(
            self._user_state, [doc.create_observation() for doc in slate_documents])
        scores = self.choice_model.scores
        selected_index = self.choice_model.choose_item()
        # Populate clicked item.
        self.generate_response(slate_documents[selected_index],
                               responses[selected_index],
                               scores[selected_index])
        return responses

    def generate_response(self, doc, response, score):
        response.clicked = True
        # linear interpolation between choc and kale.
        engagement_loc = (score * self._user_state.max_eng
                          + (1 - score) * self._user_state.min_eng)
        engagement_scale = self._user_state.eng_scale
        log_engagement = np.random.normal(loc=engagement_loc,
                                          scale=engagement_scale)
        response.engagement = np.exp(log_engagement)

    def update_state(self, slate_documents, responses):
        for doc, response in zip(slate_documents, responses):
            if response.clicked:
                score = (self._user_state.preferences * doc.genre).sum()

                alpha = self._user_state.transition_coeff

                if self._user_state.user_type == 1:
                    if score < self._user_state.transition_threshold:
                        self._user_state.preferences = alpha * self._user_state.preferences + (1 - alpha) * doc.genre

                else:
                    if score > self._user_state.transition_threshold:
                        self._user_state.preferences = alpha * self._user_state.preferences + (1 - alpha) * doc.genre

                self._user_state.time_budget -= 1
                return

    def is_terminal(self):
        """Returns a boolean indicating if the session is over."""
        return self._user_state.time_budget <= 0


# Define reward:
def clicked_engagement_reward(responses):
    reward = 0.0
    for response in responses:
        if response.clicked:
            reward += response.engagement
    return reward


def create_env():
    slate_size = 3
    num_candidates = 10
    ltsenv = environment.Environment(
        UserModel(slate_size),
        DocumentSampler(),
        num_candidates,
        slate_size,
        resample_documents=True)

    gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)

    observation_0 = gym_env.reset()
    print('Observation 0')
    print('Available documents')
    doc_strings = ['doc_id ' + key + " kaleness " + str(value) for key, value
                   in observation_0['doc'].items()]
    print('\n'.join(doc_strings))
    print('Noisy user state observation')
    print(observation_0['user'])
    # Agent recommends the first three documents.
    recommendation_slate_0 = [0, 1, 2]
    observation_1, reward, done, _ = gym_env.step(recommendation_slate_0)
    print('Observation 1')
    print('Available documents')
    doc_strings = ['doc_id ' + key + " kaleness " + str(value) for key, value
                   in observation_1['doc'].items()]
    print('\n'.join(doc_strings))
    rsp_strings = [str(response) for response in observation_1['response']]
    print('User responses to documents in the slate')
    print('\n'.join(rsp_strings))
    print('Noisy user state observation')
    print(observation_1['user'])


create_env()
