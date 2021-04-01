
from recsim_env.recsim_gym import RecSimGymEnv
from recsim_env import environment
from recsim_env.recsim_models import *


def create_env():
    slate_size = 3
    num_candidates = 10
    ltsenv = environment.Environment(
        UserModel(slate_size),
        DocumentSampler(),
        num_candidates,
        slate_size,
        resample_documents=True)

    gym_env = RecSimGymEnv(ltsenv, clicked_engagement_reward)

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
