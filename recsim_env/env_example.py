
from recsim.simulator import environment
from recsim.simulator import recsim_gym

from recsim_env.recsim_models import *


def create_env():
    slate_size = 3
    num_candidates = 10
    ltsenv = environment.Environment(
        LTSUserModel(slate_size),
        LTSDocumentSampler(),
        num_candidates,
        slate_size,
        resample_documents=True)

    lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)
