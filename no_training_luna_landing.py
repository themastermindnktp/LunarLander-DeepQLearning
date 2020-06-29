# Created by khanhdh on 6/9/20
import logging

import gym
import numpy
from gym import wrappers

from utils import plotLearning

from deep_q_learning import Agent

logger = logging.getLogger('main')


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 500
    agent = Agent(
        name='1',
        n_games=n_games,
        alpha=5e-4,
        gamma=0.99,
        epsilon=1.0,
        input_dims=[8],
        n_actions=4,
        mem_size=int(1e5),
        batch_size=64,
    )

    file_name = 'lunar_lander.png'
    scores = []
    eps_history = []

    score = 0
    best = 0

    env = wrappers.Monitor(env, "tmp/lunar-lander-4",
                            video_callable=lambda episode_id: True, force=True)

    for i in range(n_games):
        done = False
        state = env.reset()
        agent.load_models()
        score = 0
        time = 0
        randomize = False
        while not done:
            time += 1
            if time > 30:
                randomize = False
            action = agent.choose_action(state, randomize=randomize)
            new_state, reward, done, info = env.step(action)
            score += reward
            state = new_state

        scores.append(score)
        avg_score = numpy.mean(scores[max(0, i - 10):(i + 1)])
        print(f'episode: {i}\tscore: {score}\taverage score: {avg_score}')
        eps_history.append(agent.epsilon)

    x = list(range(1, n_games+1))
    plotLearning(x, scores, eps_history, file_name)
