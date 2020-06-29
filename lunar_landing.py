# Created by khanhdh on 6/9/20
import logging

import gym
import numpy
from utils import plotLearning

from deep_q_learning import Agent

logger = logging.getLogger('main')


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 1000
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
    best = -1000

    i = 0
    while True:
        i += 1
        done = False
        avg_score = numpy.mean(scores[max(0, i - 10):(i + 1)])
        if i % 10 == 0 and i > 0:
            print(f'episode: {i}\tscore: {score}\taverage score: {"%.3f"%avg_score}\tepsilon: {"%.3f"%agent.epsilon}')
        else:
            print(f'episode: {i}\tscore: {score}')
        if best < avg_score:
            best = avg_score
            print('best average score:', avg_score)
            agent.save_models()

        state = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(state, action, reward, new_state, int(done))
            state = new_state
            agent.learn()

        scores.append(score)
        eps_history.append(agent.epsilon)

    x = list(range(1, n_games+1))
    plotLearning(x, scores, eps_history, file_name)
