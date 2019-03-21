import gym
import random
random.seed(0)

numepisodes = 10


def main():
    env = gym.make('MsPacman-v0')  # Crea el juego
    env.seed(0)  # Hace los resultados reproducibles
    rewards = []

    for  in range(num_episodes):
        env.reset()
        episode_reward = 0
        while True:
            action = env.actionspace.sample()
            , reward, done, _ = env.step(action)  # Solo hace acciones random.
            episode_reward += reward
            if done:
                env.render()
                print('Reward: %d' % episode_reward)
                rewards.append(episode_reward)
                break
    print('Average reward: %.2f' % (sum(rewards) / len(rewards)))


main()