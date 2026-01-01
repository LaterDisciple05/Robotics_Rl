import gymnasium as gym
import numpy as np
import imageio

NUMBER_OF_EPISODES = 300
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.97
EPSILON = 0.2


def initialize_environment():
    env = gym.make('CliffWalking-v1')
    state_size = env.observation_space.n
    action_size = env.action_space.n
    print(f"State size: {state_size}, Action size: {action_size}")
    return env, state_size, action_size


def initialize_q_table(state_size, action_size):
    return np.zeros((state_size, action_size))


def epsilon_greedy_action_selection(state, qtable, env, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(qtable[state, :])


def update_q_value(current_state, action, reward, next_state, qtable, learning_rate, discount_factor):
    future_q_value = np.max(qtable[next_state, :])
    current_q_value = qtable[current_state, action]
    new_q_value = current_q_value + learning_rate * (reward + discount_factor * future_q_value - current_q_value)
    qtable[current_state, action] = new_q_value


def train_agent(env, qtable, num_episodes, learning_rate, discount_factor, epsilon):
    for episode_nr in range(num_episodes):
        current_state, _ = env.reset()
        done = False

        while not done:
            action = epsilon_greedy_action_selection(current_state, qtable, env, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            update_q_value(current_state, action, reward, next_state, qtable, learning_rate, discount_factor)
            current_state = next_state

        if episode_nr % 10000 == 0:
            print(f"\nQ-table after episode {episode_nr + 1}:")
            np.set_printoptions(precision=2, suppress=True)
            print(qtable)

    return qtable


def save_qtable(filename, qtable):
    np.save(filename, qtable)
    print(f"Q-table saved as {filename}")


def create_replay_video(env, qtable, filename="replay.mp4"):
    frames = []
    current_state, _ = env.reset()
    done = False

    while not done:
        frames.append(env.render())
        action = np.argmax(qtable[current_state, :])
        next_state, _, done, _, _ = env.step(action)
        current_state = next_state

    env.close()

    with imageio.get_writer(filename, fps=10) as video:
        for frame in frames:
            video.append_data(frame)

    print(f"Video saved as {filename}")


def main():
    env, state_size, action_size = initialize_environment()
    qtable = initialize_q_table(state_size, action_size)

    qtable = train_agent(env, qtable, NUMBER_OF_EPISODES, LEARNING_RATE, DISCOUNT_FACTOR, EPSILON)
    save_qtable("cliffWalking_qtable.npy", qtable)

    env = gym.make('CliffWalking-v1', render_mode="rgb_array")
    create_replay_video(env, qtable)


if __name__ == "__main__":
    main()
