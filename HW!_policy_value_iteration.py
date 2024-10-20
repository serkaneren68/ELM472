import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time

# Ortamı başlat
env = gym.make('CliffWalking-v0')

# Parametreler
n_states = env.observation_space.n  # Durum sayısı
n_actions = env.action_space.n      # Aksiyon sayısı
gamma = 0.99  # İndirim faktörü
theta = 1e-6  # Yakınsama eşiği
max_steps = 1000  # Maksimum adım sayısı (sonsuz döngüyü önlemek için)

def run_episode(env, policy):
    """Verilen politika ile bir episode çalıştır ve toplam ödülü hesapla."""
    state = env.reset()[0]
    total_reward = 0
    done = False
    step_count = 0  # Adım sayacını başlat

    while not done and step_count < max_steps:
        action = np.argmax(policy[state])
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1

    if step_count >= max_steps:
        print("Maksimum adım sayısına ulaşıldı, döngü sonlandırıldı.")

    return total_reward

def policy_evaluation(policy, env, gamma, theta):
    """Politika için değer fonksiyonunu hesaplar."""
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    if done:
                        v += action_prob * prob * reward
                    else:
                        v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_improvement(env, V, gamma):
    """Değer fonksiyonuna göre politikayı iyileştirir."""
    policy = np.zeros([n_states, n_actions])
    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                if done:
                    q_values[a] += prob * reward
                else:
                    q_values[a] += prob * (reward + gamma * V[next_state])
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0
    return policy

def policy_iteration(env, gamma, theta):
    """Policy iteration algoritmasını uygular ve toplam ödülleri takip eder."""
    policy = np.ones([n_states, n_actions]) / n_actions  # Rastgele politika
    rewards_per_iteration = []  # Her iterasyondaki toplam ödülleri sakla
    iteration = 0
    total_evaluations = 0  # Policy evaluation sayaç

    while True:
        iteration += 1
        print(f"Policy Iteration - Iteration {iteration}")

        # Policy evaluation ve evaluation sayacını artır
        V = policy_evaluation(policy, env, gamma, theta)
        total_evaluations += 1

        # Policy improvement
        new_policy = policy_improvement(env, V, gamma)

        total_reward = run_episode(env, new_policy)
        rewards_per_iteration.append(total_reward)

        print(f"Total Reward in Iteration {iteration}: {total_reward}")

        if np.all(policy == new_policy):
            print("Politika kararlı hale geldi.")
            break

        policy = new_policy

    return policy, V, rewards_per_iteration, total_evaluations


def value_iteration(env, gamma, theta):
    """Value iteration algoritmasını uygular ve optimal politikayı çıkarır."""
    V = np.zeros(n_states)
    rewards_per_iteration = []
    iteration = 0

    while True:
        delta = 0
        iteration += 1
        print(f"Value Iteration - Iteration {iteration}")

        for s in range(n_states):
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    if done:
                        q_values[a] += prob * reward
                    else:
                        q_values[a] += prob * (reward + gamma * V[next_state])

            max_q = np.max(q_values)
            delta = max(delta, abs(max_q - V[s]))
            V[s] = max_q

        total_reward = run_episode(env, derive_policy(env, V))
        rewards_per_iteration.append(total_reward)

        print(f"Total Reward in Iteration {iteration}: {total_reward}")

        if delta < theta:
            print("Value iteration tamamlandı.")
            break

    return derive_policy(env, V), V, rewards_per_iteration

def derive_policy(env, V):
    """Verilen değer fonksiyonuna göre optimal politika çıkarır."""
    policy = np.zeros([n_states, n_actions])
    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                if done:
                    q_values[a] += prob * reward
                else:
                    q_values[a] += prob * (reward + gamma * V[next_state])
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0
    return policy

def plot_rewards(rewards):
    """Her iterasyondaki toplam ödülleri görselleştirir."""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker='o')
    plt.xlabel('İterasyon Sayısı')
    plt.ylabel('Toplam Ödül')
    plt.title('Toplam Ödülün İterasyona Göre Değişimi')
    plt.grid(True)
    plt.show()

def play_human_mode(env, policy):
    """Optimal politika ile ortamda oyunu oynatır."""
    state = env.reset()[0]
    done = False
    env.render()

    step_count = 0

    while not done and step_count < max_steps:
        action = np.argmax(policy[state])
        state, reward, done, _, _ = env.step(action)

        env.render()
        print(f"Durum: {state}, Ödül: {reward}")

        step_count += 1
        time.sleep(0.5)

        if done:
            print("Terminal durumuna ulaşıldı!")
            break

    if step_count >= max_steps:
        print("Maksimum adım sayısına ulaşıldı, oyun sonlandırıldı.")

    print("Oyun Bitti!")

def plot_policy(policy):
    """Politikayı oklarla görselleştirir."""
    action_symbols = ['↑', '→', '↓', '←']
    grid = np.array([action_symbols[np.argmax(policy[s])] for s in range(n_states)])
    grid = grid.reshape(4, 12)

    plt.figure(figsize=(10, 5))
    plt.table(cellText=grid, loc='center', cellLoc='center', fontsize=20)
    plt.axis('off')
    plt.title('Optimal Politika')
    plt.show()

def main():
    """Yöntemi seçerek algoritmayı çalıştırır."""
    method = input("Hangi yöntemi çalıştırmak istiyorsunuz? ('policy' veya 'value'): ").strip().lower()
    compare_algorithms(env, gamma, theta) 
    # if method == 'policy':
    #     policy, V, rewards = policy_iteration(env, gamma, theta)
    # elif method == 'value':
    #     policy, V, rewards = value_iteration(env, gamma, theta)
    # else:
    #     raise ValueError("Geçersiz yöntem! 'policy' veya 'value' kullanın.")

    print("Optimal Politika:")
    plot_policy(policy)

    plot_rewards(rewards)

    print("Oyun Başlıyor...")
    play_human_mode(env, policy)

import time

def compare_algorithms(env, gamma, theta):
    """Policy Iteration ve Value Iteration yöntemlerini karşılaştırır."""
    
    # Policy Iteration çalıştır ve süreyi ölç
    print("\n--- Policy Iteration Başlıyor ---")
    start_time = time.time()
    _, _, policy_rewards, policy_evaluations = policy_iteration(env, gamma, theta)
    policy_duration = time.time() - start_time

    # Value Iteration çalıştır ve süreyi ölç
    print("\n--- Value Iteration Başlıyor ---")
    start_time = time.time()
    _, _, value_rewards = value_iteration(env, gamma, theta)
    value_duration = time.time() - start_time

    # Grafikle karşılaştırma yap
    plt.figure(figsize=(12, 6))

    plt.plot(policy_rewards, marker='o', label='Policy Iteration Toplam Ödül')
    plt.plot(value_rewards, marker='x', label='Value Iteration Toplam Ödül')

    plt.xlabel('İterasyon Sayısı')
    plt.ylabel('Toplam Ödül')
    plt.title('Policy Iteration vs Value Iteration - Ödüllerin Karşılaştırması')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Süre ve evaluation sayısını ekrana bas
    print(f"Policy Iteration Süresi: {policy_duration:.2f} saniye")
    print(f"Policy Iteration Policy Evaluation Sayısı: {policy_evaluations}")
    print(f"Value Iteration Süresi: {value_duration:.2f} saniye")



main()
