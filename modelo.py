# Suponiendo que estás usando Stable Baselines
from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv

# Importar tu entorno
from gimnasio import TetrisEnv

# Crear el entorno
env = TetrisEnv('Tetris.gb')

# Vectorizar el entorno (requerido por Stable Baselines)
vec_env = DummyVecEnv([lambda: env])

# Crear el modelo
model = DQN("CnnPolicy", vec_env, verbose=1)

# Entrenar el modelo
model.learn(total_timesteps=100000)

# Guardar el modelo
model.save("dqn_tetris")

# Cargar el modelo
loaded_model = DQN.load("dqn_tetris")

# Evaluar el modelo
obs = env.reset()
done = False
while not done:
    action, _ = loaded_model.predict(obs)
    obs, reward, done, info = env.step(action)