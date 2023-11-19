from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from gimnasio import TetrisEnv
import torch

#CUDA
# PyTorch esté utilizando CUDA si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available)

# Define el modelo
model = Sequential()
model.add(Flatten(input_shape=(160, 144, 3)))  # Ajusta según sea necesario
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='linear'))  # Suponiendo 4 acciones

# Compila el modelo
model.compile(optimizer=Adam(), loss='mse')

# Entrena el modelo
env = TetrisEnv('Tetris.gb')
num_episodes = 1000

for episode in range(num_episodes):
    observation = env.reset()
    done = False

    while not done:
        action = model.predict(observation)
        observation, reward, done, info = env.step(action)

        # Aquí necesitarías implementar tu lógica de actualización del modelo