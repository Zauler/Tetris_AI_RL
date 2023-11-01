from pyboy import PyBoy
from pyboy.plugins.game_wrapper_tetris import GameWrapperTetris
from gimnasio import TetrisEnv
import random

#Empezamos cargando el juego
file_path = "Tetris_3.gb"
tetris_env = TetrisEnv(game_file_path=file_path)

for _ in range(1000):  # Ejecutar 100 pasos, por ejemplo.
    action = random.randint(0, 3)  # Acción aleatoria.
    observation, reward, done, info = tetris_env.step(action)
    tetris_env.render()
    if _ == 300:
        tetris_env.reset()

