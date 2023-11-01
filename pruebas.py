from pyboy import PyBoy
from pyboy.plugins.game_wrapper_tetris import GameWrapperTetris
from gimnasio import TetrisEnv
import random
from stable_baselines3.common.env_checker import check_env

# #Empezamos cargando el juego
file_path = "Tetris_3.gb"
tetris_env = TetrisEnv(game_file_path=file_path)

# for _ in range(100):  # Ejecutar 100 pasos, por ejemplo.
#     action = random.randint(0, 3)  # Acción aleatoria.
#     observation, reward, done, info = tetris_env.step(action)

#     if _ == 300:
#         tetris_env.reset()


check_env = check_env(tetris_env, warn=True, skip_render_check=True)
