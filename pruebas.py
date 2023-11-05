from pyboy import PyBoy
from pyboy.plugins.game_wrapper_tetris import GameWrapperTetris
from gimnasio import TetrisEnv
import random
from stable_baselines3.common.env_checker import check_env

# #Empezamos cargando el juego
file_path = "Tetris_3.gb"
tetris_env = TetrisEnv(game_file_path=file_path)

for _ in range(100000):  # Ejecutar 100 pasos, por ejemplo.
    action = random.randint(0, 3)  # Acción aleatoria.
    observation, reward, done, truncated  ,info = tetris_env.step(action)
    print(tetris_env.game_wrapper.level)
    
    # print("DONEEEEEEEEEEEEEEEEEEE: ", done)
    # print("TRUNCATEEEEEEEEEE: ", truncated)
    #tetris_env.gameoverArea()

    # if tetris_env.gameoverArea():
    #     tetris_env.gameoverArea()
    #     tetris_env.close()
    #     break

    if done:
        tetris_env.close()


# check_env = check_env(tetris_env, warn=True, skip_render_check=True)
