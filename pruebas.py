from pyboy import PyBoy
from pyboy.plugins.game_wrapper_tetris import GameWrapperTetris
from gimnasio import TetrisEnv
import random
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np

# Configuración inicial para matplotlib
#plt.ion()  # Activa el modo interactivo
#fig, ax = plt.subplots()  # Crea una figura y un eje para el gráfico

# #Empezamos cargando el juego
file_path = "Tetris_3.gb"
tetris_env = TetrisEnv(game_file_path=file_path,vel=1)
ticks= 0

while True:  # Ejecutar 100 pasos, por ejemplo.
    action = random.randint(0, 4)  # Acción aleatoria.
    observation, reward, done, truncated  ,info = tetris_env.step(action)
    ticks += 1
    #if ticks % 15 == 0:
    print(reward,done,truncated)
    print(tetris_env.game_wrapper.game_over)
        
    #game_area=tetris_env.game_wrapper.game_area()
    # print(np.array(game_area))
    # print(np.array(len(game_area)))
    
    # print("DONEEEEEEEEEEEEEEEEEEE: ", done)
    # print("TRUNCATEEEEEEEEEE: ", truncated)
    #tetris_env.gameoverArea()

    # if tetris_env.gameoverArea():
    #     tetris_env.gameoverArea()
    #     tetris_env.close()
    #     break


    # # Obtener la imagen del render
    # rendered_image = tetris_env.render()

    # # Visualizar la imagen
    # ax.imshow(rendered_image)
    # ax.axis('off')  # Desactivar los ejes
    # plt.draw()  # Dibujar la imagen
    # plt.pause(0.001)  # Pequeña pausa para actualizar la imagen

    # if done:
    #     tetris_env.close()
    #     break


# check_env = check_env(tetris_env, warn=True, skip_render_check=True)


# # Desactiva el modo interactivo al finalizar
# plt.ioff()
# plt.show()