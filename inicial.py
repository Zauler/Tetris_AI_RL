import time
from pyboy import PyBoy, WindowEvent
from F_aux import *

#Empezamos cargando el juego
file_path = "Tetris.gb"
pyboy = PyBoy(file_path)

# Obtener el gestor de soporte para bots
bot_m = pyboy.botsupport_manager()

# Configurar la velocidad de emulación a la velocidad normal (60 FPS asumido)

velocidad_emulacion = 1# 60 FPS = 1
pyboy.set_emulation_speed(velocidad_emulacion)

# Variable de control para asegurar que ciertas acciones sólo se realicen una vez
acciones_realizadas = False





# Iniciar la emulación (esto abrirá una ventana con el juego)
while not pyboy.tick():

    if not acciones_realizadas:
        navegar_menu(velocidad_emulacion,pyboy)
        acciones_realizadas = True

    # Obtén una captura de pantalla del juego
    screen = bot_m.screen()
    screenshot = screen.screen_image()
    # Mostrar la captura de pantalla
    screenshot.show()


    pass



pyboy.stop()