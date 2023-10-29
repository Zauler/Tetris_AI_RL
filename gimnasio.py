import gymnasium
import numpy as np  # Importar numpy para utilizar np.float32
import matplotlib.pyplot as plt
from pyboy import PyBoy, WindowEvent
from pyboy.plugins.game_wrapper_tetris import GameWrapperTetris
from skimage import transform, color


class TetrisEnv(gymnasium.Env):
    def __init__(self, game_file_path):
        super().__init__()
        self.pyboy = PyBoy(game_file_path)
        title = self.pyboy.cartridge_title()
        self.game_wrapper = GameWrapperTetris(self.pyboy,title)
        self.game_wrapper.start_game() #inciamos el juego.
        self.action_space = gymnasium.spaces.Discrete(4)  # Suponiendo 4 acciones: mover izquierda, mover derecha, rotar, bajar
        self.observation_space = gymnasium.spaces.Box(low=0, high=1.0, shape=(84, 84), dtype=np.float32)

    def step(self, action):
        # Mapeo de acciones:
        # 0 -> mover izquierda
        # 1 -> mover derecha
        # 2 -> rotar
        # 3 -> bajar
        if action == 0:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
        elif action == 1:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
        elif action == 2:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        elif action == 3:
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
        
        observation = self._get_observation()  # Crear un método con la observación
        reward = self._get_reward()  # Crear un método con la recompensa
        done = self._is_done()  # Hace falta crear un método para ver si el episodio ha terminado
        info = {}  # Información adicional
        return observation, reward, done, info
    
    def _get_observation(self):
        screen = self.pyboy.botsupport_manager().screen()
        screenshot = screen.screen_image()
        observaction = np.array(screenshot)
        # Convierte la imagen a escala de grises
        resized_image = transform.resize(observaction, (84, 84), anti_aliasing=True) #Devuelve imagenes de 0 a 1.
        gray_image = color.rgb2gray(resized_image)
        return gray_image

    def _get_reward(self):
        # Implementar la lógica para calcular la recompensa
        score = self.game_wrapper.score() * 5
        level = self.game_wrapper.level() * 100
        lines = self.game_wrapper.lines() * 5

        return score + level + lines

    def _is_done(self):
        # Implementar la lógica para ver si el episodio ha terminado
        return self.game_wrapper.game_has_started() and self.game_wrapper.game_over()



    def reset(self):
        self.game_wrapper.reset_game()
        return self._get_observation()
    

    def render(self):
        screen = self.pyboy.botsupport_manager().screen()
        screenshot = screen.screen_image()
        observation = np.array(screenshot)

        if hasattr(self, 'rendering_window') and self.rendering_window is not None:
            # Si ya hay una ventana de renderización, solo actualiza la imagen
            self.rendering_window.set_data(observation)
            plt.draw()
            plt.pause(0.001)  # pequeña pausa para actualizar la ventana
        else:
            # Si no hay una ventana de renderización, crea una nueva
            plt.ion()  # modo interactivo ON para que no bloquee la ejecución
            self.rendering_window = plt.imshow(observation)
            plt.show()

    def close(self):
        self.pyboy.stop()
