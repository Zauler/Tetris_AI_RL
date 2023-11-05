import gymnasium
import numpy as np
from skimage.transform import resize
from pyboy import PyBoy, WindowEvent
from einops import rearrange  # Asegúrate de tener instalada la librería einops



class TetrisEnv(gymnasium.Env):
        
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 15
    }
        
    def __init__(self, game_file_path,vel=1):
        super().__init__()
        self.pyboy = PyBoy(game_file_path,window_type="SDL2", window_scale=3, debug=False, game_wrapper=True)

        self.pyboy.set_emulation_speed(vel)
        self.game_wrapper = self.pyboy.game_wrapper()
        self.game_wrapper.start_game(timer_div=0x00) #inciamos el juego.
        self.action_space = gymnasium.spaces.Discrete(4)  # Suponiendo 4 acciones: mover izquierda, mover derecha, rotar, bajar

         # Define el output_shape, asumiendo que deseas una resolución baja como 84x84 y color RGB.
        self.output_shape = (84, 84, 3)  # Ajusta según sea necesario
        
        # Actualizar el observation_space para que coincida con output_shape
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)

        #Si no aumenta el juego en x frames empezamos ciclo.
        self.prev_score = 0  # Puntuación en el último tick
        self.ticks_since_last_score_increase = 0  # Ticks desde la última vez que la puntuación aumentó



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


        #Comprueba si la puntuación ha aumentado
        if reward > self.prev_score:
         self.ticks_since_last_score_increase = 0  # Reinicia el contador de ticks
         self.prev_score = reward  # Actualiza la puntuación anterior
        else:
         self.ticks_since_last_score_increase += 1  # Incrementa el contador de ticks

        # Comprueba la condición de truncamiento
        truncated = self.ticks_since_last_score_increase >= 1800 #Elegimos 1800 x 15 fps x 120 segundos
        truncated= False

        return observation, reward, done, truncated ,info 
    
    def _get_observation(self):
        # Obtener la matriz de píxeles RGB del juego
        game_pixels = self.pyboy.botsupport_manager().screen().screen_ndarray()
        
        # Reducir la resolución de la imagen, si es necesario
        reduced_res_pixels = (255 * resize(game_pixels, self.output_shape)).astype(np.uint8)
        
        return reduced_res_pixels
    
    def _get_reward(self):
        # Implementar la lógica para calcular la recompensa
        score = self.game_wrapper.score * 5
        lines = self.game_wrapper.lines * 500
        penalization = self.ticks_since_last_score_increase*0.01

        return score + lines - penalization

    def _is_done(self):
        # Implementar la lógica para ver si el episodio ha terminado
        # done = bool(self.game_wrapper.game_over())
        done = False
        return done
    


    def reset(self,seed=None):
        if seed is None:
            seed = 0x00
        self.game_wrapper.reset_game(timer_div=seed)
        observation = self._get_observation()
        info = {} # Información adicional (en este caso, un diccionario vacío)
        return observation, info
    

    def render(self, reduce_res=True, add_memory=True, update_mem=False):

        game_pixels_render = self.pyboy.botsupport_manager().screen().screen_ndarray()  # (144, 160, 3)
        if reduce_res:
            # Reducir la resolución de la imagen del juego
            game_pixels_render = (255 * resize(game_pixels_render, self.output_shape)).astype(np.uint8)
            if update_mem:
                # Actualizar el array de fotogramas recientes
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                # Crear una imagen compuesta con memoria de exploración y memoria reciente
                pad = np.zeros(shape=(self.mem_padding, self.output_shape[1], 3), dtype=np.uint8)
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(),
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                    ),
                    axis=0
                )
        return game_pixels_render

    def close(self):
        self.pyboy.stop()
