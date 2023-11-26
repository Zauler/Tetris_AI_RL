import gymnasium
import numpy as np
import random
from skimage.transform import resize
from pyboy import PyBoy, WindowEvent
from einops import rearrange  # Asegúrate de tener instalada la librería einops



class TetrisEnv(gymnasium.Env):
        
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 15
    }
        
    def __init__(self, game_file_path,vel=1, memory_frames = False, memory_in_seconds = 5 ):
        super().__init__()
        self.pyboy = PyBoy(game_file_path,window_type="SDL2", window_scale=3, debug=False, game_wrapper=True)

        self.pyboy.set_emulation_speed(vel)
        self.game_wrapper = self.pyboy.game_wrapper()
        self.game_wrapper.start_game(timer_div=0x00) #inciamos el juego.
        self.action_space = gymnasium.spaces.Discrete(5)  # Suponiendo 4 acciones: mover izquierda, mover derecha, rotar, bajar

         # Define el output_shape, asumiendo que deseas una resolución baja como 84x84 y color RGB.
        self.output_shape = (160, 144, 3)  # Ajusta según sea necesario (baja es 84x84,3)
        
        # Actualizar el observation_space para que coincida con output_shape
        self.observation_space = gymnasium.spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)

        #sistema de recompensas
        self.game_over_color = 135 #Cuando se acaba el juego todo el area está en este color
        self.gameover_ticks = 0 #Contador de ticks desde que se acaba el juego.
        self.game_over_zone = False # TRUE si pierdes cuando la ficha llega al final.
        self.game_overs_count = 0 #Contador de veces que game_over_zone se activa.
        self.frame_count = 0  # Añadir un contador para los frames para la memoria visual
        self.memory_frames = memory_frames #Para ver la memoria de frames pasados, si FALSE no hay memoria
        self.memory_in_seconds = memory_in_seconds # Cuantos "segundos" almacenamos de memoria, es decir cada 15 frames almacenamos uno durante X periodos.


        #Para ver las siguientes fichas:
        # Table for translating game-representation of Tetromino types (8-bit int) to string
        self.tetromino_table = {
            "L": 0,
            "J": 4,
            "I": 8,
            "O": 12,
            "Z": 16,
            "S": 20,
            "T": 24,
        }
        self.inverse_tetromino_table = {v: k for k, v in self.tetromino_table.items()}
        self.NEXT_TETROMINO_ADDR = 0xC213 #Lugar de la memoria donde está la siguiente ficha a aparecer.



    def step(self, action):
        # Mapeo de acciones:
        # 0 -> mover izquierda
        # 1 -> mover derecha
        # 2 -> rotar
        # 3 -> bajar
        # 4 -> No hacer nada
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
        elif action ==4:
            self.pyboy.tick()
        
        observation = self._get_observation()  # Crear un método con la observación
        
        
        last_game_over_count = self.game_overs_count #Capturamos el anterior contador de gameovers
        reward = self._get_reward()  # miramos la recomenpensa, donde tambíen se mira los gameovers
        
        if last_game_over_count < self.game_overs_count: #si es mayor es que ha perdido
            done = True
        else:
            done = False
            
        info = {}  # Información adicional

        # Comprueba la condición de truncamiento
        truncated = False

        return observation, reward, done, truncated ,info 
    
    def _get_observation(self):
        # Obtener la matriz de píxeles RGB del juego
        game_pixels = self.pyboy.botsupport_manager().screen().screen_ndarray()
        
        # Reducir la resolución de la imagen, si es necesario
        reduced_res_pixels = (255 * resize(game_pixels, self.output_shape)).astype(np.uint8)
        
        return reduced_res_pixels


    def _get_reward(self):
        # Implementar la lógica para calcular la recompensa
        score = self.game_wrapper.score * (self.game_wrapper.level+1)
        lines = self.game_wrapper.lines * 500 * (self.game_wrapper.level+1)
        penalization = 0
        self._gameoverArea()

        if self.game_over_zone: #Si pierde la partida, se añade penalización
            penalization += 10000
            self.game_over_zone = False
            self.game_overs_count += 1 #Sumamos una vez que ha perdido
            print(self.game_overs_count)

        penalization += self._calculate_penalty()
        
        if self.gameover_ticks > 0:
             score = 0  # No dar recompensa mientras se está esperando
             lines = 0 # No dar recompensa mientras se está esperando
             penalization += 10000
        return score + lines - penalization


    def _calculate_penalty(self):
        game_area=self.game_wrapper.game_area()
        penalty= 0 
        acumulative = 0
        max_penalty_per_row = 15 #
        rows,cols = game_area.shape
        for row in range(rows):
            if any(val != 47 for val in game_area[row, :]):
                acumulative += 1
            else :
                acumulative = 0
            
            if acumulative > 10:
                    penalty += (rows-row)*max_penalty_per_row

        return penalty


    def _gameoverArea(self):
            # Si gameover_ticks es mayor que 0, decrementarlo y retornar
        if self.gameover_ticks > 0:
            self.gameover_ticks -= 1
            return
        
        # Función para comprobar si hay gameover, todo mismo color en la zona de juego.
        game_area=self.game_wrapper.game_area()
        unique_values = np.unique(game_area)
        all_same_value = len(unique_values) == 1 and unique_values[0] == self.game_over_color
        if all_same_value:
            self.game_over_zone = True
            self._reset_memory()  # Resetear la memoria de frames
            self.gameover_ticks = 90  # Esperar 90 ticks antes de contar otro gameover
            print("GAME OVER!")
            
        
        #print(np.array2string(game_area, separator=', ')) #Ver la matriz de bloques
        return self.game_over_zone
    

    def next_tetromino(self):
        """
        Returns the next Tetromino to drop.

        __NOTE:__ Don't use this function together with
        `pyboy.plugins.game_wrapper_tetris.GameWrapperTetris.set_tetromino`.

        Returns
        -------
        shape:
            `str` of which Tetromino will drop:
            * `"L"`: L-shape
            * `"J"`: reverse L-shape
            * `"I"`: I-shape
            * `"O"`: square-shape
            * `"Z"`: zig-zag left to right
            * `"S"`: zig-zag right to left
            * `"T"`: T-shape
        """
        # Bitmask, as the last two bits determine the direction
        return self.inverse_tetromino_table[self.pyboy.get_memory_value(self.NEXT_TETROMINO_ADDR) & 0b11111100]

    def get_next_tetromino_image(self):
        # Obtener el tipo de la próxima ficha
        next_tetromino_type = self.next_tetromino()

        # Colores para representar cada tipo de ficha
        tetromino_colors = {
            "L": (255/255, 0/255, 0/255),  # Rojo
            "J": (0/255, 255/255, 0/255),  # Verde
            "I": (0/255, 0/255, 255/255),  # Azul
            "O": (255/255, 255/255, 0/255), # Amarillo
            "Z": (255/255, 0/255, 255/255), # Magenta
            "S": (0/255, 255/255, 255/255), # Cian
            "T": (128/255, 0/255, 128/255)  # Púrpura
            }

        # Crear una imagen simple para la próxima ficha
        next_tetromino_image = np.full((20, 20, 3), tetromino_colors[next_tetromino_type], dtype=np.uint8)
        return next_tetromino_image
    
    def _reset_memory(self):
        if self.memory_frames:
            # Reiniciar la memoria de frames a frames vacíos
            self.past_frames = [np.zeros(self.output_shape, dtype=np.uint8) for _ in range(15)]
            self.frame_count = 0  # También reinicia el contador de frames



    def reset(self,seed=None):
        if seed is None:
            #seed =  0x00
            seed = random.randint(0, 0xffffffff)  # Genera una semilla aleatoria
        self.game_wrapper.reset_game(timer_div=seed)
        observation = self._get_observation()
        info = {} # Información adicional (en este caso, un diccionario vacío)
        self._reset_memory()
        return observation, info
    

    def render(self, show_next_tetromino=True):
        # Obtener la imagen actual del juego a la resolución reseada
        game_pixels_render = self._get_observation()

        # Añadir la visualización de la próxima ficha
        if show_next_tetromino:
                next_tetromino_image = self.get_next_tetromino_image()
                # Redimensionar next_tetromino_image para que coincida con la anchura de game_pixels_render
                next_tetromino_image = resize(next_tetromino_image, (next_tetromino_image.shape[0], game_pixels_render.shape[1], 3))

                padding_height = 10  # Altura del padding que separa las imágenes
                padding = np.zeros((padding_height, game_pixels_render.shape[1], 3), dtype=np.uint8)  # Separador vertical

                game_pixels_render = np.concatenate([game_pixels_render, padding, next_tetromino_image], axis=0)

        # Añadir memoria de frames pasados
        if self.memory_frames:
            if not hasattr(self, 'past_frames'):
                self.past_frames = [np.zeros(self.output_shape, dtype=np.uint8) for _ in range(self.memory_in_seconds)]  # Inicializar si no existe

            self.frame_count += 1
            if self.frame_count % 15 == 0:
                self.past_frames.pop(0)  # Eliminar el frame más antiguo
                self.past_frames.append(np.copy(game_pixels_render))  # Añadir el frame actual

            # Crear una imagen compuesta con los últimos 15 frames
            memory_image = np.concatenate(self.past_frames[::-1], axis=0)  # Invertir la lista para que el más reciente esté arriba
            game_pixels_render = np.concatenate([game_pixels_render, memory_image], axis=0)
            
        return game_pixels_render
    

        
    def close(self):
        self.pyboy.stop()
