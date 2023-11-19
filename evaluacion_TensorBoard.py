from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class TensorboardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str):
        super(TensorboardCallback, self).__init__()
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            try:
                # Obtener las instancias de game_wrapper de todos los entornos
                game_wrappers = self.training_env.get_attr('game_wrapper')

                # Asegúrate de que game_wrappers no está vacío y tiene los atributos necesarios
                if game_wrappers and all(hasattr(wrapper, 'score') and hasattr(wrapper, 'lines') for wrapper in game_wrappers):
                    # Calcular el promedio de score y lines de todos los entornos
                    avg_reward = np.mean([wrapper.score for wrapper in game_wrappers])
                    avg_lines = np.mean([wrapper.lines for wrapper in game_wrappers])

                    # Registrar en TensorBoard
                    self.writer.add_scalar('Average Reward', avg_reward, self.n_calls)
                    self.writer.add_scalar('Average Lines', avg_lines, self.n_calls)
                else:
                    print("Error: Algunos entornos no tienen 'game_wrapper' o les falta 'score'/'lines'.")
            except Exception as e:
                print(f"Excepción capturada en TensorboardCallback: {e}")
        return True


#NO FUNCIONA ESTA PARTE 