#Importar procesamiento de GPU
import torch
# Importar tu entorno
from gimnasio import TetrisEnv
from evaluacion_TensorBoard import TensorboardCallback
# Stable Baselines
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv , SubprocVecEnv



# PyTorch esté utilizando CUDA si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

Empezar_de_cero = False #True Empezar de 0, False usar un modelo ya preentrenado. No puedez mezclar entornos en paralelo con simples.
velocity = 0 #Velocidad del emulador, 0 es lo máximo posible.
Activar_paralelo = True # False para no entrenar paralelos // True para paralelizar.
n_envs= 16 # Numero de entornos en paralelo.
modo_entrenar = True # True para entrenar // False para probar.
ciclos_entrenamiento = 10000 # Numero de ciclos de entrenamiento.

if __name__ == '__main__':
    
    ## Crea el callback
    #callback = TensorboardCallback(check_freq=1000, log_dir="./a2c_cartpole_tensorboard/")

    if Activar_paralelo and modo_entrenar:

        #ENTORNO PARALELO
        def make_ev():
           return TetrisEnv('Tetris.gb',vel=velocity)
        vec_env = SubprocVecEnv([make_ev for _ in range(n_envs)])

    else:
         
    ## ENTORNO SIMPLE
    #Crear el entorno
         env = TetrisEnv('Tetris.gb',vel=velocity)
    # Vectorizar el entorno (requerido por Stable Baselines)
         vec_env = DummyVecEnv([lambda: env])


    if modo_entrenar:


        if Empezar_de_cero:

                
                # Crear el modelo
                model = DQN("MlpPolicy", vec_env, verbose=1,learning_rate=0.0005, gamma=0.99, batch_size=64, target_update_interval=1000,
                            buffer_size=500000,  tensorboard_log="./a2c_cartpole_tensorboard/")
        else:
            # Cargar el modelo
            model = DQN.load("dqn_tetris")
            model.set_env(vec_env)

        try:
            for i in range(ciclos_entrenamiento):
                # Entrenar el modelo
                model.learn(total_timesteps=10000, reset_num_timesteps=False) #, callback=callback

                # Guardar el modelo
                model.save("dqn_tetris")

        except BaseException as e:

            # Guardar el modelo
            model.save("dqn_tetris")
            print("Interrumpido por error:", e)


        finally:
            # Guardar el modelo
            model.save("dqn_tetris")
            print("MODELO GUARDADO")

    else:

        # Cargar el modelo
        loaded_model = DQN.load("dqn_tetris",env=None)

        # Evaluar el modelo
        obs, info  = env.reset()
        while True:
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print(reward,terminated,truncated)
            if terminated or truncated:
                obs, info = env.reset()
