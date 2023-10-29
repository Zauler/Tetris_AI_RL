import gimnasio

# Crear una instancia del entorno
env = gimnasio.TetrisEnv('Tetris.gb')

# Reinicia el entorno
observation = env.reset()

# Verifica que la observación inicial tiene la forma correcta
assert observation.shape == (84, 84)
