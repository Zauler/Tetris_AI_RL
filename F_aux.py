from pyboy import PyBoy, WindowEvent




def navegar_menu(velocidad_emulacion,objeto):
    
    for _ in range(velocidad_emulacion*6*60):
        objeto.tick()
    objeto.send_input(WindowEvent.PRESS_BUTTON_START)
    objeto.tick()
    objeto.send_input(WindowEvent.RELEASE_BUTTON_START)

    for _ in range(velocidad_emulacion*2*60):
        objeto.tick()
    objeto.send_input(WindowEvent.PRESS_BUTTON_START)
    objeto.tick()
    objeto.send_input(WindowEvent.RELEASE_BUTTON_START)   

    for _ in range(velocidad_emulacion*2*60):
        objeto.tick()
    objeto.send_input(WindowEvent.PRESS_BUTTON_START)
    objeto.tick()
    objeto.send_input(WindowEvent.RELEASE_BUTTON_START)   

    for _ in range(velocidad_emulacion*2*60):
        objeto.tick()
    objeto.send_input(WindowEvent.PRESS_BUTTON_START)
    objeto.tick()
    objeto.send_input(WindowEvent.RELEASE_BUTTON_START)   

    for _ in range(velocidad_emulacion*2*60):
        objeto.tick()
