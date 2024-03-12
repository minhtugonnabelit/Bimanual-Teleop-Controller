import pygame
import sys
 
# Initialize the pygame library
pygame.init()
 
# Setup joystick
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    raise Exception('No joystick found')
else:
    joystick = pygame.joystick.Joystick(0)  # NOTE: Change 0 to another number if multiple joysticks present
    joystick.init()
 
# Print joystick information
joy_name = joystick.get_name()
joy_axes = joystick.get_numaxes()
joy_buttons = joystick.get_numbuttons()
 
print(f'Your joystick ({joy_name}) has:')
print(f' - {joy_buttons} buttons')
print(f' - {joy_axes} axes')
 
# Main loop to check joystick functionality
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
 
    # Print buttons/axes info to the console
    button_info = [f'Button[{i}]:{joystick.get_button(i)}' for i in range(joy_buttons)]
    axis_info = [f'Axes[{i}]:{joystick.get_axis(i):.3f}' for i in range(joy_axes)]
 
    info_str = '--------------\n' + '\n'.join(button_info + axis_info) + '\n--------------\n'
    print(info_str)
 
    pygame.time.delay(50)  # Pause for 50 ms (equivalent to pause(0.05) in MATLAB)