import time

def debounce(wait):
    """
    Debounce decorator. Prevents a function from being called if it was called
    less than `wait` seconds ago.

    Args:
    wait (float): Debounce time interval in seconds.
    """
    def decorator(func):
        last_call = [0]

        def debounced_func(*args, **kwargs):
            current_time = time.time()
            if current_time - last_call[0] >= wait:
                last_call[0] = current_time
                return func(*args, **kwargs)

        return debounced_func
    return decorator

# Example usage
@debounce(0.3)  # Set the debounce interval to 0.3 seconds
def toggle_control():
    # This function should contain the logic to switch between the control modes
    # Example:
    # global current_pub
    # current_pub = pub_left_arm if current_pub == pub_right_arm else pub_right_arm
    print("Control toggled")
