from pynput.keyboard import Key, Controller
import time
keyboard = Controller()

def release():
	keyboard.release(Key.left)
	keyboard.release(Key.right)
	keyboard.release(Key.enter)
	
def stay(press_time):
	release()
	time.sleep(press_time) 

# def continu(press_time):
# 	press = press_time
# 	# release()
# 	time.sleep(press_time) 

def left(press_time):
	# release()	
	keyboard.press(Key.left)
	time.sleep(press_time)    
	# keyboard.release(Key.left)

def right(press_time):
	# release()	
	keyboard.press(Key.right)
	time.sleep(press_time)    
	# keyboard.release(Key.right)

def up(press_time):
	# release()
	keyboard.press(Key.up)
	time.sleep(press_time)    
	keyboard.release(Key.up)

def up_left(press_time):
	# release()
	keyboard.pressed(Key.left)
	keyboard.press(Key.up)
	time.sleep(press_time)    
	keyboard.release(Key.up)

def up_right(press_time):
	# release()
	keyboard.pressed(Key.right)
	keyboard.press(Key.up)
	time.sleep(press_time)    
	keyboard.release(Key.up)

# def down(press_time):
# 	# release()
# 	keyboard.press(Key.down)
# 	time.sleep(press_time)    
# 	# keyboard.release(Key.down)

def p(press_time):
	# release()
	keyboard.press(Key.enter)
	time.sleep(press_time)    
	keyboard.release(Key.enter)

def p_left(press_time):
	# release()
	keyboard.pressed(Key.left)
	keyboard.press(Key.enter)
	time.sleep(press_time)    
	keyboard.release(Key.enter)

def p_right(press_time):
	# release()	
	keyboard.pressed(Key.right)
	keyboard.press(Key.enter)
	time.sleep(press_time)    
	keyboard.release(Key.enter)

def p_up(press_time):
	# release()
	keyboard.pressed(Key.up)
	keyboard.press(Key.enter)
	time.sleep(press_time)    
	keyboard.release(Key.enter)

def p_down(press_time):
	# release()	
	keyboard.pressed(Key.down)
	keyboard.press(Key.enter)
	time.sleep(press_time)    
	keyboard.release(Key.enter)

