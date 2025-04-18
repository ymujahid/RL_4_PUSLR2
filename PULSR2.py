from statistics import mode
import pyautogui
import pygame
import math 
import os
from pulsr2API import *
import serial.tools.list_ports
import random
from pygame.locals import *

#get screen_width and screen_height
screen_width,screen_height=pyautogui.size()
# workspace_height=49
# workspace_width=115
starting_dialog_box_size=(int(screen_width/2.6),int(screen_height/4))
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (int(screen_width/2)-int(screen_width/6),int(screen_height/2)-int(screen_height/6))

pygame.init()

#select arduino comPort
comPort=[
	p.device 
	for p in serial.tools.list_ports.comports()
	if 'Arduino' in p.description
]
pulsr2=pulsr()
pulsr2.initialize_communication(2000000,comPort[0])	

#initialize starting dialog box windows
starting_dialog_box=pygame.display.set_mode(starting_dialog_box_size)
background=pygame.display.set_mode(starting_dialog_box_size)	#foundation surface definition
screen=pygame.display.set_mode(starting_dialog_box_size)	#starting dialog box surface definition

#caption and text defineition for starting dialog box
pygame.display.set_caption('PULSR2.0')	 #make caption of current display which is screen-> starting_dialog_box_size
oaufont = pygame.font.SysFont('Arial', 25)
oaufont2 = pygame.font.SysFont('Arial', 15)
oaufont3 = pygame.font.SysFont('Arial', 10)
meaningtext = oaufont.render('Platform for Upper Limb Stroke Rehabilitation', False, (0, 0, 0))
oautext = oaufont2.render('Obafemi Awolowo University, Ile-Ife', False, (0, 0, 0))
dialog_message = oaufont3.render('PULSR2 starting up.........', False, (0, 0, 0))
clock=pygame.time.Clock()

background.fill((255,255,255))	#fill dialog box background with white
screen.blit(meaningtext,(10,60))
screen.blit(oautext,(135,90))
screen.blit(dialog_message,(7,170))
pygame.display.update()
clock.tick(20)
time.sleep(1)

'''
	code to display pulsr2 starting small dialog box
		i.check communication status and report on dialog box
'''
#i.check pulsr2 communication status and report on dialog box
if pulsr2.check_communication()==True:
	dialog_message = oaufont3.render('communication port active', False, (0, 0, 0))
	background.fill((255,255,255))	#fill starting dialog box background with white
	screen.blit(meaningtext,(10,60))
	screen.blit(oautext,(135,90))
	screen.blit(dialog_message,(7,170))
	pygame.display.update()
	clock.tick(20)
	time.sleep(1)
else:
	dialog_message = oaufont3.render('communication port not active', False, (0, 0, 0))
	background.fill((255,255,255))	#fill dialog box background with white
	# background.blit(image,(0,0))
	screen.blit(meaningtext,(10,60))
	screen.blit(oautext,(135,90))
	screen.blit(dialog_message,(7,170))
	pygame.display.update()
	clock.tick(20)
	time.sleep(1)
	pygame.quit()
	quit()


'''
	code to run motor and encoder readings tests and report on dialog box
		i.send control-data to arduino and receive motor data from arduino
			...if succesful report communication test passed
			...if failed report communication test failed
		ii.test motions and reported encoder readings for correct behaviour
'''
#i.send control-data to arduino and receive motor data from arduino
pulsr2.send_command()
if pulsr2.motor_data==str():
	dialog_message = oaufont3.render('communication test failed', False, (0, 0, 0))
	background.fill((255,255,255))	#fill dialog box background with white
	# background.blit(image,(0,0))
	screen.blit(meaningtext,(10,60))
	screen.blit(oautext,(135,90))
	screen.blit(dialog_message,(7,170))
	pygame.display.update()
	clock.tick(20)
	time.sleep(1)
	pygame.quit()
	quit()
else:
	dialog_message = oaufont3.render('communication test passed', False, (0, 0, 0))
	background.fill((255,255,255))	#fill dialog box background with white
	# background.blit(image,(0,0))
	screen.blit(meaningtext,(10,60))
	screen.blit(oautext,(135,90))
	screen.blit(dialog_message,(7,170))
	pygame.display.update()
	clock.tick(20)
	time.sleep(1)
#ii.test motions and reported encoder readings for correct behaviour
dialog_message = oaufont3.render('testing motors............', False, (0, 0, 0))
background.fill((255,255,255))	#fill dialog box background with white
screen.blit(meaningtext,(10,60))
screen.blit(oautext,(135,90))
screen.blit(dialog_message,(7,170))
pygame.display.update()
clock.tick(20)
time.sleep(1)

dummy=pulsr2.upper.angle
pulsr2.motion1(1)
pulsr2.update_motor_angles()
check=True
	#check for current upper motor angle if it is greater or lesser than previous
	#if correct set check variable to True else False
if check==True:
	dummy=pulsr2.upper.angle
	pulsr2.motion2(1)
	pulsr2.update_motor_angles()
	#check for current upper motor angle if it is greater or lesser than previous
	#if correct set check variable to True else False
if check==True:
	dummy=pulsr2.lower.angle
	pulsr2.motion3(1)
	pulsr2.update_motor_angles()
	#check for current lower motor angle if it is greater or lesser than previous
	#if correct set check variable to True else False
if check==True:
	dummy=pulsr2.lower.angle
	pulsr2.motion4(1)
	pulsr2.update_motor_angles()
	#check for current lower motor angle if it is greater or lesser than previous
	#if correct close dialog_box and continue 
	#else report motors/encoders test failed, quit pygame and program
time.sleep(5)

'''
	all tests succesful,thus:
	i. define pulsr kinematics parameters and objects
	ii. define gui parameters
	iii.
'''
pygame.display.quit()
time.sleep(1)
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (int(0),int(0))
time.sleep(1)
pygame.display.init()
#i.define pulsr kinmematic parameters
pulsr2.define_geometry(22,43,43,65,42)
pulsr2.set_origin(90,225)#might be 100,180
#ii.define gui parameters and objects
black = (0,0,0)
white = (255,255,255)
cream = (255,250,170)
purple = (102,0,204)
orange = (255,153,51)
green=(255,0,255)
green2 = (0,204,0)
blue=(0,0,255)
red=(255,0,0)
grey=(222,222,222)
gui_size=((int(screen_width),int(screen_height)))
game_interface=pygame.display.set_mode(gui_size)
gameDisplay=pygame.display.set_mode(gui_size)
gui=pygame.display.set_mode(gui_size)
pygame.display.set_caption('PULSR2.0')
pygame.font.init()
pulsrfont = pygame.font.SysFont('Arial Bold', 100)
pulsrtext = pulsrfont.render('PULSR', False, white)
oaufont = pygame.font.SysFont('Arial', 15)
meaningtext = oaufont.render('Platform for Upper Limb Stroke Rehabilitation', False, white)
oautext = oaufont.render('Obafemi Awolowo University, Ile-Ife', False, white)
pygame.display.set_caption('Image')
image = pygame.image.load(r'OpenBCI figure.jpg')

new_x=pulsr2.x
new_y=pulsr2.y
yOffset=+18.5
xOffset=0
screen_workspace_height=screen_height
screen_workspace_width=screen_width*0.7
# old_x=pulsr2.x*(screen_workspace_width/50)
# old_y=pulsr2.y*(screen_workspace_height/50)
# pulsr2.get_effector_coordinate()
global running	#running variable stores software running state
running=True
global fresh
fresh=True
global matc	#matc variables is true if there is a match between pulsr end-effector and target point
matc=True
global trials	#trials variable stores no of iteration of trial to match end-effector with target point
global score	#score variable storesperformance score of user
trials=1
score=0
global level	#level variable determines level of game to load
level=0
global pulsr_mode
pulsr_mode='p'

'''
	i.select game level
	ii.select game mode
	iii.runSession function definition, which loads the GUI for every session
	iv.waitForKey function defintion
	v.game loop entry
'''

#i.select game level
gameDisplay.fill(black)
gui.blit(pulsrfont.render('select level',False, white), (50, 100))
pygame.display.update()
nxt=False
while nxt==False:
	for event in pygame.event.get():
		if event.type == MOUSEBUTTONUP:
			None
		if event.type==KEYDOWN:
			if event.key==K_SPACE:
				pause=False
				running=True
			if event.key==K_ESCAPE:
				pygame.quit()
				quit()
			if event.key==K_q:
				pygame.quit()
				quit()
			if event.key==K_0:
				level=0
				nxt=True
			if event.key==K_1:
				level=1
				nxt=True
time.sleep(1)

#i.select game mode
gameDisplay.fill(black)
gui.blit(pulsrfont.render('select mode',False, white), (50, 100))
pygame.display.update()
nxt=False
while nxt==False:
	for event in pygame.event.get():
		if event.type == MOUSEBUTTONUP:
			None
		if event.type==KEYDOWN:
			if event.key==K_SPACE:
				pause=False
				running=True
			if event.key==K_ESCAPE:
				pygame.quit()
				quit()
			if event.key==K_q:
				pygame.quit()
				quit()
			if event.key==K_p:#passive mode
				pulsr_mode='p'
				nxt=True
			if event.key==K_a:#assistive mode
				pulsr_mode='a'
				nxt=True
			if event.key==K_r:#reactive mode
				pulsr_mode='r'
				nxt=True

#ii.runSession function definition, which loads the GUI for every session
def runSession():
	#variables definition
	global running
	global new_x
	global new_y
	global old_x
	global old_y
	global fresh
	global oldTheta	#used to control moving circle location
	global currentTheta #used to control moving cirlce locaation
	global matc
	global score
	global trials
	locationIterator = 0	#ranges from 0 to 250 to create 250 point moving circle whose motion makes a full circle

	#clear surface and add necessary text on every game loop iteration
	gameDisplay.fill(black)
	gui.blit(pulsrtext, (30, 40))
	gui.blit(meaningtext, (20, 150))
	gui.blit(oautext, (55, 180))
	radius = screen_workspace_height*0.35
	screenCenter=[screen_width*0.5,screen_height*0.5]
	movingCircleCenter = [0, 0]
	oldTheta=0
	currentTheta=0

	circleColor=purple

	while running:
		#update end effector coordinates which gives old effector coordinates on every running loop iteration
		pulsr2.send_command()
		pulsr2.update_motor_angles()
		pulsr2.get_effector_coordinate()
		#definition and check of pulsr control keys state on every running loop iteration 
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pulsr2.disable_motions()
				running = False
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					pulsr2.disable_motions()
					running = False
				if event.key == pygame.K_q:
					pulsr2.disable_motions()
					pygame.quit()
					quit()
				if event.key == pygame.K_r:
					pulsr2.upper.angle=0
					pulsr2.lower.angle=0
					pulsr2.set_origin()
				if event.key == pygame.K_w:
					pulsr2.motion1(0)
				if event.key == pygame.K_s:
					pulsr2.motion2(0)
				if event.key == pygame.K_a:
					pulsr2.motion3(0)
				if event.key == pygame.K_d:
					pulsr2.motion4(0)
				if event.key == pygame.K_m:
					pulsr2.control_data &= 191	#computercontrolled mode
					pulsr2.send_command()
				if event.key == pygame.K_u:
					pulsr2.control_data |= 64	#user controlled mode
					pulsr2.send_command()
				if event.key == pygame.K_c:
					pulsr2.control_data |= 32 #computer controlled mode move in circle
					pulsr2.send_command()
					pulsr2.circleMode=True
				if event.key == pygame.K_x:
					if pulsr2.circleMode==True:
						pulsr2.disable_motions()
						pulsr2.circleMode=False
					pulsr2.control_data &= 223 #computer controlled mode move according to pre-defined keys direction
					pulsr2.send_command()
				if event.key == pygame.K_t:
					pulsr2.disable_upper_motor()	#disable upper motor
				if event.key == pygame.K_g:
					pulsr2.enable_upper_motor()	#enable upper motor
				if event.key == pygame.K_y:
					pulsr2.disable_lower_motor()	#disable lower motor
				if event.key == pygame.K_h:
					pulsr2.enable_lower_motor()	#enable lower motor
		
		'''
			i.compute score based on matc and level variable
			ii.compute next position coordinate for movingCircle based on level variable
		'''
		if level==0:
			if matc==True:
				score=score+1
			radius=screen_workspace_height*0.2
			angleInDegrees = (locationIterator/250.0)*360
			oldAngleInDegrees = ((locationIterator-1)/250.0)*360
			currentTheta=math.radians(angleInDegrees)
			oldTheta = math.radians(oldAngleInDegrees)
		elif level==1:
			if matc==True:
				score=score+1
				radius=screen_workspace_height*random.uniform(0.0,0.3)
				oldTheta = currentTheta
				currentTheta=math.radians(random.randint(0,360))
				matc=False

		#compute and clear old coordinates of movingCircle
		movingCircleCenter[0] = screenCenter[0]+(radius*math.cos(oldTheta))	#y-coordinates on screen for center translates to horizontal
		movingCircleCenter[1] = screenCenter[1]-(radius*math.sin(oldTheta))	#x-coordinates on screen for center translates to vertical
		clearDot = pygame.draw.circle(gameDisplay, black, (movingCircleCenter[0], movingCircleCenter[1]), 12, width=12) #clear current position of movingCircle

		'''
			definition of level specific display statics
			i.draw circular path for moving circle  in level 0
		'''
		#i.draw circular path for moving circle  in level 0
		if level==0:
			redrawGray = pygame.draw.circle(gameDisplay, circleColor, (screenCenter[0],screenCenter[1]), radius, width=10)	#draw circular path for movingCircle

		#compute and draw new coordinates of movingCircle
		movingCircleCenter[0] = screenCenter[0] + (radius * math.cos(currentTheta))
		movingCircleCenter[1] = screenCenter[1] - (radius * math.sin(currentTheta))
		newDot = pygame.draw.circle(gameDisplay, white, (movingCircleCenter[0],movingCircleCenter[1]), 12, width=12) #draw current position

		#clear old effector coordinate
		old_x=new_x
		old_y=new_y
		clearOldCoordinate=pygame.draw.circle(gameDisplay, black, (screenCenter[0]+old_y, screenCenter[1]+old_x), 8, width=8)

		#update effector coordinate to give new effector eoordinates
		pulsr2.send_command()
		pulsr2.update_motor_angles()
		pulsr2.get_effector_coordinate()

		#level specific matc condition checking and altering of display parameters and color according to level and matc condition
		matc=False
		global pointColor
		new_x=(pulsr2.x+xOffset)*(screen_workspace_height/61)
		new_y=(pulsr2.y-yOffset)*(screen_workspace_width/61)
		if abs(math.sqrt((new_x**2)+(new_y**2))-radius)<=100:
			if abs((screenCenter[0]+new_y)-movingCircleCenter[0])<=70 and abs((screenCenter[1]+new_x)-movingCircleCenter[1])<=70:
				matc=True
				pointColor=green2
				circleColor=green2
				if level==1:
					newDot = pygame.draw.circle(gameDisplay, circleColor, (movingCircleCenter[0],movingCircleCenter[1]), 12, width=12) #draw current position
			else:
				pointColor=orange
				circleColor=orange
				if level==1:
					newDot = pygame.draw.circle(gameDisplay, circleColor, (movingCircleCenter[0],movingCircleCenter[1]), 12, width=12) #draw current position
		else:
			pointColor=red
			circleColor=purple
		drawNewCoordinate=pygame.draw.circle(gameDisplay, pointColor, (screenCenter[0]+new_y, screenCenter[1]+new_x), 8, width=8)
		if fresh==True:
			fresh=False
		else:
			drawPathLine=pygame.draw.line(gameDisplay,green2,(screenCenter[0]+old_y, screenCenter[1]+old_x), (screenCenter[0]+new_y, screenCenter[1]+new_x), width=3)
		leftBorder=pygame.draw.line(gameDisplay,white,(screenCenter[0]-(screen_workspace_width/2),screenCenter[1]+(screen_workspace_height/2)),(screenCenter[0]-(screen_workspace_width/2),screenCenter[1]-(screen_workspace_height/2)),width=6)
		rightBorder=pygame.draw.line(gameDisplay,white,(screenCenter[0]+(screen_workspace_width/2),screenCenter[1]+(screen_workspace_height/2)),(screenCenter[0]+(screen_workspace_width/2),screenCenter[1]-(screen_workspace_height/2)),width=6)

		'''
			To be used in assisive and reactive mode
        #     ACTIVITIES PER LOOP
        '''
		'''
			ASSISTIVE MODE
		'''
		if pulsr_mode=='a':
			# i.save current effector coordinate on gui and movingCircle coordinate on gui
			effector_y=screenCenter[1]+new_x
			effector_x=screenCenter[0]+new_y
			moving_x=movingCircleCenter[0]
			moving_y=movingCircleCenter[1]
			# ii.compute distance and angle between effector coordinate and movingCircle coordinate
			dist=0
			angle=0
			dist=math.sqrt(abs(effector_x-moving_x)**2 + abs(effector_y-moving_y)**2)
			angle=math.degrees(math.atan2(moving_y-effector_y,moving_x-effector_x))
			if angle==0 or angle==-0:
				angle=0
			elif angle<0:
				angle=abs(angle)
			elif angle>0:
				angle=180+(180-angle)
			pygame.draw.rect(gameDisplay,black,(1650,400,320,300))
			pygame.display.update()
			clock.tick(20)
			gui.blit(pulsrfont.render(str(float("{:.2f}".format(angle))),False, white), (1650, 400))
			pygame.display.update()
			clock.tick(20)

			# iii. determine the difference between the angles
				#determine difference between closestTheta and rotaryTheta

			# iv. move the motors closer to the angle
				#move rotaryTheta to closestTheta

			# v. advance the iterator to take on the value closest to the index of the current location in the location array


			#vi. Advance the arc of the actual movement
		'''
			REACTIVE MODE
		'''
        # i. retrieve the angles of the encoders
            #compute theta from angle's 0 and 1 as rotaryTheta

        # ii. check through the array of 250 angles to determine the closest
            #compute closestTheta between currentTheta and rotaryTheta

        # iii. determine the difference between the angles
            #determine difference between closestTheta and rotaryTheta

        # iv. move the motors closer to the angle
            #move rotaryTheta to closestTheta

        # v. advance the iterator to take on the value closest to the index of the current location in the location array


        #vi. Advance the arc of the actual movement

		'''
			alter behaviour aof moving circle according to level
			level0=circular iteration fom 0 t0 250
			level1=random
		'''
		if level==0:
			if locationIterator==250:
				locationIterator=0
			else:
				locationIterator+=1

		pygame.display.update()
		clock.tick(20)

		#update effector coordinates
		pulsr2.send_command()
		pulsr2.update_motor_angles()
		pulsr2.get_effector_coordinate()

		#level specific delay after matc condition True, i.e effector matchtarget point
		if matc==True:
			if level==0:
					time.sleep(0.25)
			elif level==1:
					time.sleep(3)
		clearDot = pygame.draw.circle(gameDisplay, black, (movingCircleCenter[0], movingCircleCenter[1]), 12, width=12) #clear current movingCircle point

		#score calculation and display
		pygame.draw.rect(gameDisplay,black,(10,400,320,300))
		pygame.display.update()
		clock.tick(20)
		gui.blit(pulsrfont.render(str(float("{:.2f}".format(score/trials))),False, white), (50, 400))
		pygame.display.update()
		clock.tick(20)
		trials=trials+1
		if trials>=4:
			if trials%4==0:
				trials=int(trials/4)

	return

#ii.waitForKey function definition
def waitForKey():
    global running
    while running == False:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = True
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
    return


#iii.game loop entry
while True:
    if running == False:
        print("running Wait function")
        waitForKey()
    if running == True:
        print("running Rehab function")
        runSession()
