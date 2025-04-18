import time
import serial
import math


global start
global olda1
olda1=0
global oldq1i
oldq1i=0
global oldq2i
oldq2i=0

def listToString(s):
    str1=""
    for ele in s:
        str1+=ele
    return str1

class Motor:
	def __init__(self):
		self.time_of_change=0	#to be used in experiment mode
		self.angular_change=0	#to be used in experiment mode
		self.angle=0
		self.zero_position_angle=0
		self.sv=0	#to be used in experiment mode

	def updateSV(self,SV):
		self.sv=float(SV)

class pulsr:
	def __init__(self):
		# self.control_data=['0','0','0','0','0','0','0','\n']
		self.control_data=int(b'0000000')
		self.motor_data=str()
		self.upper=Motor()
		self.lower=Motor()
		self.upper_angle_sign=1
		self.lower_angle_sign=1
		self.xi=0
		self.yi=0
		self.x=0
		self.y=0
		self.r=0
		self.tether=0
		self.l1=0
		self.l2=0
		self.l3=0
		self.l4=0
		self.l5=0
		self.circleMode=False

	def forward_kinematics(self,q1i,q2i):
		global olda1
		global oldq1i
		global oldq2i
		try:
			qb = 180-(q2i-q1i)
			a = math.sqrt((self.l4-self.l5)**2 + self.l2**2 - 2*(self.l4-self.l5)*self.l2*math.cos(math.radians(qb)))	
			a1 = math.degrees(math.acos((a**2+self.l3**2-self.l1**2)/((2*a*self.l3))))
			oldq1i=q1i
			oldq2i=q2i
		except ValueError:
			print("ValueError...........")
			print('upperAngle',q1i,self.upper.angle)
			print('lowerAngle:',q2i,self.lower.angle)
			print('a:',a)
			q1i=oldq1i+0.1
			q2i=oldq2i+0.1
			qb = 180-(q2i-q1i)
			a = math.sqrt((self.l4-self.l5)**2 + self.l2**2 - 2*(self.l4-self.l5)*self.l2*math.cos(math.radians(qb)))	
			a1 = math.degrees(math.acos((a**2+self.l3**2-self.l1**2)/((2*a*self.l3))))
			# fail=True
			# while fail==True:
			# 	if 'n'==input():
			# 		fail=False
		p = math.sqrt(a**2 + self.l5**2 - 2*a*self.l5*math.cos(math.radians(qb+a1)))
		a2 = math.degrees(math.acos((self.l4**2 + p**2 - self.l2**2)/((2*self.l4*p))))
		xi = p*math.cos(math.radians(a2+q1i))
		yi = p*math.sin(math.radians(a2+q1i))
		return [xi,yi]

	def define_geometry(self,l1,l2,l3,l4,l5):
		'''
		l1 = float(input('l1: ')) upper link length
		l2 = float(input('l2: ')) lower link length
		l3 = float(input('l3: ')) length of link parallel to lower link
		l4 = float(input('l4: ')) full length of link parallel to upper link
		l5 = float(input('l5: ')) length of end effecor from parallelogram
		'''
		self.l1=l1
		self.l2=l2
		self.l3=l3
		self.l4=l4
		self.l5=l5

	def set_origin(self,upperq,lowerq):
		# self.upper.zero_position_angle=int(input("Enter initial upper link angle from baseplate: "))
		# self.lower.zero_position_angle=int(input("Enter initial lower link angle from baseplate: "))
		self.upper.zero_position_angle=upperq
		self.lower.zero_position_angle=lowerq
		self.xi,self.yi=self.forward_kinematics(self.upper.angle+self.upper.zero_position_angle,self.lower.angle+self.lower.zero_position_angle)
		return [self.xi,self.yi]

	def get_effector_coordinate(self):
		# self.send_command()	#remember to uncomment
		# self.update_motor_angles()	#remember to uncomment
		x,y=self.forward_kinematics(self.upper.angle,self.lower.angle)
		self.x=x-self.xi
		self.y=y-self.yi

	def initialize_communication(self,baudrate,port):
		self.pulsr2_coms=serial.Serial()
		self.pulsr2_coms.baudrate=baudrate
		self.pulsr2_coms.port=str(port)
		self.pulsr2_coms.open()
		self.check_communication()

	def close_communication(self):
		self.pulsr2_coms.close()

	def check_communication(self):
		if self.pulsr2_coms.isOpen()==True:
			print("pulsr communication check succesful")
			return True
		else:
			print("pulsr communication check failed, reinitialize communication")
			return False

	def send_command(self):
		if self.pulsr2_coms.isOpen()==False:
			self.pulsr2_coms.open()
			print("pulsr communication not open, thus can't write command")
		else:
			# self.control_data[6]='1'	#enable data transfer
			# self.pulsr2_coms.reset_input_buffer()
			# self.pulsr2_coms.write(bytes(listToString(self.control_data), 'ascii'))
			self.pulsr2_coms.reset_input_buffer()
			self.pulsr2_coms.reset_output_buffer()
			self.pulsr2_coms.write(bytes([self.control_data]))
			self.motor_data=str(self.pulsr2_coms.read_until())	#reading from arduino containing required angles for next computation
			print(self.motor_data)
			#disable usgage of update_motor_angles change function until you redefine it
			# # self.update_motor_angles_change()

	def enable_upper_motor(self):
		self.control_data|=2 	#enable up motor
		self.send_command()

	def disable_upper_motor(self):
		self.control_data&=125	#disable upper motor
		self.send_command()

	def enable_lower_motor(self):
		self.control_data|=8	#enable lower motor
		self.send_command()

	def disable_lower_motor(self):
		self.control_data&=119	#disable lower motor
		self.send_command()

	def motion1(self,en_time):
		#north direction motion
		# self.control_data[5]='1'	#enable upper motor
		# self.control_data[4]='0'	#set upper motor to F direction
		self.control_data|=2 	#enable up motor
		self.control_data&=123	#set upper motor to F direction	
		self.send_command()
		if en_time==0:
			# self.update_motor_angles()
			pass
		else:
			time.sleep(en_time)
			self.disable_motions()

	def motion2(self,en_time):
		#south direction motion
		# self.control_data[5]='1'	#enable upper motor
		# self.control_data[4]='1'	#set upper motor to R direction
		self.control_data|=2 	#enable up motor
		self.control_data|=4	#set upper motor to R direction
		self.send_command()
		if en_time==0:
			# self.update_motor_angles()
			pass
		else:
			time.sleep(en_time)
			self.disable_motions()

	def motion3(self,en_time):
		#east direction motion
		# self.control_data[3]='1'	#enable lower motor
		# self.control_data[2]='0'	#set upper motor to F direction
		self.control_data|=8	#enable lower motor
		self.control_data&=111	#set upper motor to F direction
		self.send_command()
		if en_time==0:
			# self.update_motor_angles()
			pass
		else:
			time.sleep(en_time)
			self.disable_motions()

	def motion4(self,en_time):
		#west direction motion
		# self.control_data[3]='1'	#enable lower motor
		# self.control_data[2]='1'	#set upper motor to R direction
		self.control_data|=8	#enable lower motor
		self.control_data|=16	#set upper motor to R direction
		self.send_command()
		if en_time==0:
			# self.update_motor_angles()
			pass
		else:
			time.sleep(en_time)
			self.disable_motions()

	def disable_motions(self):
		# self.control_data[5]='0'	#disable upper motor
		# self.control_data[3]='0'	#disable lower motor
		self.control_data&=125	#disable upper motor
		self.control_data&=119	#disable lower motor
		self.send_command()

	def update_motor_angles(self):
		self.send_command()
		self.upper_angle_sign=int(self.motor_data[self.motor_data.find('c')+1:self.motor_data.find('d')])
		if self.upper_angle_sign==1:
			# print("upper pve")
			self.upper.angle=self.upper.zero_position_angle+float(self.motor_data[self.motor_data.find('u')+1:self.motor_data.find('c')])
		else:
			# print("upper -ve")
			self.upper.angle=(-1)*float(self.motor_data[self.motor_data.find('u')+1:self.motor_data.find('c')])
			self.upper.angle=self.upper.zero_position_angle+self.upper.angle
		
		self.lower_angle_sign=int(self.motor_data[self.motor_data.find('v')+1:self.motor_data.find('v')+2])
		if self.lower_angle_sign==1:
			# print("lower pve")
			self.lower.angle=self.lower.zero_position_angle+float(self.motor_data[self.motor_data.find('d')+1:self.motor_data.find('v')])
		else:
			# print("lower -ve")
			self.lower.angle=(-1)*float(self.motor_data[self.motor_data.find('d')+1:self.motor_data.find('v')])
			self.lower.angle=self.lower.zero_position_angle+self.lower.angle

	#remember to update definition of update_angular_changes function
	def update_motor_angles_change(self):
		self.upper.angular_change=abs(self.upper.angle-float(self.motor_data[self.motor_data.find('u')+1:self.motor_data.find('c')]))
		self.lower.angular_change=abs(self.lower.angle-float(self.motor_data[self.motor_data.find('d')+1:self.motor_data.find('v')]))

if __name__=='__main__':
	neuro=pulsr()
	neuro.initialize_communication(2000000,'COM10')
	neuro.define_geometry(15,40,40,50,35)
	time.sleep(3)
	print(neuro.set_origin())