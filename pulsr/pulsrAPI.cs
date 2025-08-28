using System;
using System.Collections.Generic;
using System.IO.Ports;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Management;



namespace PULSR_3
{


    public class Motor
    {
        public int angle { get; set; }
        public int target_speed { get; set; }
        public int link_force { get; set; }
        public int fore_threshold { get; set; }

        public Motor()
        {
            angle = 0;
            target_speed = 10;
            link_force = 0;
            fore_threshold = 0;
        }
    }

    public class pulsr
    {
        ///// INPUT VARIABLES AND COMMUNICATION DATA ///////
        public int control_data;
        public byte[] motor_data { get; set; }
        public byte[] force_data { get; set; }
        public int wanted_load_cell;
        private bool circleMode;

        ///// MOTOR INSTANCES  ///////
        public Motor upper;
        public Motor lower;

        ////// KINEMATICS PARAMETERS  ///////
        public double ll;
        public double lu;
        public double le;
        public double e1;
        public double e2;
        public int xi;
        public int yi;
        public int x;
        public int y;
        public double offset_angle;

        public SerialPort load_cell_coms;
        public SerialPort encoder_coms;
        public SerialPort pulsr2_coms;

        public string load_port;
        public string enc_port;
        public string motor_port;

        private const int baudrate = 2000000;

        public bool front { get; private set; }
        public bool back { get; private set; }
        public bool left { get; private set; }
        public bool right { get; private set; }

        public pulsr()
        {
            control_data = 0;
            //motor_data = "";
            //force_data = 0;
            wanted_load_cell = 0;
            circleMode = false;

            upper = new Motor();
            lower = new Motor();

            ////// KINEMATICS PARAMETERS INITIALIZATION ///////
            ll = 26;
            lu = 26;
            le = 26;
            e1 = 0;
            e2 = 0;
            xi = 0;
            yi = 0;
            x = 0;
            y = 0;
            offset_angle = 20;

            /// Read the USB serial numbers from a file ///
            string usbSerialNos;
            using (System.IO.StreamReader usbFile = new System.IO.StreamReader("usb_ser.txt"))
            {
                usbSerialNos = usbFile.ReadToEnd();
                Console.WriteLine(usbSerialNos);
            }
            string[] portNumbers = usbSerialNos.Split();
            load_port = portNumbers[0];
            enc_port = portNumbers[2];
            motor_port = portNumbers[4];
            var ports = SerialPort.GetPortNames();
            foreach (var port in ports)
            {
                using (var serialPort = new SerialPort(port))
                {
                    var serialNumber = GetSerialNumber(serialPort);
                    if (serialNumber == load_port)
                    {
                        load_port = port;
                    }
                    else if (serialNumber == enc_port)
                    {
                        enc_port = port;
                    }
                    else if (serialNumber == motor_port)
                    {
                        motor_port = port;
                    }
                }
            }
            Console.WriteLine($"{load_port} {enc_port} {motor_port}");

            string GetSerialNumber(SerialPort serialPort)
            {
                try
                {
                    serialPort.Open();
                    return serialPort.ReadExisting().Trim();
                }
                catch (Exception)
                {
                    return null;
                }
                finally
                {
                    serialPort.Close();
                }
            }
        }

        /// <summary>
        /// COMMUNICATION DEFINITIONS
        /// </summary>
        public void InitializeCommunication()
        {
            load_cell_coms = new SerialPort(load_port, baudrate);
            load_cell_coms.Open();

            encoder_coms = new SerialPort(enc_port, baudrate);
            encoder_coms.Open();

            pulsr2_coms = new SerialPort(motor_port, baudrate);
            pulsr2_coms.Open();

            Console.WriteLine("load_cell_coms: " + load_port);
            Console.WriteLine("encoder_coms: " + enc_port);
            Console.WriteLine("motor_coms: " + motor_port);

            CheckCommunication();
        }

        public void CloseCommunication()
        {
            pulsr2_coms.Close();
            load_cell_coms.Close();
            encoder_coms.Close();
        }

        public bool CheckCommunication()
        {
            if (load_cell_coms.IsOpen)
            {
                Console.WriteLine("load cell board connected");
            }
            else
            {
                Console.WriteLine("load cell board not found");
                return false;
            }

            if (encoder_coms.IsOpen)
            {
                Console.WriteLine("encoder board connected");
            }
            else
            {
                Console.WriteLine("encoder board not found");
                return false;
            }

            if (pulsr2_coms.IsOpen)
            {
                Console.WriteLine("pulsr communication check successful");
                return true;
            }
            else
            {
                Console.WriteLine("pulsr communication check failed, restart system to reinitialize communication");
                return false;
            }
        }

        public void SendCommand()
        {
            if (!pulsr2_coms.IsOpen)
            {
                pulsr2_coms.Open();
                Console.WriteLine("pulsr communication not open, thus can't write command");
            }
            else
            {
                pulsr2_coms.DiscardInBuffer();
                pulsr2_coms.DiscardOutBuffer();
                pulsr2_coms.Write(new byte[] { (byte)control_data }, 0, 1);
                pulsr2_coms.DiscardInBuffer();
                pulsr2_coms.DiscardOutBuffer();
                pulsr2_coms.Write(new byte[] { (byte)Math.Abs(upper.target_speed) }, 0, 1);
                pulsr2_coms.DiscardInBuffer();
                pulsr2_coms.DiscardOutBuffer();
                pulsr2_coms.Write(new byte[] { (byte)Math.Abs(lower.target_speed) }, 0, 1);


            }
        }

        public void UpdateLinkForce()
        {
            if (!load_cell_coms.IsOpen)
            {
                load_cell_coms.Open();
                Console.WriteLine("pulsr communication not open, thus can't write command, restart system");
            }
            else
            {
                load_cell_coms.DiscardInBuffer();
                load_cell_coms.DiscardOutBuffer();
                load_cell_coms.Write(new byte[] { (byte)wanted_load_cell }, 0, 1);
                Thread.Sleep(50);
                if (force_data == null || force_data.Length != 4)
                {
                    force_data = new byte[4];
                }
                load_cell_coms.Read(force_data, 0, 4);
            }
        }

        public void UpdateAngles()
        {
            if (!encoder_coms.IsOpen)
            {
                encoder_coms.Open();
                Console.WriteLine("pulsr communication not open, thus can't write command, restart system");
            }
            else
            {
                encoder_coms.DiscardInBuffer();
                encoder_coms.DiscardOutBuffer();
                encoder_coms.Write(new byte[] { 0 }, 0, 1);
                Thread.Sleep(50);
                if (motor_data == null || motor_data.Length != 5)
                {
                    motor_data = new byte[5];
                }
                encoder_coms.Read(motor_data, 0, 5);
            }
        }

        public void ResetAngles()
        {
            if (!encoder_coms.IsOpen)
            {
                encoder_coms.Open();
                Console.WriteLine("pulsr communication not open, thus can't write command, restart system");
            }
            else
            {
                encoder_coms.Write(new byte[] { 255 }, 0, 1);
                Thread.Sleep(50);
                if (motor_data == null || motor_data.Length != 5)
                {
                    motor_data = new byte[5];
                }
                encoder_coms.Read(motor_data, 0, 5);
            }
        }

        /// <summary>
        /// MODES
        /// </summary>
        public void ComputerMode()
        {
            control_data &= 63;
            SendCommand();
        }

        public void CircleMode()
        {
            control_data |= 32;
            ComputerMode();
        }

        public void KeyMode()
        {
            control_data &= 95;
            ComputerMode();
        }

        public void UserMode()
        {
            control_data |= 64;
            SendCommand();
        }

        /// <summary>
        /// PULSR SENSOR DATA UPDATES
        /// </summary> 
        public int UpdateUpperLoadCell()
        {
            wanted_load_cell = 1;
            UpdateLinkForce();
            int h = (force_data[1] << 2) | (force_data[2] >> 3);
            int l = (force_data[2] << 5) | (force_data[3]);
            upper.link_force = (h << 8) + l;
            //Console.WriteLine(upper.link_force);
            return upper.link_force;
        }

        public int UpdateLowerLoadCell()
        {
            wanted_load_cell = 2;
            UpdateLinkForce();
            int h = (force_data[1] << 2) | (force_data[2] >> 3);
            int l = (force_data[2] << 5) | (force_data[3]);
            lower.link_force = (h << 8) + l;
            //Console.WriteLine(lower.link_force);
            return lower.link_force;
        }

        public void UpdateMotorAngles()
        {
            UpdateAngles();
            upper.angle = (motor_data[1] << (8)) + motor_data[2];
            lower.angle = (motor_data[3] << 8) + motor_data[4];
            // remove later //
            //Console.WriteLine("angles: " + upper.angle + ", " + lower.angle);
        }

        public void UpdateSensorData()
        {
            UpdateLowerLoadCell();
            UpdateUpperLoadCell();
            UpdateMotorAngles();
            Console.WriteLine("angles: " + upper.angle + ", " + lower.angle);
            Console.WriteLine("force: " + upper.link_force + ", " + lower.link_force);
        }

        /// <summary>
        /// MOTOR COMMANDS
        /// </summary>
        public void SetMotorDirections(bool front, bool back, bool left, bool right)
        {
            this.front = front;
            this.back = back;
            this.left = left;
            this.right = right;
        }

        public void EnableUpperMotor()
        {
            control_data |= 2;  //enable up motor
            SendCommand();
        }

        public void DisableUpperMotor()
        {
            upper.target_speed = 0;
            control_data &= 125;   //disable upper motor
            SendCommand();
        }

        public void EnableLowerMotor()
        {
            control_data |= 8;    //enable lower motor
            SendCommand();
        }

        public void DisableLowerMotor()
        {
            lower.target_speed = 0;
            control_data &= 119;   //disable lower motor
            SendCommand();
        }

        public void UpperMotorCW(int en_time, int speed)
        {
            upper.target_speed = Math.Abs(speed);
            control_data |= 2;    //enable up motor
            control_data &= 123;  //set upper motor to F direction	
            SendCommand();
            if (en_time != 0)
            {
                Thread.Sleep(en_time);
                DisableUpperMotor();
            }
        }

        public void UpperMotorCCW(int en_time, int speed)
        {
            upper.target_speed = Math.Abs(speed);
            control_data |= 2;  //enable up motor
            control_data |= 4;  //set upper motor to R direction
            SendCommand();
            if (en_time != 0)
            {
                Thread.Sleep(en_time);
                DisableLowerMotor();
            }
        }

        public void LowerMotorCW(int en_time, int speed)
        {
            lower.target_speed = Math.Abs(speed);
            control_data |= 8;    //enable lower motor
            control_data &= 111;  //set lower motor to F direction
            SendCommand();
            if (en_time != 0)
            {
                Thread.Sleep(en_time);
                DisableUpperMotor();
            }
        }

        public void LowerMotorCCW(int en_time, int speed)
        {
            lower.target_speed = Math.Abs(speed);
            control_data |= 8;   //enable lower motor
            control_data |= 16;  //set lower motor to R direction
            SendCommand();
            if (en_time != 0)
            {
                Thread.Sleep(en_time);
                DisableLowerMotor();
            }
        }

        public void UpdateMotorSpeed(int upper_target, int lower_target)
        {
            if (upper_target == 0)
            {
                DisableUpperMotor();
            }
            else if (upper_target < 0)
            {
                UpperMotorCW(0, upper_target);
            }
            else if (upper_target > 0)
            {
                UpperMotorCCW(0, upper_target);
            }

            if (lower_target == 0)
            {
                DisableLowerMotor();
            }
            else if (lower_target < 0)
            {
                LowerMotorCW(0, lower_target);
            }
            else if (lower_target > 0)
            {
                LowerMotorCCW(0, lower_target);
            }
        }

        /// <summary>
        /// KINEMATICS DEFINITIONS
        /// </summary>
        public void DefineGeometry(double ll, double lu, double le)
        {
            /// ll : lower link length
            /// lu : upper link length
            /// le : distance between end effector and parallel links intersection
            this.ll = ll;
            this.lu = lu;
            this.le = le;
        }

        public double[] ForwardKinematics(double upper_link_angle, double lower_link_angle)
        {
            double l_u = lower_link_angle - upper_link_angle;
            double u = upper_link_angle;
            double l = lower_link_angle;
            double f_l_u = 180 - l_u;
            //int ll = this.ll    
            //int lu = this.lu;
            //int le = this.le;
            double ln = ll + le;
            double alpha = (180 - f_l_u) / 2;

            //int m = (int)Math.Sqrt((lu * lu) + (ll * ll) - (2 * lu * ll * Math.Cos(f_l_u)));
            //int k = (int)Math.Sqrt((m * m) + (le * le) - (2 * m * le * Math.Cos(alpha + f_l_u)));
            //int beta = (int)Math.Asin((le * Math.Sin(alpha + f_l_u)) / k);
            /*double m = (double)Math.Sqrt(Math.Pow(lu, 2) + Math.Pow(ll, 2) - 2 * lu * ll * Math.Cos(DegreeToRadian(f_l_u)));
            double k = (double)Math.Sqrt(Math.Pow(m, 2) + Math.Pow(le, 2) - 2 * m * le * Math.Cos(DegreeToRadian(alpha + f_l_u)));
            double beta = RadianToDegree((double)Math.Asin((le * Math.Sin(DegreeToRadian(alpha + f_l_u))) / k));

            e2 = (double)(k * Math.Cos(DegreeToRadian(beta + alpha + u)));
            e1 = (double)(k * Math.Sin(DegreeToRadian(beta + alpha + u))); */

            e2 = (double)((lu * Math.Cos(DegreeToRadian(u))) + (ln * Math.Cos(DegreeToRadian(l))));
            e1 = (double)((lu * Math.Sin(DegreeToRadian(u))) + (ln * Math.Sin(DegreeToRadian(l))));

            return new double[] { e2, e1 };
        }

        // Helper functions to convert between degrees and radians //
        double DegreeToRadian(double angle)
        {
            return (Math.PI * angle / 180.0);
        }
        double RadianToDegree(double angle)
        {
            return (angle * (180.0 / Math.PI));
        }

        public void SetOrigin(double offset_angle)
        {
            this.offset_angle = offset_angle;
            int[] coordinates = ComputeXY();
            xi = coordinates[1];
            yi = coordinates[0];
            this.x = coordinates[1] - xi;
            this.y = coordinates[0] - yi;
        }

        public double[] ComputePulsrCoordinates()
        {
            UpdateMotorAngles();
            ForwardKinematics(upper.angle, lower.angle);
            return new double[] { e2, e1 };
        }

        public int[] ComputeXY()
        {
            ComputePulsrCoordinates();
            //int y = (int)(-(e2 * Math.Cos(DegreeToRadian(offset_angle))) - (e1 * Math.Sin(DegreeToRadian(offset_angle))));
            //int x = (int)((e1 * Math.Cos(DegreeToRadian(offset_angle))) - (e2 * Math.Sin(DegreeToRadian(offset_angle))));
            //y = glass_to_screen_scaler * y;
            //x = glass_to_screen_scaler * x;

             //x = (int)e1;
             //y = (int)e2;

            y = (int)(-(e2 * Math.Cos(DegreeToRadian(20))) - (e1 * Math.Sin(DegreeToRadian(20))));
            x = (int)((e1 * Math.Cos(DegreeToRadian(20))) - (e2 * Math.Sin(DegreeToRadian(20))));

            y = y * 25;   //scaler can go in here, fixed of 25
            x = x * 25;   //scaler can go in here, fixed of 25

            return new int[] { y, x };
        }

        public int[] ReturnXYCoordinate()
        {
            int[] coordinates = ComputeXY();
            //x = coordinates[1] - xi; 
            //y = coordinates[0] - yi;
            //return new int[] { y, x };

            x = coordinates[1] - xi; //minus gui offset 
            y = coordinates[0] - yi; //add gui offset
            return new int[] { y, x };
        }

        /*public static void Main(string[] args)
        {
            pulsr neuro = new pulsr();
            neuro.InitializeCommunication();    `
            Thread.Sleep(3000);
            Console.WriteLine("communication handshake successful");
            neuro.lower.angle = 109;
            neuro.upper.angle = 0;
            neuro.DefineGeometry(26, 26, 26);
            neuro.SetOrigin(20);
            neuro.upper.target_speed = 0;
            neuro.lower.target_speed = 0;
            Console.WriteLine(neuro.ReturnXYCoordinate());
            while (true)
            {
                neuro.UpdateSensorData();
                Thread.Sleep(250);
            }
        }*/
    }
}
