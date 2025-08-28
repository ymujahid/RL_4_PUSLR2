using System;
using System.IO.Ports;
using System.Threading;

public class Motor
{
    public int angle;
    public int target_speed;
    public int link_force;
    public int fore_threshold;
}

public class pulsr
{
    private int control_data;
    private string motor_data;
    private int force_data;
    private int wanted_load_cell;
    private bool circleMode;

    private Motor upper;
    private Motor lower;

    private int ll;
    private int lu;
    private int le;
    private int e1;
    private int e2;
    private int xi;
    private int yi;
    private int x;
    private int y;
    private int offset_angle;

    private SerialPort load_cell_coms;
    private SerialPort encoder_coms;
    private SerialPort pulsr2_coms;

    private string load_port;
    private string enc_port;
    private string motor_port;

    private const int baudrate = 2000000;

    public pulsr()
    {
        control_data = 0;
        motor_data = "";
        force_data = 0;
        wanted_load_cell = 0;
        circleMode = false;

        upper = new Motor();
        lower = new Motor();

        ll = 0;
        lu = 0;
        le = 0;
        e1 = 0;
        e2 = 0;
        xi = 0;
        yi = 0;
        x = 0;
        y = 0;
        offset_angle = 0;

        string[] usb_serial_nos = File.ReadAllText("usb_ser.txt").Split();
        load_port = usb_serial_nos[0];
        enc_port = usb_serial_nos[1];
        motor_port = usb_serial_nos[2];

        foreach (var p in SerialPort.GetPortNames())
        {
            var serialNumber = new ManagementObjectSearcher("SELECT * FROM Win32_SerialPort")
                .Get()
                .Cast<ManagementBaseObject>()
                .Where(x => x["DeviceID"].ToString() == p)
                .Select(x => x["PNPDeviceID"].ToString())
                .FirstOrDefault();

            if (serialNumber == load_port)
            {
                load_port = p;
            }
            else if (serialNumber == enc_port)
            {
                enc_port = p;
            }
            else if (serialNumber == motor_port)
            {
                motor_port = p;
            }
        }

        Console.WriteLine(load_port + " " + enc_port + " " + motor_port);
    }

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

    public void CheckCommunication()
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
            force_data = load_cell_coms.ReadByte() << 2 | load_cell_coms.ReadByte() >> 3;
            force_data = force_data << 5 | load_cell_coms.ReadByte();
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
            motor_data = encoder_coms.ReadByte().ToString() + encoder_coms.ReadByte().ToString() + encoder_coms.ReadByte().ToString() + encoder_coms.ReadByte().ToString();
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
            motor_data = encoder_coms.ReadByte().ToString() + encoder_coms.ReadByte().ToString() + encoder_coms.ReadByte().ToString() + encoder_coms.ReadByte().ToString();
        }
    }

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

    public int UpdateUpperLoadCell()
    {
        wanted_load_cell = 1;
        UpdateLinkForce();
        upper.link_force = force_data << 8 | force_data;
        return upper.link_force;
    }

    public int UpdateLowerLoadCell()
    {
        wanted_load_cell = 2;
        UpdateLinkForce();
        lower.link_force = force_data << 8 | force_data;
        return lower.link_force;
    }

    public void UpdateMotorAngles()
    {
        UpdateAngles();
        upper.angle = motor_data[1] << 8 | motor_data[2];
        lower.angle = motor_data[3] << 8 | motor_data[4];
    }

    public void UpdateSensorData()
    {
        UpdateLowerLoadCell();
        UpdateUpperLoadCell();
        UpdateMotorAngles();
        Console.WriteLine("angles: " + upper.angle + ", " + lower.angle);
        Console.WriteLine("force: " + upper.link_force + ", " + lower.link_force);
    }

    public void SetMotorDirections(bool front, bool back, bool left, bool right)
    {
        upper.target_speed = front ? Math.Abs(upper.target_speed) : -Math.Abs(upper.target_speed);
        lower.target_speed = back ? Math.Abs(lower.target_speed) : -Math.Abs(lower.target_speed);
    }

    public void EnableUpperMotor()
    {
        control_data |= 2;
        SendCommand();
    }

    public void DisableUpperMotor()
    {
        upper.target_speed = 0;
        control_data &= 125;
        SendCommand();
    }

    public void EnableLowerMotor()
    {
        control_data |= 8;
        SendCommand();
    }

    public void DisableLowerMotor()
    {
        lower.target_speed = 0;
        control_data &= 119;
        SendCommand();
    }

    public void UpperMotorCW(int en_time, int speed)
    {
        upper.target_speed = Math.Abs(speed);
        control_data |= 2;
        control_data &= 123;
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
        control_data |= 2;
        control_data |= 4;
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
        control_data |= 8;
        control_data &= 111;
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
        control_data |= 8;
        control_data |= 16;
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

    public void DefineGeometry(int ll, int lu, int le)
    {
        this.ll = ll;
        this.lu = lu;
        this.le = le;
    }

    public int[] ForwardKinematics(int upper_link_angle, int lower_link_angle)
    {
        int l_u = lower_link_angle - upper_link_angle;
        int u = upper_link_angle;
        int l = lower_link_angle;
        int f_l_u = 180 - l_u;
        int ll = this.ll;
        int lu = this.lu;
        int le = this.le;
        int ln = this.ll + this.le;
        int alpha = (180 - f_l_u) / 2;
        int m = (int)Math.Sqrt((lu * lu) + (ll * ll) - (2 * lu * ll * Math.Cos(f_l_u)));
        int k = (int)Math.Sqrt((m * m) + (le * le) - (2 * m * le * Math.Cos(alpha + f_l_u)));
        int beta = (int)Math.Asin((le * Math.Sin(alpha + f_l_u)) / k);
        int e1 = (int)(k * Math.Cos(beta + alpha + u));
        int e2 = (int)(k * Math.Sin(beta + alpha + u));
        return new int[] { e2, e1 };
    }

    public void SetOrigin(int offset_angle)
    {
        this.offset_angle = offset_angle;
        int[] coordinates = ComputeXY();
        xi = coordinates[1];
        yi = coordinates[0];
        x = coordinates[1] - xi;
        y = coordinates[0] - yi;
    }

    public int[] ComputePulsrCoordinates()
    {
        UpdateMotorAngles();
        return ComputeXY();
    }

    public int[] ComputeXY()
    {
        int[] coordinates = ComputePulsrCoordinates();
        int y = -(coordinates[1] * Math.Cos(offset_angle)) - (coordinates[0] * Math.Sin(offset_angle));
        int x = (coordinates[0] * Math.Cos(offset_angle)) - (coordinates[1] * Math.Sin(offset_angle));
        y = glass_to_screen_scaler * y;
        x = glass_to_screen_scaler * x;
        return new int[] { y, x };
    }

    public int[] ReturnXYCoordinate()
    {
        int[] coordinates = ComputeXY();
        x = coordinates[1] - xi;
        y = coordinates[0] - yi;
        return new int[] { y, x };
    }

    public static void Main(string[] args)
    {
        pulsr neuro = new pulsr();
        neuro.InitializeCommunication();
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
    }
}