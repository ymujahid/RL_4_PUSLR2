using PULSR_3;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Timers;
using System.Windows.Forms;


namespace PULSR_3
{
    public partial class Form1 : Form
    {
        /// <summary>
        /// INITIALIZE PULSR
        /// </summary>

        pulsr pulsr3 = new pulsr();

        int selectedMode; //Store the Selected mode
        bool running;   // ///////testing/// store state for assistive mode 

        //int upper_force_t { get; set; }
        //int lower_force_t { get; set; }
        //int[] upper_force_t;
        List<int> upper_force_t;
        List<int> lower_force_t;
        int threshold { get; set; }
        double threshold_upper, threshold_lower;

        //public int value { get; set; }
        public int value;

        public int old_y;
        public int old_x;
        public int new_y;
        private int current_x;
        private int current_y;
        public int new_x;
        public int yOffset = 38;
        public int xOffset = 195;

        int levelStart = 0;   // This should be threshold // see effect and remove later
        int cycle = 0;
        int score = 0;
        int distance;
        //
        //private float angle = 0;
        //private const float orbitRadius = 100;
        //private const float centerOffset = 150;
        //
        private int centerX;
        private int centerY;
        private int largeCircleRadius = 250;
        private int smallCircleRadius = 10;
        private double angle = 270;
        private double angularSpeed = 1; //can be changed, initiated to 1
        private int cycleCount = 0;

        public string fileName ;
        public int cyclename;


        List<Point> trailPoints = new List<Point>();


        public int smallCircleX { get; private set; }
        public int smallCircleY { get; private set; }
        public int DEP { get; private set; }
        public int DEM { get; private set; }

        public Form1()
        {
            // COMMUNICATION INITIALIZATION AND DEFINITION //
            pulsr3.InitializeCommunication();
            //Thread.Sleep(500);

            // COMMUNICATION CHECK //
            if (pulsr3.CheckCommunication())
            {
                Console.WriteLine("Communication Port is Active"); /// change to dialog

                InitializeComponent();
            }
            else
            {
                //Console.WriteLine("Communication Port Not Active");
                //Close();

                Console.WriteLine("Communication check failed for the following components:");
                if (!pulsr3.load_cell_coms.IsOpen)
                {
                    Console.WriteLine(" - Load cell board not found");
                }
                if (!pulsr3.encoder_coms.IsOpen)
                {
                    Console.WriteLine(" - Encoder board not found");
                }
                if (!pulsr3.pulsr2_coms.IsOpen)
                {
                    Console.WriteLine(" - Pulsr communication check failed, restart system to reinitialize communication");
                }
                Close();
            }

            //// Game level selection ////
            //public int value { get; set;}   /////key metric
            if (InputBox("GAME LEVEL", "Select a Level!", ref value) == DialogResult.OK)
            {
                if (value == 0)
                {
                    MessageBox.Show("Passive Mode");
                    selectedMode = 0;
                }
                else if (value == 4)
                {
                    MessageBox.Show("Assistive Mode");
                    selectedMode = 4;
                }
                else if (value == 8)
                {
                    MessageBox.Show("Active Mode");
                    selectedMode = 8;
                }
                else
                {
                    MessageBox.Show("Please Restart The Program and Select a Mode !!!");

                    //Application.Exit();  // This will close the application
                    //Close();

                }
            }

            // RESET ANGLE PARAMETERS AND DISABLE MOTORS //
            pulsr3.UpdateMotorSpeed(0, 0);
            pulsr3.ResetAngles();

            // PULSR KINEMATICS PARAMETERS DEFINITIION AND INITIALIZATION //
            pulsr3.DefineGeometry(26, 26, 26);
            pulsr3.SetOrigin(20);

            //int[] new_yx = pulsr3.ReturnXYCoordinate();
            //old_y = new_yx[0];
            //old_x = new_yx[1];



            //InitializeComponent();
            //panel12.Paint += panel12_Paint;
            //timer.Tick += timer_Tick;
            //timer.Start();
        }


        /// <summary>
        /// MAIN GUI
        /// </summary>

        // GUI INITIALIZATION //
        private void Form1_Load(object sender, EventArgs e)
        {
            //timer.Start();    //commentend not to start when the form loads, the start button controls it now

            //this.KeyPreview = true;       // Enable keyevents for the move
            //this.KeyDown += keyisdown;   // Register the KeyDown event handler

            /// Set initial parameters based on selectedMode ///
            if (selectedMode == 0) // Set parameters for Passive Mode
            {
                // Set parameters for Passive Mode

                pulsr3.UserMode();



                Console.WriteLine("Now in Passive");


                pulsr3.UpdateMotorSpeed(0, 0);
                // reset motion function
                pulsr3.ResetAngles();
                pulsr3.UpdateMotorAngles();
            }
            else if (selectedMode == 4)// Set parameters for Assistive Mode
            {
                // Set parameters for Assistive Mode

                //int[] upper_force_t = Enumerable.Repeat(32000, 70).ToArray();
                //int[] lower_force_t = Enumerable.Repeat(32000, 70).ToArray();
                upper_force_t = new List<int>(Enumerable.Repeat(32000, 70)); //70 samples of load cell data used in plotting graph
                lower_force_t = new List<int>(Enumerable.Repeat(32000, 70)); //70 samples of load cell data used in plotting graph

                threshold = 0;  //real value in start click


                pulsr3.ComputerMode();
                pulsr3.KeyMode();
                Console.WriteLine("Now in Assistive");


                pulsr3.UpdateMotorSpeed(0, 0);
                // reset motion function
                pulsr3.ResetAngles();
                pulsr3.UpdateMotorAngles();
            }
            else if (selectedMode == 8)// Set parameters for Active Mode
            {
                // Set parameters for Active Mode


                pulsr3.ComputerMode();
                pulsr3.KeyMode();
                Console.WriteLine("Now in Active");


                pulsr3.UpdateMotorSpeed(0, 0);
                // reset motion function
                pulsr3.ResetAngles();
                pulsr3.UpdateMotorAngles();
            }

        }

        /////// Timer //////////////
        private void timer_Tick(object sender, EventArgs e)
        {
            angle -= angularSpeed;
            if (angle == -90)
            {
                timer.Stop();
                cycleCount += 1;
                button1.Text = cycleCount.ToString();
                //button1.ForeColor = Color.White;
                //angle = 0;
                if (selectedMode == 4) 
                {
                    running = false; ///////testing///

                    pulsr3.UpdateMotorSpeed(0, 0); // Stop motor movement 
                }
                //file.Close();  //Close the file after each cycle

            }
            orbitPanel.Invalidate(); // Trigger a redraw of the panel

            //// Calculate and display the coordinates
            //float centerX = panel12.Width / 2;
            //float centerY = panel12.Height / 2;
            //float orbitingX = centerX + (float)(orbitRadius * Math.Cos(angle)) - centerOffset;
            //float orbitingY = centerY + (float)(orbitRadius * Math.Sin(angle));

            //string coordinates = $"Orbiting Circle: ({orbitingX}, {orbitingY})";
            //Console.WriteLine(coordinates); // Print to the command line

        }

        ////  Close Maximize Minimize  ////
        private void closeButton(object sender, MouseEventArgs e)
        {
            DialogResult dr = MessageBox.Show("Do you want to close this window", "Confirm", MessageBoxButtons.YesNoCancel, MessageBoxIcon.Warning);
            if (dr == DialogResult.Yes) { Close(); }
            else if (dr == DialogResult.No) { }
            else { }
        }
        private void maximizeButton(object sender, MouseEventArgs e)
        {
            this.WindowState = FormWindowState.Maximized;
            int width = this.Width;
            int height = this.Height;
            Console.WriteLine("MAXIMIZED Width: " + width + ", Height: " + height);
        }
        private void minimizeButton(object sender, MouseEventArgs e)
        {
            this.WindowState = FormWindowState.Normal;
            int width = this.Width;
            int height = this.Height;
            Console.WriteLine("NORMAL Width: " + width + ", Height: " + height);
        }

        //////// START Button //////
        private void startMouseEnter(object sender, EventArgs e)
        {
            panel8.BackgroundImage = global::PULSR_3.Properties.Resources.hoverButton;
        }

        private void startMouseLeave(object sender, EventArgs e)
        {
            panel8.BackgroundImage = global::PULSR_3.Properties.Resources.IdleButton;
        }

        public void startClick(object sender, MouseEventArgs e)
        {
            timer.Stop(); // Stop the timer if it is already running

            angle = 270;
            score = 0;

            //cycleCount = 0;
            //cycleCount += 1;
            //textBox1.Text = cycleCount.ToString();

            //timer.Start(); // Start the timer to begin the animation  // Take it to the bottom and see the effect

            
            if (selectedMode == 4)
            {
                threshold = 1000;
                running = true; /////////////testing///

                button2.Text = Convert.ToString(threshold);
            }


            // Initiate the filename to log parameters //
            cyclename = cycleCount + 1;
            fileName = "sessions_files/" + (cyclename) + ".csv";   ///Create the file for that session in session folder

            if (File.Exists(fileName))   // Check if it existed before
            {
                File.Delete(fileName);
            }

            using (StreamWriter file = new StreamWriter(fileName, true))
            {
                file.WriteLine("THRESHOLD, SCORE, UPPER ANGLE, UPPER THRESHOLD, UPPER_LOAD_CELL, LOWER ANGLE, LOWER THRESHOLD, LOWER_LOAD_CELL, DEP, DEM, TIMESTAMP, SAMPLE_NO");
                //Do not close the file yet, close after one cycle in ...//

                file.Close();
            }

            timer.Start();
        }

        ////////// RESET Button///
        private void resetMouseEnter(object sender, EventArgs e)
        {
            panel9.BackgroundImage = global::PULSR_3.Properties.Resources.hoverButton;
        }

        private void resetMouseLeave(object sender, EventArgs e)
        {
            panel9.BackgroundImage = global::PULSR_3.Properties.Resources.IdleButton;
        }

        private void resetClick(object sender, MouseEventArgs e)
        {
            button1.ResetText();  // Clear cycle textbox
            angle = 270;
            cycleCount = 0;

            score = 0;
            button3.ResetText();  // Clear score textbox
            running = false; ///////testing///
            timer.Stop();
        }

        ////// INCREASE Button ////
        private void increaseMouseEnter(object sender, EventArgs e)
        {
            panel10.BackgroundImage = global::PULSR_3.Properties.Resources.hoverButton;
        }

        private void increaseMouseLeave(object sender, EventArgs e)
        {
            panel10.BackgroundImage = global::PULSR_3.Properties.Resources.IdleButton;
        }

        private void increaseClick(object sender, MouseEventArgs e)
        {
            //// For threshold increment 
            ///
            //button2.Text = Convert.ToString(levelStart += 1000);

            threshold += 1000;
            button2.Text = Convert.ToString(threshold); //test

        }

        ////// DECREASE Button /////
        private void decreaseMouseEnter(object sender, EventArgs e)
        {
            panel11.BackgroundImage = global::PULSR_3.Properties.Resources.hoverButton;
        }
        private void decreaseMouseLeave(object sender, EventArgs e)
        {
            panel11.BackgroundImage = global::PULSR_3.Properties.Resources.IdleButton;
        }
        private void decreaseClick(object sender, MouseEventArgs e)
        {
            if (/*levelStart > 1000*/ threshold > 1000)
            {

                //button2.Text = Convert.ToString(levelStart -= 1000);

                threshold -= 1000;
                button2.Text = Convert.ToString(threshold);  //test

            }
        }
        ///////// Orbiting Circle /////////////
        public void orbitPanelPaint(object sender, PaintEventArgs e)
        {
            if (running == true) // For assistive mode
            {
                pulsr3.UpdateUpperLoadCell(); //logic
                pulsr3.UpdateLowerLoadCell(); // logic

                upper_force_t.RemoveAt(0); // logic
                upper_force_t.Add(pulsr3.upper.link_force); // logic

                lower_force_t.RemoveAt(0); // logic
                lower_force_t.Add(pulsr3.lower.link_force); // logic
                                                            /////////

                /////////
                ///
                if (selectedMode == 4)
                {
                    threshold_upper = (threshold / 2) + ((threshold * Math.Cos(Math.PI * pulsr3.upper.angle / 180)) / 2);
                    threshold_lower = (threshold / 2) + ((threshold * Math.Cos(Math.PI * (pulsr3.lower.angle - 110) / 180)) / 2);
                }
                else
                {
                    threshold_upper = threshold;
                    threshold_lower = threshold;
                }

                double ls, us;
                double dest = 10;

                if (pulsr3.lower.link_force < 30000 - threshold_lower)
                {
                    ls = dest;
                }
                else if (pulsr3.lower.link_force > 34000 + threshold_lower)
                {
                    ls = -dest;
                }
                else
                {
                    ls = 0;
                }

                if (pulsr3.upper.link_force < 30000 - threshold_upper)
                {
                    us = dest;
                }
                else if (pulsr3.upper.link_force > 34000 + threshold_upper)
                {
                    us = -dest;
                }
                else
                {
                    us = 0;
                }

                //capping the speed
                if (us < 0)
                {
                    if (Math.Abs(us) > dest)
                    {
                        us = -dest;
                    }
                }
                else
                {
                    if (Math.Abs(us) < dest)
                    {
                        us = dest;
                    }
                }

                if (ls < 0)
                {
                    if (Math.Abs(ls) > dest)
                    {
                        ls = -dest;
                    }
                }
                else
                {
                    if (Math.Abs(ls) < dest)
                    {
                        ls = dest;
                    }
                }

                // Update speed instruction
                pulsr3.UpdateMotorSpeed((int)us, (int)ls);
                //Thread.Sleep(100);

                //////////////////
                //Pen largeCirclePen = new Pen(Color.FromArgb(0xB0, 0x80, 0x2E), 5.0f);  //moved to buttom
                float centerX = orbitPanel.Width / 2;
                float centerY = orbitPanel.Height / 2;
                //float orbitingX = centerX + (float)(orbitRadius * Math.Cos(angle)) - centerOffset;
                //float orbitingY = centerY + (float)(orbitRadius * Math.Sin(angle));

                /// Draw the larger circle
                float largeCircleX = centerX - largeCircleRadius;
                float largeCircleY = centerY - largeCircleRadius;
                float largeCircleDiameter = 2 * largeCircleRadius;

                //e.Graphics.DrawEllipse(largeCirclePen, largeCircleX, largeCircleY, largeCircleDiameter, largeCircleDiameter); //moved to buttom


                /// Draw Small Obiting Circle
                double radians = angle * Math.PI / 180;
                smallCircleX = (int)(centerX + largeCircleRadius * Math.Cos(radians));
                smallCircleY = (int)(centerY + largeCircleRadius * Math.Sin(radians));

                int smallCircleDiameter = 2 * smallCircleRadius;

                int smallCircleXPos = smallCircleX - smallCircleRadius;
                int smallCircleYPos = smallCircleY - smallCircleRadius;

                //e.Graphics.FillEllipse(Brushes.Green, smallCircleXPos, smallCircleYPos, smallCircleDiameter, smallCircleDiameter); //moved to buttom

                /// Display the coordinates in the terminal ///
                Console.WriteLine("Small Orbiting Circle Coordinates: X = {0}, Y = {1}", smallCircleXPos, smallCircleYPos);

                ///// Draw the small rectangle connected to end effector  /////
                float rectWidth = 20;
                float rectHeight = 20;
                float rectX = centerX - rectWidth / 2;
                //float rectY = centerY - orbitingCircleRadius - rectHeight;
                //e.Graphics.FillRectangle(Brushes.Blue, rectX, rectY, rectWidth, rectHeight);
                //e.Graphics.FillRectangle(Brushes.Blue, 57, 345, rectWidth, rectHeight);

                /// update effector coordinate to give new effector coordinates ///
                //old_x = new_x;
                //old_y = new_y;

                pulsr3.ReturnXYCoordinate();
                //pulsr3.ComputeXY();

                new_x = (xOffset - pulsr3.x);
                new_y = (pulsr3.y - yOffset);

                //new_x = pulsr3.x; // + 306;
                //new_y = pulsr3.y; // + 95;

                current_x = (int)(centerX - new_x);
                current_y = (int)(centerY + new_y);

                // Add the current position to the trail points
                trailPoints.Add(new Point(current_y, current_x));


                //e.Graphics.FillRectangle(Brushes.Blue, current_y, current_x, rectWidth, rectHeight); // moved to buttom

                //Console.WriteLine("End effector : X = {0}, Y = {1}", new_x, new_y);
                Console.WriteLine("End effector Coordinates: X = {0}, Y = {1}", current_y, current_x);


                /// Scoring Calculations ///
                int xDiff = current_y - smallCircleXPos;
                int yDiff = current_x - smallCircleYPos;

                distance = (int)Math.Sqrt((xDiff * xDiff) + (yDiff * yDiff));
                Console.WriteLine("Distance between .... :" + distance);



                Brush ellipseBrush = Brushes.Green;  //default colour
                SolidBrush rectangleBrush = (SolidBrush)Brushes.Blue; //default colour

                Pen largeCirclePen = new Pen(Color.FromArgb(0xB0, 0x80, 0x2E), 5.0f);

                if (distance <= 100)
                {
                    if (distance <= 70)
                    {
                        //change to green green
                        ellipseBrush = Brushes.Green;
                        rectangleBrush = (SolidBrush)Brushes.Green;
                        largeCirclePen = new Pen(Color.Green, 5.0f);


                        //update score
                        button3.Text = Convert.ToString(score += 1);
                    }
                    else
                    {
                        // change to orange orange
                        ellipseBrush = Brushes.Orange;
                        rectangleBrush = (SolidBrush)Brushes.Orange;
                    }
                }
                else
                {
                    // change to green purple
                    ellipseBrush = Brushes.Purple;
                    rectangleBrush = (SolidBrush)Brushes.Green;
                    largeCirclePen = new Pen(Color.Purple, 5.0f);
                }

                e.Graphics.DrawEllipse(largeCirclePen, largeCircleX, largeCircleY, largeCircleDiameter, largeCircleDiameter);
                e.Graphics.FillEllipse(ellipseBrush, smallCircleXPos, smallCircleYPos, smallCircleDiameter, smallCircleDiameter);

                // Draw the trail
                Pen trailPen = new Pen(Color.White, 2); // Change the color and thickness as needed
                trailPen.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash; // Set the DashStyle to Dash
                for (int i = 0; i < trailPoints.Count - 1; i++)
                {
                    e.Graphics.DrawLine(trailPen, trailPoints[i], trailPoints[i + 1]);
                }

                e.Graphics.FillRectangle(rectangleBrush, current_y, current_x, rectWidth, rectHeight);



                //TO force exit assistive //
                /*if (Console.KeyAvailable)
                {
                    var key = Console.ReadKey(true).Key;
                    if (key == ConsoleKey.X)
                    {
                        Console.WriteLine("You pressed 'x'. Exiting the loop.");
                        pulsr3.UpdateMotorSpeed(0, 0);
                        //break;
                        
                    }
                }*/


                /// Loggging parameter into CSV file ///
                parameterLogging();

                fileName = "sessions_files/" + (cyclename) + ".csv";
                using (StreamWriter file = new StreamWriter(fileName, true))
                {
                    file.WriteLine(threshold + "," + score + "," + pulsr3.upper.angle + "," + threshold_upper + "," + pulsr3.upper.link_force + "," + pulsr3.lower.angle + "," + threshold_lower + "," + pulsr3.lower.link_force + "," + DEP + "," + DEM + "," + DateTime.Now.ToString() + "," + cycleCount);
                    file.Close();

                    //file.WriteLine($"{threshold},{score},{pulsr2.upper.angle},{threshold_upper},{pulsr2.upper.link_force},{pulsr2.lower.angle},{threshold_lower},{pulsr2.lower.link_force},{DEP},{DEM},{DateTime.Now.Ticks},{n}");
                }
                Console.WriteLine($"{pulsr3.upper.angle} , {pulsr3.lower.angle} , {pulsr3.upper.link_force} , {pulsr3.lower.link_force}");
            }
            else if ( selectedMode != 4 && selectedMode == 0 || selectedMode == 8) // For active and passive
            {
                pulsr3.UpdateUpperLoadCell(); //logic
                pulsr3.UpdateLowerLoadCell(); //logic

                //Pen largeCirclePen = new Pen(Color.FromArgb(0xB0, 0x80, 0x2E), 5.0f);  //moved to buttom
                float centerX = orbitPanel.Width / 2;
                float centerY = orbitPanel.Height / 2;
                //float orbitingX = centerX + (float)(orbitRadius * Math.Cos(angle)) - centerOffset;
                //float orbitingY = centerY + (float)(orbitRadius * Math.Sin(angle));

                /// Draw the larger circle
                float largeCircleX = centerX - largeCircleRadius;
                float largeCircleY = centerY - largeCircleRadius;
                float largeCircleDiameter = 2 * largeCircleRadius;

                //e.Graphics.DrawEllipse(largeCirclePen, largeCircleX, largeCircleY, largeCircleDiameter, largeCircleDiameter);  //moved to buttom


                /// Draw Small Obiting Circle
                double radians = angle * Math.PI / 180;
                smallCircleX = (int)(centerX + largeCircleRadius * Math.Cos(radians));
                smallCircleY = (int)(centerY + largeCircleRadius * Math.Sin(radians));

                int smallCircleDiameter = 2 * smallCircleRadius;

                int smallCircleXPos = smallCircleX - smallCircleRadius;
                int smallCircleYPos = smallCircleY - smallCircleRadius;

                //e.Graphics.FillEllipse(Brushes.Green, smallCircleXPos, smallCircleYPos, smallCircleDiameter, smallCircleDiameter); //moved to buttom

                /// Display the coordinates in the terminal ///
                Console.WriteLine("Small Orbiting Circle Coordinates: X = {0}, Y = {1}", smallCircleXPos, smallCircleYPos);

                ///// Draw the small rectangle connected to end effector  /////
                float rectWidth = 20;
                float rectHeight = 20;
                float rectX = centerX - rectWidth / 2;
                //float rectY = centerY - orbitingCircleRadius - rectHeight;
                //e.Graphics.FillRectangle(Brushes.Blue, rectX, rectY, rectWidth, rectHeight);
                //e.Graphics.FillRectangle(Brushes.Blue, 57, 345, rectWidth, rectHeight);

                /// update effector coordinate to give new effector coordinates ///
                //old_x = new_x;
                //old_y = new_y;

                pulsr3.ReturnXYCoordinate();
                //pulsr3.ComputeXY();

                new_x = (xOffset - pulsr3.x);
                new_y = (pulsr3.y - yOffset);

                //new_x = pulsr3.x; // + 306;
                //new_y = pulsr3.y; // + 95;

                current_x = (int)(centerX - new_x);
                current_y = (int)(centerY + new_y);

                // Add the current position to the trail points
                trailPoints.Add(new Point(current_y, current_x));


                //e.Graphics.FillRectangle(Brushes.Blue, current_y, current_x, rectWidth, rectHeight); // moved to buttom

                //Console.WriteLine("End effector : X = {0}, Y = {1}", new_x, new_y);
                Console.WriteLine("End effector Coordinates: X = {0}, Y = {1}", current_y, current_x);


                /// Scoring Calculations ///
                int xDiff = current_y - smallCircleXPos;
                int yDiff = current_x - smallCircleYPos;

                distance = (int)Math.Sqrt((xDiff * xDiff) + (yDiff * yDiff));
                Console.WriteLine("Distance between .... :" + distance);



                Brush ellipseBrush = Brushes.Green;  //default colour
                SolidBrush rectangleBrush = (SolidBrush)Brushes.Blue; //default colour

                Pen largeCirclePen = new Pen(Color.FromArgb(0xB0, 0x80, 0x2E), 5.0f);

                if (distance <= 100)
                {
                    if (distance <= 70)
                    {
                        //change to green green
                        ellipseBrush = Brushes.Green;
                        rectangleBrush = (SolidBrush)Brushes.Green;
                        largeCirclePen = new Pen(Color.Green, 5.0f);


                        //update score
                        button3.Text = Convert.ToString(score += 1);
                    }
                    else
                    {
                        // change to orange orange
                        ellipseBrush = Brushes.Orange;
                        rectangleBrush = (SolidBrush)Brushes.Orange;
                        
                    }
                }
                else
                {
                    // change to green purple
                    ellipseBrush = Brushes.Purple;
                    rectangleBrush = (SolidBrush)Brushes.Green;
                    largeCirclePen = new Pen(Color.Purple, 5.0f);
                }

                e.Graphics.DrawEllipse(largeCirclePen, largeCircleX, largeCircleY, largeCircleDiameter, largeCircleDiameter);
                e.Graphics.FillEllipse(ellipseBrush, smallCircleXPos, smallCircleYPos, smallCircleDiameter, smallCircleDiameter);

                // Draw the trail
                Pen trailPen = new Pen(Color.White, 1); // Change the color and thickness as needed
                trailPen.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash; // Set the DashStyle to Dash
                for (int i = 0; i < trailPoints.Count - 1; i++)
                {
                    e.Graphics.DrawLine(trailPen, trailPoints[i], trailPoints[i + 1]);
                }

                e.Graphics.FillRectangle(rectangleBrush, current_y, current_x, rectWidth, rectHeight);

                /////////
                ///
                if (selectedMode == 4)
                {
                    threshold_upper = (threshold / 2) + ((threshold * Math.Cos(Math.PI * pulsr3.upper.angle / 180)) / 2);
                    threshold_lower = (threshold / 2) + ((threshold * Math.Cos(Math.PI * (pulsr3.lower.angle - 110) / 180)) / 2);
                }
                else
                {
                    threshold_upper = threshold;
                    threshold_lower = threshold;
                }


                /// Loggging parameter into CSV file ///
                parameterLogging();

                fileName = "sessions_files/" + (cyclename) + ".csv";
                using (StreamWriter file = new StreamWriter(fileName, true))
                {
                    file.WriteLine(threshold + "," + score + "," + pulsr3.upper.angle + "," + threshold_upper + "," + pulsr3.upper.link_force + "," + pulsr3.lower.angle + "," + threshold_lower + "," + pulsr3.lower.link_force + "," + DEP + "," + DEM + "," + DateTime.Now.ToString() + "," + cycleCount);
                    file.Close();

                    //file.WriteLine($"{threshold},{score},{pulsr2.upper.angle},{threshold_upper},{pulsr2.upper.link_force},{pulsr2.lower.angle},{threshold_lower},{pulsr2.lower.link_force},{DEP},{DEM},{DateTime.Now.Ticks},{n}");
                }
                Console.WriteLine($"{pulsr3.upper.angle} , {pulsr3.lower.angle} , {pulsr3.upper.link_force} , {pulsr3.lower.link_force}");

            }
        }

            


        /// Dialog ///

        public static DialogResult InputBox(string title, string promptText, ref int value)
        {
            Form form = new Form();
            Label label = new Label();
            TextBox textBox = new TextBox();
            Button buttonOk = new Button();
            Button buttonCancel = new Button();

            form.Text = title;
            label.Text = promptText;

            buttonOk.Text = "OK";
            buttonCancel.Text = "Cancel";
            buttonOk.DialogResult = DialogResult.OK;
            buttonCancel.DialogResult = DialogResult.Cancel;

            label.SetBounds(60, 5, 100, 13);
            textBox.SetBounds(75, 23, 50, 20);
            buttonOk.SetBounds(25, 50, 70, 35);
            buttonCancel.SetBounds(100, 50, 70, 35);

            label.AutoSize = true;
            form.ClientSize = new Size(200, 100);
            form.FormBorderStyle = FormBorderStyle.FixedDialog;
            form.StartPosition = FormStartPosition.CenterScreen;
            form.MinimizeBox = false;
            form.MaximizeBox = false;

            form.Controls.AddRange(new Control[] { label, textBox, buttonOk, buttonCancel });
            form.AcceptButton = buttonOk;
            form.CancelButton = buttonCancel;

            DialogResult dialogResult = form.ShowDialog();

            value = Convert.ToInt32(textBox.Text);
            return dialogResult;



        }


        /// 
        void parameterLogging()
        {
            // compute DEP, TEP, DEM, TEM

            double TEP = (Math.Atan2((centerX - current_x), (current_y - centerY)) * (180 / Math.PI));
            double closest_x = largeCircleRadius * Math.Sin((TEP * (Math.PI / 180)));
            double closest_y = largeCircleRadius * Math.Cos((TEP * (Math.PI / 180)));
            double x = new_x ;
            double y = new_y ;
            TEP = TEP >= 0 ? TEP : 360 + TEP;
            double dx = (closest_x - x);
            double dy = (closest_y - y);
             DEP = ((int)Math.Sqrt(Math.Pow(dx, 2) + Math.Pow(dy, 2)));
            dx = (smallCircleX - current_x) ;
            dy = (smallCircleY - current_y);
             DEM = (int)Math.Sqrt(Math.Pow(dy, 2) + Math.Pow(dx, 2));

            

            /*using (StreamWriter file = new StreamWriter(fileName, true))
            {
                file.WriteLine( 1 + "," + score + "," + pulsr3.upper.angle + "," + 2 + "," + pulsr3.upper.link_force + "," + pulsr3.lower.angle + "," + 3 + "," + pulsr3.lower.link_force + "," + DEP + "," + DEM + "," + DateTime.Now.ToString() + "," + 4);
                file.Close();

                //file.WriteLine($"{threshold},{score},{pulsr2.upper.angle},{threshold_upper},{pulsr2.upper.link_force},{pulsr2.lower.angle},{threshold_lower},{pulsr2.lower.link_force},{DEP},{DEM},{DateTime.Now.Ticks},{n}");
            }*/

        }
    }
}

