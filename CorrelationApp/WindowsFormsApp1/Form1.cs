﻿using System;using System.Collections.Generic;using System.ComponentModel;using System.Data;using System.Drawing;using System.IO;using System.Linq;using System.Text;using System.Threading.Tasks;using System.Windows.Forms;using WindowsFormsApp1.DeviceComputes;using WindowsFormsApp1.Math.Statistics;using WindowsFormsApp1.Utils;using GASS.CUDA;using MathNet.Numerics.Statistics;using MathNet;using static MathNet.Numerics.Random.RandomExtensions;namespace WindowsFormsApp1{    public partial class Form1 : Form    {        private List<double[]> _fileArrays;        private volatile bool isRedAlready = false;        private string prevFileName;                private IComputeDevice device;        public Form1()        {            InitializeComponent();            computeDevice.SelectedIndex = 0;        }        private void ButtonLoadClick(object sender, EventArgs e)        {            _fileArrays = new List<double[]>();            isRedAlready = false;            using (var ofd = new System.Windows.Forms.OpenFileDialog())            {                if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)                {                    TryInit(ofd.FileName);                    prevFileName = ofd.FileName;                }                else                {                    MessageBox.Show("Не выбран файл!");                }            }        }        private void TryInit(string filePath)        {            Task.Run(() =>            {                var lines = File.ReadLines(filePath).Skip(4);                char fileSeparator = GetCurrentSeparator(lines.First());                InitArrays(lines.Count(), lines.First().Split(fileSeparator).Length);                VolatileRead(lines, fileSeparator);                isRedAlready = true;            });        }        void InitArrays(int rowLength, int rows)        {            for (int i = 0; i < rows; i++)            {                _fileArrays.Add(new double[rowLength]);            }        }        private char GetCurrentSeparator(string line1)        {            foreach (var sep in " \t,")            {                if (line1.Contains(sep) && line1.Split(sep).Length > 1)                {                    return sep;                }            }            throw new FileLoadException("Bad file format!");        }        private void VolatileRead(IEnumerable<string> lines, char separator)        {            for (int i = 0; i < lines.Count(); i++)            {                var splitted = lines.ElementAt(i).Split(separator);                for (int j = 0; j < splitted.Length; j++)                {                    _fileArrays.ElementAt(j)[i] = Double.Parse(splitted[j]);                }            }        }        private void InitComputeDevice()        {            int batchSize, shiftWidth;            if (int.TryParse(BatchSizeBox.Text, out batchSize) && int.TryParse(shiftBox.Text, out shiftWidth))            {                switch (computeDevice.SelectedIndex) // CPU Parallel                 {                   case 0: device = new CpuComputeDevice("OutputFiles", prevFileName); break;                   case 1: device = new OpenBLASCompute("OutputFiles", prevFileName); break;                   case 2: device = new CUDACompute("OutputFiles", prevFileName); break;                }            }        }                private void CalculateClick(object sender, EventArgs e)        {            _fileArrays = new List<double[]>();            for (int i = 0; i < 5; i++)            {                _fileArrays.Add((new Random(i)).NextDoubles(10000));            }                        InitComputeDevice();                                    var watch = System.Diagnostics.Stopwatch.StartNew();            device.ShiftCompute( _fileArrays, int.Parse(BatchSizeBox.Text), int.Parse(shiftBox.Text));            watch.Stop();            var elapsedMs = ((double)watch.ElapsedMilliseconds / 1000);            Console.WriteLine("Time of exection: " + elapsedMs);        }                    }}