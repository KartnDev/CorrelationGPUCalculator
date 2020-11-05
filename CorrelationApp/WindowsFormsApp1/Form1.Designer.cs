namespace WindowsFormsApp1
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.button1 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.computeDevice = new System.Windows.Forms.ComboBox();
            this.shiftBox = new System.Windows.Forms.TextBox();
            this.shiftLabel = new System.Windows.Forms.Label();
            this.batchSizeLabel = new System.Windows.Forms.Label();
            this.BatchSizeBox = new System.Windows.Forms.TextBox();
            this.DeviceListBox = new System.Windows.Forms.TextBox();
            this.PickOutputFolder = new System.Windows.Forms.Button();
            this.dataGridView1 = new System.Windows.Forms.DataGridView();
            this.MainSignal = new System.Windows.Forms.DataGridViewCheckBoxColumn();
            this.AtiveSignals = new System.Windows.Forms.DataGridViewCheckBoxColumn();
            ((System.ComponentModel.ISupportInitialize) (this.dataGridView1)).BeginInit();
            this.SuspendLayout();
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(12, 12);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(111, 37);
            this.button1.TabIndex = 0;
            this.button1.Text = "Загрузить Файл";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.ButtonLoadClick);
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(13, 56);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(110, 39);
            this.button2.TabIndex = 2;
            this.button2.Text = "Пуск";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.CalculateClick);
            // 
            // computeDevice
            // 
            this.computeDevice.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.computeDevice.FormattingEnabled = true;
            this.computeDevice.Items.AddRange(new object[] {"CPU_Parallel", "OpenBLAS", "GPU_CUDA", "CPU_Cpp"});
            this.computeDevice.Location = new System.Drawing.Point(592, 12);
            this.computeDevice.Name = "computeDevice";
            this.computeDevice.Size = new System.Drawing.Size(195, 21);
            this.computeDevice.TabIndex = 3;
            this.computeDevice.SelectedIndexChanged += new System.EventHandler(this.computeDevice_SelectedIndexChanged);
            // 
            // shiftBox
            // 
            this.shiftBox.Location = new System.Drawing.Point(12, 128);
            this.shiftBox.Name = "shiftBox";
            this.shiftBox.Size = new System.Drawing.Size(111, 20);
            this.shiftBox.TabIndex = 4;
            this.shiftBox.Text = "100";
            // 
            // shiftLabel
            // 
            this.shiftLabel.ImageAlign = System.Drawing.ContentAlignment.TopCenter;
            this.shiftLabel.Location = new System.Drawing.Point(13, 102);
            this.shiftLabel.Name = "shiftLabel";
            this.shiftLabel.Size = new System.Drawing.Size(90, 23);
            this.shiftLabel.TabIndex = 5;
            this.shiftLabel.Text = "Шаг сдвига:";
            // 
            // batchSizeLabel
            // 
            this.batchSizeLabel.ImageAlign = System.Drawing.ContentAlignment.TopCenter;
            this.batchSizeLabel.Location = new System.Drawing.Point(12, 162);
            this.batchSizeLabel.Name = "batchSizeLabel";
            this.batchSizeLabel.Size = new System.Drawing.Size(90, 23);
            this.batchSizeLabel.TabIndex = 6;
            this.batchSizeLabel.Text = "Длина окна:";
            // 
            // BatchSizeBox
            // 
            this.BatchSizeBox.Location = new System.Drawing.Point(12, 188);
            this.BatchSizeBox.Name = "BatchSizeBox";
            this.BatchSizeBox.Size = new System.Drawing.Size(111, 20);
            this.BatchSizeBox.TabIndex = 7;
            this.BatchSizeBox.Text = "1000";
            // 
            // DeviceListBox
            // 
            this.DeviceListBox.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.DeviceListBox.Location = new System.Drawing.Point(592, 53);
            this.DeviceListBox.Multiline = true;
            this.DeviceListBox.Name = "DeviceListBox";
            this.DeviceListBox.ReadOnly = true;
            this.DeviceListBox.Size = new System.Drawing.Size(195, 109);
            this.DeviceListBox.TabIndex = 8;
            // 
            // PickOutputFolder
            // 
            this.PickOutputFolder.Location = new System.Drawing.Point(12, 401);
            this.PickOutputFolder.Name = "PickOutputFolder";
            this.PickOutputFolder.Size = new System.Drawing.Size(111, 37);
            this.PickOutputFolder.TabIndex = 9;
            this.PickOutputFolder.Text = "Output Folder\r\n";
            this.PickOutputFolder.UseVisualStyleBackColor = true;
            this.PickOutputFolder.Click += new System.EventHandler(this.PickOutputFolderClick);
            // 
            // dataGridView1
            // 
            this.dataGridView1.AllowUserToOrderColumns = true;
            this.dataGridView1.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dataGridView1.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {this.MainSignal, this.AtiveSignals});
            this.dataGridView1.Location = new System.Drawing.Point(129, 12);
            this.dataGridView1.Name = "dataGridView1";
            this.dataGridView1.Size = new System.Drawing.Size(457, 380);
            this.dataGridView1.TabIndex = 10;
            // 
            // MainSignal
            // 
            this.MainSignal.HeaderText = "Опорый";
            this.MainSignal.Name = "MainSignal";
            this.MainSignal.Resizable = System.Windows.Forms.DataGridViewTriState.True;
            this.MainSignal.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.Automatic;
            // 
            // AtiveSignals
            // 
            this.AtiveSignals.HeaderText = "Активные";
            this.AtiveSignals.Name = "AtiveSignals";
            this.AtiveSignals.Resizable = System.Windows.Forms.DataGridViewTriState.True;
            this.AtiveSignals.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.Automatic;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.dataGridView1);
            this.Controls.Add(this.PickOutputFolder);
            this.Controls.Add(this.DeviceListBox);
            this.Controls.Add(this.BatchSizeBox);
            this.Controls.Add(this.batchSizeLabel);
            this.Controls.Add(this.shiftLabel);
            this.Controls.Add(this.shiftBox);
            this.Controls.Add(this.computeDevice);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.button1);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize) (this.dataGridView1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        private System.Windows.Forms.DataGridViewCheckBoxColumn AtiveSignals;
        private System.Windows.Forms.DataGridViewCheckBoxColumn MainSignal;

        private System.Windows.Forms.DataGridView dataGridView1;
        

        private System.Windows.Forms.Button PickOutputFolder;

        private System.Windows.Forms.TextBox DeviceListBox;

        private System.Windows.Forms.Label batchSizeLabel;

        private System.Windows.Forms.TextBox BatchSizeBox;
        private System.Windows.Forms.ComboBox computeDevice;

        private System.Windows.Forms.TextBox shiftBox;
        private System.Windows.Forms.Label shiftLabel;

        #endregion

        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button2;
    }
}

