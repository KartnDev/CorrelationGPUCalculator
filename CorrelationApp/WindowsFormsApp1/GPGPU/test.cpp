std::ifstream f;
           /* if argument given */
    f.open ("ClosedEyes.asc");

    std::string line, val;                  /* string for line & value */
    std::vector<std::vector<float>> array;    /* vector of vector<int>  */

    while (std::getline (f, line)) {       /* read each line */
        std::vector<float> v;                 /* row vector v */
        std::stringstream s (line);         /* stringstream line */
        while (getline (s, val, ' '))       /* get each value (',' delimited) */
            v.push_back (std::stof (val));  /* add to row vector */
        array.push_back (v);                /* add row vector to array */
    }

    unsigned int n = array.size();
    int signal_count = array[0].size();
        

    float** h_x = (float**)malloc(signal_count*sizeof(float*));
    
    for(int i = 0; i < signal_count; i++)
    {
        h_x[i] = (float*)malloc(n * sizeof(float));
    }

    for(int i = 0; i < signal_count; i++)
    {
        for(int j = 0; j < n; j++)
        {
            h_x[i][j] = array[i][j];
            std::cout << array[i][j] << " ";
        }
        std::cout << "\n";
    }

    float * result = gpgpu_correlation_mat(h_x, n, signal_count);

    for(int i = 0; i < signal_count; i++)
    {
        for(int j = 0; j < signal_count; j++)
        {   
            printf("%.2f\t|\t", result[i * signal_count + j]);
        }
        std::cout << "\n";
    }
    
    
    system("pause");