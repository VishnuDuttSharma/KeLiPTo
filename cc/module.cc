#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include "H5Cpp.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
    
using std::cout;
using std::endl;

using namespace H5;
using namespace rapidjson;
using Eigen::tanh;
using Eigen::MatrixXd;

inline double min(double a, double b) { return(((a)<(b))?(a):(b));}
inline double max(double a, double b) { return(((a)>(b))?(a):(b));}

double hard_sigmoid(double x){
    return(max(0.0, min(1.0, x*0.2+0.5)));
}

double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

inline MatrixXd Mult(MatrixXd input1, MatrixXd input2){
    return(input1.array() * input2.array());
}

char* read_model(const char *filename)
{
    char *arch_json;
    
    H5File file(filename, H5F_ACC_RDWR);
    Attribute attr(file.openAttribute("model_config"));
    DataType type(attr.getDataType());
    
    
    attr.read(type, &arch_json);
    return arch_json;
}


inline MatrixXd readCSV(const char* filename, int row_n, int col_n){
    
    std::ifstream file(filename);
    int col_flag = 0;

    MatrixXd m(row_n, col_n);
    

    if(row_n == 1){
        col_flag = 1;
        m.resize(col_n, row_n);
    }

    
    std::string line;
    
    int row = 0;
    int col = 0;
    
    
    if(file.is_open()){
        while(std::getline(file, line)){
            char *ptr = (char *)line.c_str();
            int len = line.length();
            
            col = 0;
            
            char *start = ptr;
            for(int i = 0; i < len; i++){
                if(ptr[i] == ','){
                    m(row, col++) = atof(start);
                    start = ptr + i + 1;
                }
            }
            m(row, col) = atof(start);
            
            row++;
        }
        file.close();   
    }
    
    if(col_flag)
        return(m.transpose());

    return(m);
}

class Embedding{
    public:
        int row;
        int col;
        int out_size;
        MatrixXd W;

        Embedding(){

        }

        Embedding(const char* filename, int row, int col){
            this->row = row;
            this->col = col;
            this->out_size = col;

            W.resize(this->row, this->col);
            this->W = readCSV(filename, this->row, this->col);
        
        }

        MatrixXd operator()(MatrixXd input){
            MatrixXd Out(input.cols(), this->out_size);
            for(int t=0; t < Out.rows(); t++)
                Out.row(t) = W.row( input(0, t) );

            return(Out);
        }
};

class LSTM{
    public:
        int inp_size;
        int out_size;
        MatrixXd kernel;
        MatrixXd recurrent_kernel;
        MatrixXd bias;

        LSTM(){

        }

        LSTM(const char* kernel_filename, const char* recurrent_kernel_filename, const char* bias_filename, int inp_size, int out_size){
            this->inp_size = inp_size;
            this->out_size = out_size;

            kernel.resize(this->inp_size, this->out_size * 4);
            kernel = readCSV(kernel_filename, this->inp_size, this->out_size * 4);

            recurrent_kernel.resize(this->out_size, this->out_size * 4);
            recurrent_kernel = readCSV(recurrent_kernel_filename, this->out_size, this->out_size * 4);      
            
            bias.resize(1, this->out_size * 4);
            bias = readCSV(bias_filename, 1, this->out_size * 4);       

        }

        MatrixXd operator()(MatrixXd input){
            int LSTM_OUT = this->out_size;
            int MAXLEN   = input.rows();

            MatrixXd C_t = MatrixXd::Zero(1, this->out_size);
            MatrixXd h_t = MatrixXd::Zero(1, this->out_size);
            
            for(int t = 0; t < MAXLEN; t++){
                MatrixXd tmp_out(1, 4*LSTM_OUT);
                MatrixXd i_t(1, LSTM_OUT);
                MatrixXd f_t(1, LSTM_OUT);
                MatrixXd o_t(1, LSTM_OUT);
                MatrixXd g_t(1, LSTM_OUT);
                
                // IFCO
                tmp_out = input.row(t) * kernel + h_t * recurrent_kernel + bias;
                
                i_t = (tmp_out.block(0, 0*LSTM_OUT, 1, LSTM_OUT)).unaryExpr(&hard_sigmoid);
                f_t = (tmp_out.block(0, 1*LSTM_OUT, 1, LSTM_OUT)).unaryExpr(&hard_sigmoid);
                o_t = (tmp_out.block(0, 3*LSTM_OUT, 1, LSTM_OUT)).unaryExpr(&hard_sigmoid);
                g_t = tanh((tmp_out.block(0, 2*LSTM_OUT, 1, LSTM_OUT)).array());
                
                C_t = f_t.array() * C_t.array() + i_t.array() * g_t.array();
                h_t = o_t.array() * tanh(C_t.array());
            }
            return h_t; 
        }   
};

class GRU{
    public:
        int inp_size;
        int out_size;
        MatrixXd kernel;
        MatrixXd recurrent_kernel;
        MatrixXd bias;

        GRU(){

        }

        GRU(const char* kernel_filename, const char* recurrent_kernel_filename, const char* bias_filename, int inp_size, int out_size){
            this->inp_size = inp_size;
            this->out_size = out_size;

            kernel.resize(this->inp_size, this->out_size * 4);
            kernel = readCSV(kernel_filename, this->inp_size, this->out_size * 4);

            recurrent_kernel.resize(this->out_size, this->out_size * 4);
            recurrent_kernel = readCSV(recurrent_kernel_filename, this->out_size, this->out_size * 4);      
            
            bias.resize(1, this->out_size * 4);
            bias = readCSV(bias_filename, 1, this->out_size * 4);       

        }

        MatrixXd operator()(MatrixXd input){
            int LSTM_OUT = this->out_size;
            int MAXLEN   = input.rows();

            MatrixXd hh_t = MatrixXd::Zero(1,LSTM_OUT);
            MatrixXd h_t = MatrixXd::Zero(1,LSTM_OUT);
            MatrixXd one_arr = MatrixXd::Ones(1,LSTM_OUT);
            
            MatrixXd tmp_out(1, 3*LSTM_OUT);
            MatrixXd tmp_W(1, LSTM_OUT);
            MatrixXd tmp_U(LSTM_OUT, LSTM_OUT);
            MatrixXd z_t(1, LSTM_OUT);
            MatrixXd r_t(1, LSTM_OUT);
            
            tmp_U = recurrent_kernel.block(0, 2*LSTM_OUT, LSTM_OUT, LSTM_OUT);
            
            for(int t = 0; t < MAXLEN; t++){
                
                // IFCO
                
                tmp_W = input.row(t) * kernel + bias ;
                tmp_out = tmp_W + h_t * recurrent_kernel;
                    
                z_t = (tmp_out.block(0, 0*LSTM_OUT, 1, LSTM_OUT)).unaryExpr(&hard_sigmoid);
                r_t = (tmp_out.block(0, 1*LSTM_OUT, 1, LSTM_OUT)).unaryExpr(&hard_sigmoid);
                r_t = r_t.array() * h_t.array();
                
                hh_t = tanh((r_t * tmp_U + tmp_W.block(0, 2*LSTM_OUT, 1, LSTM_OUT)).array() );
                h_t = (one_arr - z_t).array() * hh_t.array() + z_t.array() * h_t.array(); 
            
            }
            return h_t;
        }   
};

class Dense{
    public:
        int inp_size;
        int out_size;
        string activation;
        MatrixXd kernel;
        MatrixXd bias;

        Dense(){

        }

        Dense(const char* kernel_filename, const char* bias_filename, int inp_size, int out_size, string activation="linear"){
            this->inp_size = inp_size;
            this->out_size = out_size;
            this->activation = activation;

            kernel.resize(this->inp_size, this->out_size);
            kernel = readCSV(kernel_filename, this->inp_size, this->out_size);

            bias.resize(1, this->out_size);
            bias = readCSV(bias_filename, 1, this->out_size);
        }

        MatrixXd operator()(MatrixXd input){

            if(this->activation == "linear")
                return( input * kernel + bias );
            else if(this->activation == "sigmoid")
                return( (input * kernel + bias).unaryExpr(&sigmoid) );
        }
};
