#ifndef NET_H
#define NET_H

typedef struct {
    double** neuron;
    double*** weight;       /* weights go backwards  */
    int* npl;               /* neuron per layer      */
    int num_layers;         /* io included           */
    double* d_out;          /* delta of output layer */
    double** d_hid;         /* delta of hidden layer */
    char activation[10];    /* activation for every layer but input */

}network;


double random_weight();
network* net_init(int num_layers, int* nrns_per_layer, char* activation);
float sigmoid(float a);
float sig_der(float a);
void forward(network* net,double input[]);
void net_print(network* net);
double out_err(network* net, double output[]);
void delta_out(network* net, double output[]);
int max_tbl(int tbl[],int len);
void delta_hid(network* net);
int adjust_weights(network* net, double lr);
int train(network* net, double* input, double* output, int length, double lr, double err_tgt);


#endif /*   NET_H   */
