#ifndef net
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define dbg 0
#define log_every 100


typedef struct {
    double** neuron;
    double*** weight;       /* weights go backwards  */
    int* npl;               /* neuron per layer      */
    int num_layers;         /* io included           */
    double* d_out;          /* delta of output layer */
    double** d_hid;         /* delta of hidden layer */
    char activation[10];    /* activation for every layer but input */
    
}network;


double random_weight(){
    return (double)rand()/(double)RAND_MAX-0.5;
}

network* net_init(int num_layers, int* nrns_per_layer, char* activation){
    if(num_layers<1 || nrns_per_layer==NULL || activation==NULL){
        printf("wrong arguments");
        return NULL;
    }

    network* net=(network*)malloc(sizeof(network));

    net->num_layers=num_layers;
    net->npl=(int*)malloc(num_layers*sizeof(int));
    net->neuron=(double**)malloc(num_layers*sizeof(double*));
    net->weight=(double***)malloc(num_layers*sizeof(double**));
    net->d_out=(double*)malloc(net->npl[ net->num_layers - 1 ]*sizeof(double));
    net->d_hid=(double**)malloc(net->num_layers*sizeof(double*));

    /*INIT NEURONS*/
    for (int i=0; i<num_layers;i++){        /*for each layer*/
        net->npl[i]=nrns_per_layer[i];
        net->neuron[i]=(double*)malloc((1+net->npl[i])*sizeof(double));
        net->neuron[i][0]=1;                 /*set bias to 1*/
        for(int j=1;j<=net->npl[i];j++){      /*for each neuron in layer except bias*/
            net->neuron[i][j]=0;    
        }
    }

    /*INIT WEIGHTS*/
    for (int i=1; i<num_layers;i++){        /*for each layer*/
        net->weight[i]=(double**)malloc((1+net->npl[i-1])*sizeof(double*));
        for(int j=1;j<=net->npl[i];j++){      /*for each neuron in layer except bias*/
            net->weight[i][j]=(double*)malloc((1+net->npl[i-1])*sizeof(double));
            for (int k=0;k<=net->npl[i-1];k++)
                net->weight[i][j][k]=random_weight();
        }
    }
    
    /*INIT delta_hidden*/
    for (int i=net->num_layers-2;i>0;i--){
            net->d_hid[i]=(double*) malloc(net->npl[i]*sizeof(double));
    }

    return net;
}

float sigmoid(float a){
    // printf("activation:%f\n",a );
    return 1 / (1 + exp(-a));
}

float sig_der(float a){
    return a*(1-a);
}

void forward(network* net,double input[]){

    /*SETTING FIRST LAYER INPUT*/
    for(int i=1;i<=net->npl[0];i++){
        net->neuron[0][i]=input[i-1];
    }

    /*FORWARD PASS*/
    for (int i=1; i<net->num_layers;i++){        /*for each layer except first*/

        for(int j=1;j<=net->npl[i];j++){      /*for each neuron in layer except bias*/
            net->neuron[i][j]=0;
            for(int k=0;k<=net->npl[i-1];k++){
                net->neuron[i][j]+=net->neuron[i-1][k]*net->weight[i][j][k];
            }
            //SIGMOID GOES HERE
            net->neuron[i][j]=sigmoid(net->neuron[i][j]);
        }
    }

}


void net_print(network* net){
    printf("num_layers=%d\n",net->num_layers);
    printf("\n NEURONS\n");

    for (int i=0; i<net->num_layers;i++){        /*for each layer*/
        printf("npl[%d]=%d\n",i,net->npl[i]);
        for(int j=0;j<=net->npl[i];j++){      /*for each neuron in layer bias included*/
            printf("%f\t",net->neuron[i][j]);    
        }
        printf("\n");
    }

    printf("\nWEIGHTS\n");

    for (int i=1; i<net->num_layers;i++){        /*for each layer except first*/
        printf("LAYER %d->%d\n",i-1,i);
        for(int j=1;j<=net->npl[i];j++){      /*for each neuron in layer except bias*/
            for (int k=0;k<=net->npl[i-1];k++) /*for each neuron in last layer + bais*/
                printf("%f,\t",net->weight[i][j][k]);
            printf("\n");
        }
    }

}


double out_err(network* net, double output[]){
    double err=0;
    int out_dim=net->npl[ net->num_layers - 1 ];
    for (int i=1 ; i<= out_dim ; i++ ){ /*for all neurons in last layer except bias*/
        err += (output[i-1] - net->neuron[ net->num_layers - 1 ][i]) * (output[i-1] - net->neuron[ net->num_layers - 1 ][i]);
    }
    return err;
}

void delta_out(network* net, double output[]){

    int out_dim=net->npl[ net->num_layers - 1 ];
    //double d_out[ out_dim ];

    for (int i=1; i<=out_dim;i++){
        net->d_out[i-1] = ( output[ i-1 ] - net->neuron[net->num_layers - 1 ][i] ) * sig_der(net->neuron[ net->num_layers - 1 ][i]);
    }


}


int max_tbl(int tbl[],int len){
    int max;
    if (len<=0){
        printf("EMPTY table provided");
        return -1;
    }
    max=tbl[0];
    for(int i=1;i<len;i++){
        if( tbl[i]>max )
            max=tbl[i];
    }
    return max;
}

void delta_hid(network* net){
    int i,j,k;

    /*  INIT d_hid*/
    for(i=net->num_layers-2;i>0;i--){
        for(j=1; j<=net->npl[i];j++){   /*no bias*/
            net->d_hid[i][j-1] = 0;
        }                
    }

    /*compute error d_hid*/
    for(i=net->num_layers-2;i>0;i--){   /*for all hidden layers*/
        for(j=1; j<=net->npl[i];j++){   /*for all neurons except bias*/
            for(k=1; k<=net->npl[i+1]; k++){  /*for all neurons in next layer (error backpropagates)*/
                if( i == net->num_layers-2 ) /*if propagating from last hidden layer*/
                    net->d_hid[i][j-1] += net->d_out[k-1] * net->weight[i+1][k][j];
                else
                    net->d_hid[i][j-1] += net->d_hid[i+1][k-1] * net-> weight[i+1][k][j];
            }
            net->d_hid[i][j-1] *= sig_der( net->neuron[i][j] );
        }
    }

}


int adjust_weights(network* net, double lr){
    int i,j,k;

    for(i=net->num_layers-1;i>0;i--){   /*for all layers except input*/
        for(j=1; j<=net->npl[i];j++){   /*for all neurons except bias*/
            for(k=0; k<=net->npl[i-1]; k++){  /*for all neurons in previous layer bias included (weight from current to previous)*/
                if( i == net->num_layers-1 ) /*if adjusting last hidden layer*/
                    net->weight[i][j][k] += lr * net->d_out[j-1] * net->neuron [ i-1 ][ k ];
                else
                    net->weight[i][j][k] += lr * net->d_hid[i][j-1] * net->neuron [ i-1 ][ k ];
            }
        }
    }
    return 0;
}



int train(network* net, double input[][2], double output[][1], int length, double lr, double err_tgt){
    int max_it=110000;
    int report_every=100;
    double err;
    printf("start training\n ");
    for( int i=0; i<max_it; i++ ){
        err=0;
        for( int j=0; j<length; j++ ){
            forward(net, input[j]);
            /*out_err, delta_out, delta_hidd*/
            err+=out_err(net, output[j]);
            delta_out(net,output[j]);
            delta_hid(net);
            adjust_weights(net,lr);
            if(dbg){
                printf("delta_out=%f\n",net->d_out[0]);
                net_print(net);
            }
        }

        err=err*1/(2*length);
        if(i%report_every==0){
            printf("iteration=%d error=%f\n",i,err);
        }
        if (err<=err_tgt)
            return i;
    }
    return 0;
}




#endif
