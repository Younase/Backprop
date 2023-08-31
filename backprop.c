#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "net.c"

int num_layers=3;
int nrns_per_layer[]={2,2,1};
char activation[]="sigmoid";
double f[]={1,1};
double  k[4][2]={{1,1},{0,1},{1,0},{0,0}};
double  o[4][1]={{0},{1},{1},{0}};


double init_w_l2[2][3]={{-0.8,0.5,0.4},{0.1,0.9,1}};
double init_w_l3[1][3]={{-0.3,-1.2,1.1}};
double lr=0.1;





int main(){
    network* net;


    net=net_init(num_layers, nrns_per_layer, activation);
    train(net,(double *)k,(double *)o,4,lr,0.001);

    while (1){
        for( int i=0; i<2; i++ ){
            printf("X[%d]=",i);
            scanf("%lf",&f[i]);
            printf("X[%d]=%lf\n",i,f[i]);
        }
        forward(net,f);
        net_print(net);

    }


    return 0;

}
