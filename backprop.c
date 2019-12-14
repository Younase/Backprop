#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "net.c"
#define dbg 1

float x(float a){
	// printf("activation:%f\n",a );
	return a;
}


int main(){
int j[3]={3,2,3};
// k[8]={1,1,0,1,1,0,0,0},o[4]={0,1,1,0};
net* network=init(3,j,&x);
// network->weight[0][0][0]=-0.8;
// network->weight[0][0][1]=0.1;
// network->weight[0][1][0]=0.5;
// network->weight[0][1][1]=0.9;
// network->weight[0][2][0]=0.4;
// network->weight[0][2][1]=1;
// network->weight[1][0][0]=-0.3;
// network->weight[1][1][0]=-1.2;
// network->weight[1][2][0]=1.1;
int c[3]={1,0,1};
eval(network,c);
print(network);
int b[3]={7,7,7};
if(equals(network,b))
  printf("b equals net output\n");
  else
  printf("not\n" );
// printf("train result: %d\n",train(network,k,o,4,0.1));
// //eval(network,k);
// printf("%f\n",network->neuron[1][1]);
// eval(network,&(k[0]));
// print(network);
// eval(network,&(k[2]));
// print(network);
// eval(network,&(k[4]));
// print(network);
// eval(network,&(k[6]));
// print(network);
return 0;

}
