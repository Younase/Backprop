#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "net.c"


int main(){
    int j[3]={2,2,1},
        k[8]={1,1,0,1,1,0,0,0},o[4]={0,1,1,0};
    
    printf("train result: %d\n",train(network,k,o,4,0.1,0.01));

        return 0;

}
