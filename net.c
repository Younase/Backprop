#ifndef net
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define dbg 1

typedef struct {
float** neuron;
float*** weight;
int* npl; //neuron per layer
int num_layers;   // io included
float (*activation)(float);//activation for every layer but input
float (*act_der)(float);///activation derivative
/////to do: add activation table: activation for each layer
}net;

///////    A REFAIRE
float rand_(){

return 1;
}

////// TESTED
net* init(int num_layers,int* npl,float (*act)(float),float (*act_der)(float)){
if(num_layers<1 || npl==NULL){
	printf("init args=null");
	return NULL;
}
//////test npl>0
int i,j,k;
net* network=(net*)malloc(sizeof(net));

network->activation=act;
network->act_der=act_der;
network->num_layers=num_layers;
network->npl=(int*)malloc(num_layers*sizeof(int));
network->neuron=(float**)malloc(num_layers*sizeof(float*));
network->weight=(float***)malloc(num_layers*sizeof(float**));

network->neuron[0]=(float*)malloc((1+npl[0])*sizeof(float));
network->npl[0]=npl[0];
network->neuron[0][0]=1;
for(i=1;i<num_layers;i++){
	network->weight[i-1]=(float**)malloc((1+npl[i-1])*sizeof(float*)); //1 for bias
	network->npl[i]=npl[i];
	network->neuron[i]=(float*)malloc((1+npl[i])*sizeof(float)); // bias in each layer remodded
	network->neuron[i][0]=1;
	for(j=0;j<=npl[i-1];j++){
		network->weight[i-1][j]=(float*)malloc(npl[i]*sizeof(float));
		for(k=0;k<npl[i];k++){
	 		network->weight[i-1][j][k]=rand_();
		}
	}
}
/////////// last layer has unused bias
return network;

}

/////   TESTED
int eval(net* network,int* in){
int i,j,k;
for(i=1;i<=network->npl[0];i++){
	network->neuron[0][i]=in[i-1];
	//printf("%d",in[i-1]);
}
for(i=1;i<network->num_layers;i++){
	for(j=1;j<=network->npl[i];j++){//skip bias modded
		network->neuron[i][j]=0;
		for(k=0;k<=network->npl[i-1];k++){//bias
			network->neuron[i][j]+=network->neuron[i-1][k]*network->weight[i-1][k][j-1];

		}
		network->neuron[i][j]=network->activation(network->neuron[i][j]);
	// network->neuron[i][j]=(network->neuron[i][j]>0)?1:0;
	}
}
// network->neuron[i-1][j-1]=(network->neuron[i-1][j-1]>0)?1:0;
return 1;

}

//////   TESTED
void print(net* network){
	int i,j,k;
// printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t\n",network->neuron[0][0],network->neuron[0][1],
// network->neuron[0][2],network->neuron[1][1],network->weight[0][0][0],
// network->weight[0][1][0],network->weight[0][2][0]);
printf("neurons\n" );
for ( i = 0; i < network->num_layers; i++){
	for (j = 0; j <= network->npl[i]; j++)
		printf("%f\t",network->neuron[i][j]);
  printf("\n");
  }
printf("\n" );
printf("weights\n");
		for ( i = 0; i < network->num_layers-1; i++){
			for (j = 0; j <= network->npl[i]; j++)
			for (k = 0; k < network->npl[i+1]; k++)
			 printf("%f\t",network->weight[i][j][k]);
      printf("\n");
     }
printf("\n" );
}

/////    TESTED
int equals(net* network,int* out){
int i;
	for(i=0;i<network->npl[network->num_layers-1];i++)
		if(network->neuron[network->num_layers-1][i+1]!=out[i])
			return 0;
return 1;
}

int max_tab(int* t,int size){
int max=t[0],i;
	if (size>2)
		for ( i = 1; i < size; i++)
			if(t[i]>max)
				max=t[i];
	return max;
}

void print_err(float* err,int s){
	int i;
	for(i=0;i<s;i++){
		printf("%f\t",err[i] );
	}
	printf("\n");

}

int train(net* network,int* in,int* out,int length,float le_rate){
  //flag holds last it where weights modded
int max_it=10000,curr_it,flag=0,i,j,k,l,max=max_tab(network->npl,network->num_layers);
float error,**weights,*tmp,*err=malloc(max*sizeof(float)),*last_err=malloc(max*sizeof(float));
curr_it=max_it;
while(flag!=curr_it+length && curr_it){
  curr_it--;
  for (i = 0; i < length; i++) {
    error=0;
    eval(network,in+i*network->npl[0]);//////evaluate network
    if(dbg){
      printf("desired output:\n" );
      for(l=0;l<network->npl[network->num_layers-1];l++)//////// print desired output
        printf("%d\t",out[i*network->npl[1]+l]);
      printf("\n");
      print(network);/////print network state
    }

    if(equals(network,out+i*network->npl[1])){
      continue;
    }
    flag=curr_it;
    printf("weights getting modded\n");
//////     TESTED UNTIL HERE
    ///allocating for old_weights
    weights=(float**)malloc((1+network->npl[network->num_layers-2])*sizeof(float*));//bias not needed, but added for j loop compat
    for(j=0;j<=network->npl[network->num_layers-2];j++){
      weights[j]=(float*)malloc(network->npl[network->num_layers-1]*sizeof(float*));
    }
    //////     OUTPUT ERROR
    for(k=1;k<=network->npl[network->num_layers-1];k++){
      err[k-1]=(out[i*network->npl[1]+k-1]-network->neuron[network->num_layers-1][k])
              *network->act_der(network->neuron[network->num_layers-1][k]);   ///// no error for bias
      error+=(out[i*network->npl[1]+k-1]-network->neuron[network->num_layers-1][k])*
      (out[i*network->npl[1]+k-1]-network->neuron[network->num_layers-1][k]);
      ///     OUTPUT WEIGHT

      for(j=0;j<=network->npl[network->num_layers-2];j++){
        weights[j][k-1]=network->weight[network->num_layers-2][j][k-1];
        network->weight[network->num_layers-2][j][k-1]+=le_rate*err[k-1]*network->neuron[network->num_layers-2][j];
      }
    }

print_err(err,max);
tmp=last_err;
last_err=err;
err=tmp;

// print_err(err,max);
// print_err(last_err,max);

/**********   HIDDEN ERRORS    ***********/

  for(l=network->num_layers-2;l>0;l--){




      for(j=1;j<=network->npl[l];j++){
        err[j-1]=0;
        for(k=0;k<network->npl[l+1];k++){
          err[j-1]+=weights[j][k]*last_err[k];//add -1
        }
        err[j-1]*=network->act_der(network->neuron[l][j]);
        free(weights[j]);
        // printf("%f\t",err[j-1]);
        }
        // for(j=0;j<=network->npl[l];j++){
        //   free(weights[j]);
        // }
        free(weights);
        weights=(float**)malloc((1+network->npl[l])*sizeof(float*));//bias not needed, but added for j loop compat
        for(j=0;j<=network->npl[l];j++){
          weights[j]=(float*)malloc(network->npl[l+1]*sizeof(float*));
        }
        ///// free/allocate done
        /////save weights
        /////mod weights
        for(k=0;k<=network->npl[l+1];k++){
          for(j=0;j<=network->npl[l];j++){
            weights[j][k]=network->weight[l-1][j][k];
            network->weight[l-1][j][k]+=le_rate*err[k]*network->neuron[l-1][j];
          }
        }
        //
        // for(k=1;k<=network->npl[l+1];k++){
        //   for(j=0;j<=network->npl[l];j++){
        //     weights[j][k-1]=network->weight[l][j][k-1];
        //     network->weight[l][j][k-1]+=le_rate*err[k-1]*network->neuron[l][j];
        //   }
        // }

        print_err(err,max);
        tmp=last_err;
        last_err=err;
        err=tmp;
    }



}
error=0.5*error;
printf("error = %f///////////////////////////////////////\n",error );
}
return curr_it;
}






#endif
