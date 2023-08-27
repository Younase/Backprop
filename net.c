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
    for (int i=0; i<num_layers; i++){
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

int train(net* network,int* in,int* out,int length,float le_rate,float error_target){
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
    if(dbg)
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
if(dbg)
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
if(dbg)
        print_err(err,max);
        tmp=last_err;
        last_err=err;
        err=tmp;
    }



}
error=0.5*error;
if(curr_it%log_every==0)
printf("current iteration:%7d error = %10f\n",max_it-curr_it,error );
if(error<error_target){
  printf("current iteration:%7d error = %10f\n",max_it-curr_it,error );
  break;
}
}
return curr_it;
}






#endif
