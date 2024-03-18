#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>


#define num_layers 5
#define num_features 500
#define num_nodes 19717
#define learning_rate 0.001
#define num_edges 88648
#define num_edges 88648


typedef struct Node{
	int node;
	float *feature;

} Node;

typedef struct GNN{
	float bias;
	float *weight;
} GNN;



float relu(float x){
	return x > 0 ? x:0;
}

float sigmoid(float x){
	return 1/(1+exp(-x));
}

void initialize(GNN *layer){
       for(int i = 0;i<num_layers;i++){
       	       layer[i].bias = ((float)rand() / RAND_MAX) - 2.3;
 	       layer[i].weight = (float*)(malloc(num_features * sizeof(float)));
               for(int j = 0 ; j<num_features ; ++j){
			layer[i].weight[j] = ((float)rand() / RAND_MAX);			}
       }
}



void messagePassing(Node *node, GNN *layer, int *csr, int *dest, int *label){
	int x = 0, prev = 0, cnt = 0;
	for(int l = 0 ; l<num_layers ; ++l){
	#pragma omp parallel for num_threads(omp_get_max_threads())
		for(int i = 0; i< num_nodes; ++i){
			for(int j = 0; j<num_features; ++j){
				cnt = 0;
				float new_feat;
				for(int k = prev; k<=csr[i];k++){
					if(label[i] == label[dest[k]] && cnt<50){
						cnt++;
						new_feat += node[dest[k]].feature[j];
					}
				}
				node[i].feature[j] = relu(new_feat*layer[l].weight[j] + layer[l].bias);
			}
			x++;
			prev = csr[x];
		}
	}
}

float computeError(Node *node, int *labels){

    float mse = 0.0;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0;j<num_features;j++){
		float error = labels[i] - node[i].feature[j];
		mse += (error*error);
	    }
	}
	    mse /= num_nodes; 
	    return mse;
}


void backwardPass(Node *nodes,GNN* layer,int *labels) {           

    float error = computeError(nodes, labels);
    for (int i = 0; i < num_layers; i++) {
        for (int j = 0; j < num_features; j++) {
            layer[i].weight[j] -= learning_rate * error;
        }
        layer[i].bias -= learning_rate * error;
    }
}


void run(Node *nodes, GNN *layers,int labels[],int csr[], int dest[]){

    for (int epoch = 0; epoch < 100; epoch++) {
            messagePassing(nodes, layers,csr,dest,labels);
            printf("Hello\n");
        double current_mse = computeError(nodes, labels);
        printf("Epoch %d, MSE: %lf\n", epoch, current_mse);
        
        for (int layer = num_layers - 1; layer >= 0; layer--) {
            backwardPass(nodes, &layers[layer], labels);
        }
      }
 }


int main(){
    GNN layer[num_layers];
    Node node[num_nodes];
    int labels[num_nodes];
    int desttt[num_edges];

    FILE *src = fopen("/home/anubhav/GraphNN/GNN/pubmed/src.txt", "r");
    FILE *dest = fopen("/home/anubhav/GraphNN/GNN/pubmed/destination.txt", "r");
    FILE *feat = fopen("/home/anubhav/GraphNN/GNN/pubmed/features.txt", "r");
    FILE *label = fopen("/home/anubhav/GraphNN/GNN/pubmed/labels.txt", "r");

    

    int csr[num_nodes];
    int i = 0,x = -1;
    int prev = -11111;
    int destination;

    while (fscanf(src, "%d", &destination) ==1) { 
        if(prev != destination){
            prev = destination;
            node[i].node = destination;
            csr[i] = x;
            if(i==0){
            csr[i] = 0;}
            i++;
        }
        x++;
    }
    int j = 0;
     i = -1;
     float det;
    while (fscanf(feat, "%f", &det) ==1 ) {

        if(j%num_features==0){
            i++;
            j = 0;
            node[i].feature = (float*)malloc(num_features * sizeof(float)); 
        }
            // printf("%d  %d\n",j,i);
            node[i].feature[j] = det;
            // printf("%lf\n",nodes[i].feature[j]);
            j++;

    }

    i = 0;
    while (fscanf(label, "%d", &destination)==1 ) { 
            labels[i] = destination;
            i++;

    }


    i = 0;
    while (fscanf(dest, "%d", &destination) == 1 ) {
        desttt[i] = destination;
        i++;
    }
    initialize(layer);
    printf("%f",layer[0].weight[9]);


    run(node,layer,labels, csr, desttt);

    return 0;
}







		
