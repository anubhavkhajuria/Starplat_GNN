#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<omp.h>

#define learning_rate 0.001
#define num_features 500
#define num_edges 88648
#define num_layers 5
#define num_nodes 19717
#define epsilon 1e-8

typedef struct Node{
    int  node;
    double *feature;
} Node;

typedef struct NodeWeight{
    double *weights;
    double bias;
}NodeWeight;


void initializeGNLayer(NodeWeight *layer) {  

        for (int i = 0; i < num_nodes; i++) {
            layer[i].weights = (double *) malloc(num_features * sizeof(double));
            layer[i].bias = 0.01;
          //initializing random values to the weight matrix 
            for (int j = 0; j < num_features; j++) {
                layer[i].weights[j] = ((float)rand() / RAND_MAX) - 0.5;
            }
        }

}

float relu(double x) {       //activation function
    return x > 0 ? x : 0;
}

float sigmoid(double x){
    return 1/(1+exp(-x));
}

void messagePassing(Node nodes[], NodeWeight *layer,int csr[],int dest[],int label[]) {
    long x=  0,prev = 0;
    for (int i = 0; i < num_nodes; i++) { 
              #pragma omp parallel for                          // updates the features of a node
        for (int j = 0; j < num_features; j++) {
            double new_feature = 0.0;
            for (int k = prev; k <= csr[i]; k++) {

                     int g = dest[k];
                     //if(label[i] == label[g])
                        new_feature += nodes[g].feature[1] ;
                        //printf("%lf\n",new_feature);

            }
           double y = relu(new_feature*layer[i].weights[j]+layer[i].bias);
            nodes[i].feature[j] = y;

        }
        x++;
        prev = csr[x];
    }
}


double computeMSE(Node nodes[], int labels[]) {          //Calculating mean square error to reduce loss
    double mse = 0.0;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0;j<num_features;j++){
            double prediction = nodes[i].feature[j]; 
            double error = prediction - labels[i];
            mse += (error * error);
        }
    }
    return mse / num_nodes;
}


double computeGradient(Node nodes[],  NodeWeight* layer, int labels[], int node_index,int feature_index,double mse) {
    double loss = mse;
    double original_weight = layer[node_index].weights[0];


    layer[node_index].weights[feature_index] += epsilon;        // Perturb the weight slightly and compute the loss
    //double perturbed_loss = computeMSE(nodes, labels);

    layer[node_index].weights[feature_index] = original_weight;         // Reset the weight

    double gradient = (loss) / epsilon;                // Compute the gradient

    return gradient;
}



void backwardPass(Node nodes[],  NodeWeight* layer,int labels[],double mse) {           //Backward propagation function
//#pragma omp parallel for
    for (int i = 0; i < num_features; i++) {
        #pragma omp parallel for
        for (int j= 0;j<num_features;j++){
            double gradient = computeGradient(nodes,  layer, labels, i,j,mse);
            layer[i].weights[j] -= learning_rate * gradient;
        }
    }
}


void run(Node *nodes, NodeWeight *layers,int labels[],int csr[], int dest[]){
    float start = omp_get_wtime();
    for (int epoch = 0; epoch < 100; epoch++) {
        for (int layer = 0; layer < num_layers; layer++) {
            messagePassing(nodes, layers,csr,dest,labels);
        }

        double current_mse = computeMSE(nodes, labels);
        printf("Epoch %d, MSE: %lf\n", epoch, current_mse);

        for (int layer = num_layers - 1; layer >= 0; layer--) {
            backwardPass(nodes,layers, labels,current_mse);
            // printf("Hello world\n");
        }
      }
      float end = omp_get_wtime();
      printf("%f",end-start);
      printf("Done");
 }


int main(){
    Node *nodes = (Node *) malloc(num_nodes * sizeof(Node));
    NodeWeight *layers = (NodeWeight *) malloc(num_nodes * sizeof(NodeWeight));
    initializeGNLayer(layers);    
    int dest,csr[num_nodes];
    int i = 0,x = -1;
    int prev = -11111;
    FILE *noddes = fopen("/home/anubhav/GraphNN/GNN/pubmed/src.txt", "r");
    i = 0;
    while (fscanf(noddes, "%d", &dest) ==1) { 
        if(prev != dest){
            prev = dest;
            nodes[i].node = dest;
            csr[i] = x;
            if(i==0){
            csr[i] = 0;}
            i++;
        }
        x++;
    }

    int j = 0;
    i = 0;
    double destn;
   
    FILE *feat = fopen("/home/anubhav/GraphNN/GNN/pubmed/features.txt", "r");
    while (fscanf(feat, "%lf", &destn) ==1 ) {

        if(j%num_features==0){
            j = 0;
            nodes[i].feature = (double *) malloc(num_features * sizeof(double)); 
            i++;
        }
            nodes[i-1].feature[j] = destn;
            j++;
    }
                    printf("%lf \n",nodes[0].feature[0]);



    FILE *desst = fopen("/home/anubhav/GraphNN/GNN/pubmed/destination.txt", "r");
    int destination[num_edges];
    i = 0;
    while (fscanf(desst, "%d", &dest) == 1 ) {
        destination[i] = dest;
        i++;
    }


    int *labels,labl;
    labels = (int*) malloc(num_nodes * sizeof(int));
    FILE *label = fopen("/home/anubhav/GraphNN/GNN/pubmed/labels.txt", "r");
    i = 0;
    while (fscanf(label, "%d", &labl)==1 ) { 
            labels[i] = destn;
            i++;
    }
    run(nodes,layers,labels,csr,destination);

    return 0;


}
 