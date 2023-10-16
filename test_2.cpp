#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define learning_rate 0.001
#define num_features 500
#define num_edges 88648
#define num_layers 5
#define num_nodes 19717
#define epsilon 1e-8

typedef struct Node{
    double  node;
    double *feature;
} Node;


typedef struct Egdes{
    double src_node;
    double dest_node;
} Edges;

typedef struct GNLayers{
    double weights[num_features][num_features];
    double *bias;
}GNLayers;

void initializeGNLayer(GNLayers * layer) {
    for (int i = 0; i < num_features; i++) {
        for (int j = 0; j < num_features; j++) {
            layer->weights[i][j] = ((float)rand() / RAND_MAX) - 0.5;
        }
        layer->bias[i] = 0.0;
    }
}
 

float relu(float x) {       //activation function
    return x > 0 ? x : 0;
}

void messagePassing(Node nodes[], Edges edges[], GNLayers* layer) {
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_features; j++) {
            float new_feature = 0.0;
            for (int k = 0; k < num_nodes; k++) {
                if (k != i) {
                    new_feature += nodes[k].feature[j] * 1;
                }
            }
            nodes[i].feature[j] = relu(new_feature * layer->weights[j][j] + layer->bias[j]);
        }
    }
}


double computeMSE(Node nodes[], double labels[]) {
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



double computeGradient(Node nodes[], Edges edges[], GNLayers* layer, double labels[], int node_index, int feature_index) {
    double loss = computeMSE(nodes, labels);
    double original_weight = layer->weights[feature_index][feature_index];


    layer->weights[feature_index][feature_index] += epsilon;        // Perturb the weight slightly and compute the loss
    double perturbed_loss = computeMSE(nodes, labels);

    layer->weights[feature_index][feature_index] = original_weight;         // Reset the weight

    double gradient = (perturbed_loss - loss) / epsilon;                // Compute the gradient

    return gradient;
}



void backwardPass(Node nodes[], Edges edges[], GNLayers* layer,double labels[]) {

    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_features; j++) {
            double gradient = computeGradient(nodes, edges, layer, labels, i, j);
            layer->weights[j][j] -= learning_rate * gradient;
        }
    }
}


int main(){
    GNLayers layers[num_layers];
    Edges edges[88648];
    Node nodes[19718];


    double labels[88648];

    FILE *src = fopen("/home/anubhav/GraphNN/GNN/pubmed/src.txt", "r");
    FILE *dest = fopen("/home/anubhav/GraphNN/GNN/pubmed/destination.txt", "r");

    double source, destination;
    int i = 0;
    while (fscanf(dest, "%lf", &source) == 1) { // Check the return value of fscanf
        edges[i].dest_node = source;
        //printf("%lf  %lf   ", edges[i].src_node,source); // Print the value you just stored
        i++;
        if (i == 88648) {
            break;
        }
}
    // printf("3\n");
    while (fscanf(src, "%lf", &source) == 1) { // Check the return value of fscanf
        edges[i].src_node = source;
        // printf("%lf    ", edges[i].src_node); // Print the value you just stored
        i++;
        if (i == 88648) {
            break;
        }
    }
    i = 0;
    double prev = -11111;
    FILE *noddes = fopen("/home/anubhav/GraphNN/GNN/pubmed/src.txt", "r");
    i = 0;
    while (fscanf(noddes, "%lf", &destination) ) { 
        if(prev != destination){
            prev = destination;
            nodes[i].node = destination;
            i++;
            // printf("%d\n",i);
        }
        if(i==19717)
        break;
    }
    int j = 1;
    i = 0;
   
    FILE *feat = fopen("/home/anubhav/GraphNN/GNN/pubmed/features.txt", "r");
    

    while (fscanf(feat, "%lf", &destination) ==1 ) {
            nodes[i].feature = (double*)malloc(500 * sizeof(double)); 
            //printf("%d  %d\n",j,i);
            nodes[i].feature[j] = destination;
           // printf("%lf\n",nodes[i].feature[j]);
            j++;
        if(j%500==0){
            i++;
            j = 0;
        }

        if(i==19717){
            break;
        }
    }
    printf("  %lf\n",nodes[19717].feature[7]);

    FILE *label = fopen("/home/anubhav/GraphNN/GNN/pubmed/labels.txt", "r");
    i = 0;
    while (fscanf(label, "%lf", &destination) ) { 
            labels[i] = destination;
            i++;
    }



    for (int epoch = 0; epoch < 100; epoch++) {
        for (int layer = 0; layer < num_layers; layer++) {
            messagePassing(nodes, edges, &layers[layer]);
        }

        double current_mse = computeMSE(nodes, labels);
        printf("Epoch %d, MSE: %lf\n", epoch, current_mse);

        for (int layer = num_layers - 1; layer >= 0; layer--) {
            backwardPass(nodes, edges, &layers[layer], labels);
        }
    }

    return 0;
}
