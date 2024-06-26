#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include "read_data.h"
#include "activation.h"


// Initialize constants used in optimizers
const double epsilon = 1e-8;
const double beta = 0.9;
const double beta_1 = 0.9;
const double beta_2 = 0.999;


// Pseudo-random number generator 
int seed;
double randn(){
  int a = 1103515245;
  int m = 2147483647;
  int c = 12345;
  seed = (a * seed + c) % m;
  double x = (double)seed/(double)m;
  return x;
}


// Neural Network struct definition
// n_layers -> Integer -> Stores the number of layers.
// n_neurons_per_layer -> Integer 1D array -> Stores the number of neurons in each layer.
// w -> Double 3D array -> Stores the weights between each neurons in each pair of layers.
// b -> Double 2D array -> Stores the bias weights from bias unit to neurons in the next layer.
// momentum_w -> Double 3D array -> Stores the first order moment for the weights.
// momentum_b -> Double 2D array -> Stores the first order moment for the bias.
// momentum2_w -> Double 3D array -> Stores the second order moment for the weights.
// momentum2_b -> Double 2D array -> Stores the second order moment for the weights.
// delta -> Double 2D array -> Stores the errors computed for each neuron in each layer.
// in -> Double 2D array -> Stores the input to the activation function in each layer.
// out -> Double 2D array -> Stores the output of the activation function in each layer.
// targets -> Double 1D array -> Stores the actual output for a given sample. It is a one-hot vector.
struct NeuralNet{
    int n_layers;
    int* n_neurons_per_layer;
    double*** w;
    double** b;
    double*** momentum_w;
    double*** momentum2_w;
    double** momentum_b;
    double** momentum2_b;
    double** delta;
    double** in;
    double** out;
    double* targets;
};


struct NeuralNet* newNet(int n_layers, int n_neurons_per_layer[]){
    struct NeuralNet* nn = malloc(sizeof(struct NeuralNet));
    nn->n_layers = n_layers;
    nn->n_neurons_per_layer = malloc(nn->n_layers * sizeof(int));
    
    // Copy number of neurons per layer
    for(int i=0;i<n_layers;i++){
        nn->n_neurons_per_layer[i] = n_neurons_per_layer[i];
    }
    
    // Allocate memory for weights and biases in parallel
    #pragma omp parallel for
    for(int i=0;i<nn->n_layers-1;i++){
        nn->w[i] = malloc((nn->n_neurons_per_layer[i] + 1)*sizeof(double*));
        nn->momentum_w[i] = malloc((nn->n_neurons_per_layer[i] + 1)*sizeof(double*));
        nn->momentum2_w[i] = malloc((nn->n_neurons_per_layer[i] + 1)*sizeof(double*));
        nn->b[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->momentum_b[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->momentum2_b[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        
        for(int j=0;j<nn->n_neurons_per_layer[i]+1;j++){
            nn->w[i][j] = malloc((nn->n_neurons_per_layer[i+1] + 1)*sizeof(double));
            nn->momentum_w[i][j] = malloc((nn->n_neurons_per_layer[i+1] + 1)*sizeof(double));
            nn->momentum2_w[i][j] = malloc((nn->n_neurons_per_layer[i+1] + 1)*sizeof(double));
        }
    }
    
    // Allocate memory for input, output, and delta in parallel
    #pragma omp parallel for
    for(int i=0;i<nn->n_layers;i++){
        nn->in[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->out[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->delta[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
    }
    
    // Allocate memory for targets
    nn->targets = malloc((nn->n_neurons_per_layer[nn->n_layers-1]+1)*sizeof(double));
    
    return nn;
}


// Function to free the dynamically allocated memory
void free_NN(struct NeuralNet* nn){
    for(int i=0;i<nn->n_layers-1;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i]+1;j++){
            free(nn->w[i][j]);
            free(nn->momentum_w[i][j]);
            free(nn->momentum2_w[i][j]);
        }
        free(nn->w[i]);
        free(nn->momentum_w[i]);
        free(nn->momentum2_w[i]);
        free(nn->b[i]);
        free(nn->momentum_b[i]);
        free(nn->momentum2_b[i]);
    }
    free(nn->w);
    free(nn->momentum_w);
    free(nn->momentum2_w);
    free(nn->b);
    free(nn->momentum_b);
    free(nn->momentum2_b);
    for(int i=0;i<nn->n_layers;i++){
        free(nn->in[i]);
        free(nn->out[i]);
        free(nn->delta[i]);
    }
    free(nn->in);
    free(nn->out);
    free(nn->delta);
    free(nn->targets);
    free(nn->n_neurons_per_layer);
}


// Initialize the neural network
void init_nn(struct NeuralNet* nn){
    for(int k=0;k<nn->n_layers-1;k++){
        for(int i=1;i<nn->n_neurons_per_layer[k]+1;i++){
            nn->b[k][i] = 0.0;
            nn->momentum_b[k][i] = 0.0;
            nn->momentum2_b[k][i] = 0.0;
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
                nn->w[k][i][j] = randn();
                nn->momentum_w[k][i][j] = 0.0;
                nn->momentum2_w[k][i][j] = 0.0;
            }
        }
    }
}


// Function to shuffle elements of an array
void shuffle(int* arr, size_t n){
    if(n > 1){
        for(size_t i=0;i<n-1;i++){
        size_t j = i+rand()/(RAND_MAX/(n-i)+1);
          int t = arr[j];
          arr[j] = arr[i];
          arr[i] = t;
        }
    }
}




void forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss){
    #pragma omp parallel for
    for(int i=0;i<nn->n_layers;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i]+1;j++){
            nn->in[i][j] = 0.0;
        }
    }
    for(int k=1;k<nn->n_layers;k++){
        // Compute the weighted sum
        #pragma omp parallel for
        for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
            nn->in[k][j] += 1.0 * nn->b[k-1][j];
        }
        #pragma omp parallel for collapse(2)
        for(int i=1;i<nn->n_neurons_per_layer[k-1]+1;i++){
            for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                nn->in[k][j] += nn->out[k-1][i] * nn->w[k-1][i][j];
            }
        }
        // Apply non-linear activation function to the weighted sums
        if(k == nn->n_layers-1){
            if(strcmp(loss, "mse") == 0){
                #pragma omp parallel for
                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                    nn->out[k][j] = sigmoid(nn->in[k][j]);
                }
            }
            else if(strcmp(loss, "ce") == 0){
                double max_input_to_softmax = (double)INT_MIN;
                #pragma omp parallel for reduction(max:max_input_to_softmax)
                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                    if(fabs(nn->in[k][j]) > max_input_to_softmax){
                        max_input_to_softmax = fabs(nn->in[k][j]);
                    }
                }
                double deno = 0.0;
                #pragma omp parallel for reduction(+:deno)
                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                    nn->in[k][j] /= max_input_to_softmax;
                    deno += exp(nn->in[k][j]);
                }
                #pragma omp parallel for
                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                    nn->out[k][j] = (double)exp(nn->in[k][j])/(double)deno;
                }
            }
        }
        else{
            #pragma omp parallel for
            for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                if(strcmp(activation_fun, "sigmoid") == 0){
                    nn->out[k][j] = sigmoid(nn->in[k][j]);
                }
                else if(strcmp(activation_fun, "tanh") == 0){
                    nn->out[k][j] = tanh(nn->in[k][j]);
                }
                else if(strcmp(activation_fun, "relu") == 0){
                    nn->out[k][j] = relu(nn->in[k][j]);
                }
                else{
                    nn->out[k][j] = sigmoid(nn->in[k][j]);
                }
            }
        }
    }
}


// Function to calculate loss
double calc_loss(struct NeuralNet* nn, char* loss){
    double loss_val = 0.0;
    int last_layer = nn->n_layers-1;
    for(int i=1;i<nn->n_neurons_per_layer[last_layer]+1;i++){
        if(strcmp(loss, "mse") == 0){
            loss_val += (0.5)*(nn->out[last_layer][i] - nn->targets[i]) * (nn->out[last_layer][i] - nn->targets[i]);
        }
        else if(strcmp(loss, "ce") == 0){
            loss_val -= nn->targets[i]*(log(nn->out[last_layer][i]));
        }
	}
    return loss_val;
}


#include <omp.h>

// Function for back propagation step
void back_propagation(struct NeuralNet* nn, char* activation_fun, double learning_rate, char* loss, char* opt, int itr) {
    int last_layer = nn->n_layers - 1;

    // Calculate the error in the output layer
    #pragma omp parallel for
    for (int i = 1; i < nn->n_neurons_per_layer[last_layer] + 1; i++) {
        double grad;
        if (strcmp(loss, "mse") == 0) {
            grad = sigmoid_d(nn->out[last_layer][i]);
        } else if (strcmp(loss, "ce") == 0) {
            grad = 1.0; // No need to compute the gradient explicitly
        }
        nn->delta[last_layer][i] = grad * (nn->out[last_layer][i] - nn->targets[i]);
    }

    // Backpropagate the error from the last layer to the first layer
    for (int k = nn->n_layers - 2; k > 0; k--) {
        #pragma omp parallel for
        for (int i = 1; i < nn->n_neurons_per_layer[k] + 1; i++) {
            double sum = 0.0;
            for (int j = 1; j < nn->n_neurons_per_layer[k + 1] + 1; j++) {
                sum += nn->b[k][j] * nn->delta[k + 1][j];
                sum += nn->w[k][i][j] * nn->delta[k + 1][j];
            }
            double grad;
            if (strcmp(activation_fun, "sigmoid") == 0) {
                grad = sigmoid_d(nn->out[k][i]);
            } else if (strcmp(activation_fun, "tanh") == 0) {
                grad = tanh_d(nn->out[k][i]);
            } else if (strcmp(activation_fun, "relu") == 0) {
                grad = relu_d(nn->out[k][i]);
            } else {
                grad = sigmoid_d(nn->out[k][i]);
            }
            nn->delta[k][i] = grad * sum;
        }
    }

    // Update the weights according to the given optimization technique
    #pragma omp parallel for collapse(3)
    for (int k = 0; k < nn->n_layers - 1; k++) {
        for (int i = 1; i < nn->n_neurons_per_layer[k] + 1; i++) {
            for (int j = 1; j < nn->n_neurons_per_layer[k + 1] + 1; j++) {
                double dw = nn->delta[k + 1][j] * nn->out[k][i];
                // Update weights based on optimization technique
                if (strcmp(opt, "sgd") == 0) {
                    nn->w[k][i][j] -= learning_rate * dw;
                } else if (strcmp(opt, "momentum") == 0) {
                    // Add your implementation for momentum optimization
                } else if (strcmp(opt, "rmsprop") == 0) {
                    // Add your implementation for RMSprop optimization
                } else if (strcmp(opt, "adam") == 0) {
                    // Add your implementation for Adam optimization
                }
            }
        }
        // Update the bias weights
        #pragma omp parallel for
        for (int j = 1; j < nn->n_neurons_per_layer[k + 1] + 1; j++) {
            double db = nn->delta[k + 1][j] * 1.0;
            // Update bias weights based on optimization technique
            if (strcmp(opt, "sgd") == 0) {
                nn->b[k][j] -= learning_rate * db;
            } else if (strcmp(opt, "momentum") == 0) {
                // Add your implementation for momentum optimization
            } else if (strcmp(opt, "rmsprop") == 0) {
                // Add your implementation for RMSprop optimization
            } else if (strcmp(opt, "adam") == 0) {
                // Add your implementation for Adam optimization
            }
        }
    }
}



// Function to train the model for 1 epoch
double* model_train(struct NeuralNet* nn, double** X_train, double** y_train, double* y_train_temp, 
                    char* activation_fun, char* loss, char* opt, double learning_rate,
                    int num_samples_to_train, int itr){     
    int arr[N_SAMPLES];
    for(int i=0;i<N_SAMPLES;i++){
        arr[i] = i;
    }
    shuffle(arr, N_SAMPLES);
    int shuffler[num_samples_to_train];
    for(int i=0;i<num_samples_to_train;i++){
        shuffler[i] = arr[i];
    }
    int correct = 0;
    double loss_val = 0.0;
    for(int i=0;i<num_samples_to_train;i++){
        shuffle(shuffler, num_samples_to_train);
        int idx = -1;
        double max_val = (double)INT_MIN;
        for(int j=1;j<nn->n_neurons_per_layer[0]+1;j++){
            nn->out[0][j] = X_train[arr[i]][j-1];
        }
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            nn->targets[j] = y_train[arr[i]][j-1];
        }
        forward_propagation(nn, activation_fun, loss);
        back_propagation(nn, activation_fun, learning_rate, loss, opt, itr);
        loss_val += calc_loss(nn, loss);
            
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            if(nn->out[nn->n_layers-1][j] > max_val){
                max_val =nn->out[nn->n_layers-1][j];
                idx = j-1;
            }
        }
        if(idx == (int)y_train_temp[arr[i]]){
            correct++;
        }
    }
    loss_val /=(double)num_samples_to_train;
    double accuracy = (double)correct/(double)num_samples_to_train;
    static double metrics[2];
    metrics[0] = loss_val;
    metrics[1] = accuracy;
    return metrics;
}


// Function to test the model
double* model_test(struct NeuralNet* nn, double** X_test, double** y_test, double* y_test_temp, char* activation_fun, char* loss){
    int correct = 0;
    double loss_val = 0.0;
    for(int i=0;i<N_TEST_SAMPLES;i++){
        int idx = -1;
        double max_val = (double)INT_MIN;
        for(int j=1;j<nn->n_neurons_per_layer[0]+1;j++){
            nn->out[0][j] = X_test[i][j-1];
        }
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            nn->targets[j] = y_test[i][j-1];
        }
        forward_propagation(nn, activation_fun, loss);
        loss_val += calc_loss(nn, loss);
            
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            if(nn->out[nn->n_layers-1][j] > max_val){
                max_val =nn->out[nn->n_layers-1][j];
                idx = j-1;
            }
        }
        if(idx == (int)y_test_temp[i]){
            correct++;
        }
    }
    loss_val /= (double)N_TEST_SAMPLES;
    double accuracy = (double)correct/(double)N_TEST_SAMPLES;
    static double metrics[2];
    metrics[0] = loss_val;
    metrics[1] = accuracy;
    return metrics;
}


int main(){

    // Used for setting a random seed
    srand(time(NULL));
    int seed = rand();

    // Initialize neural network architecture parameters
    int n_layers = 4;
    int n_neurons_per_layer[] = {784, 64, 32, 10};

    // Create and initialize the neural network
    struct NeuralNet* nn = newNet(n_layers, n_neurons_per_layer);
    init_nn(nn);

    // Initialize the learning rate, optimizer, loss, and other hyper-parameters
    double learning_rate = 1e-4;
    double init_lr = 1e-4;
    char* activation_fun = "relu";
    char* loss = "ce";
    char* opt = "adam";
    int num_samples_to_train = 10000;
    int epochs = 5;

    // Fetch the training and test data and pre-process them
    double** X_train = malloc(N_SAMPLES*sizeof(double*));
    for(int i=0;i<N_SAMPLES;i++){
        X_train[i] = malloc(N_DIMS*sizeof(double));
    }
    double** y_train = malloc(N_SAMPLES * sizeof(double*));
    for(int i=0;i<N_SAMPLES;i++){
        y_train[i] = malloc(N_CLASSES * sizeof(double));
    }
    double* y_train_temp = malloc(N_SAMPLES*sizeof(double));
    read_csv_file(X_train, y_train_temp, y_train, "train");
    scale_data(X_train, "train");

    double** X_test = malloc(N_TEST_SAMPLES*sizeof(double*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        X_test[i] = malloc(N_DIMS*sizeof(double));
    }
    double** y_test = malloc(N_TEST_SAMPLES * sizeof(double*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        y_test[i] = malloc(N_CLASSES * sizeof(double));
    }
    double* y_test_temp = malloc(N_TEST_SAMPLES*sizeof(double));
    read_csv_file(X_test, y_test_temp, y_test, "test");
    scale_data(X_test, "test");
    normalize_data(X_train, X_test);

    // Initialize file to store metrics info for each epoch
    FILE* file = fopen("metrics_64_32.txt", "w");
    fprintf(file, "train_loss,train_acc,test_loss,test_acc\n");
    
    // Train the model for given number of epoch and test it after every epoch
    for(int itr=0;itr<epochs;itr++){
        double* train_metrics = model_train(nn, X_train, y_train, y_train_temp, activation_fun, loss, opt, learning_rate, num_samples_to_train, itr+1);
        double train_loss = train_metrics[0];
        double train_acc = train_metrics[1];
        double* test_metrics = model_test(nn, X_test, y_test, y_test_temp, activation_fun, loss);
        double test_loss = test_metrics[0];
        double test_acc = test_metrics[1];

        fprintf(file, "%lf,", train_loss);
        fprintf(file, "%lf,", train_acc);
        fprintf(file, "%lf,", test_loss);
        fprintf(file, "%lf\n", test_acc);

        printf("Epoch: %d -> ", itr+1);
        printf("Train loss: %lf, ", train_loss);
        printf("Train Accuracy: %lf, ", train_acc);
        printf("Test loss: %lf, ", test_loss);
        printf("Test Accuracy: %lf\n", test_acc);

        learning_rate = init_lr * exp(-0.1 * (itr+1));

    }

    // Close the file
    fclose(file);

    // Free the dynamically allocated memory
    free_NN(nn);

    return 0;
}
