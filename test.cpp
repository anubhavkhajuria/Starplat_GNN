#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include<omp.h>

// Hyperparameters
#define learning_rate 0.001
#define beta1 0.9
#define beta2 0.999
#define epsilon 1e-8



double loss_cal(double* y_true, double* y_pred) {   //Calculating Loss 
    double sum = 0.0;
    for (int i = 0; i < 19717; i++) {
        double diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    return sum / 19717;
}

void linear_regression(double** x, double* y_pred, double w, double b) {      // Calculating linear regression
    //#pragma omp parallel for
    for (int i = 0; i < 19717; i++) {
        for(int j = 0; j < 500; j++)
            y_pred[i] = w * x[i][j] + b;
    }
}


void compute_gradients(double** x, double* y_true, double* y_pred, double* gradient_w, double* gradient_b) {
   //#pragma omp parallel for                                     // Calculating gradient descent to miminize loss
    for (int i = 0; i < 19717; i++) {
        for(int j = 0; j < 500; j++){
        double diff = y_true[i] - y_pred[i];
        gradient_w[0] += -2* x[i][j] * diff;
        gradient_b[0] +=  -2 * diff;
        }
}   
    gradient_w[0] /= 19717;
    gradient_b[0] /= 19717;
}

int main() {

    FILE *file;
    file = fopen("/home/anubhav/GraphNN/GNN/pubmed/features.txt", "r");         // reading the dataset  
    int rows = 19717;
    int cols = 500;

    double **x = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        x[i] = (double *)malloc(cols * sizeof(double));
    }
   for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf", &x[i][j]) != 1) {
                fprintf(stderr, "Error reading data from the file\n");
                return 1;
            }
        }
   }
    file = fopen("/home/anubhav/GraphNN/GNN/pubmed/labels.txt", "r");  
    double *y_true = (double *)malloc(rows * sizeof(double *));
   for (int i = 0; i < rows; i++) {
            if (fscanf(file, "%lf", &y_true[i]) != 1) {
                fprintf(stderr, "Error reading data from the file\n");
                return 1;
            }
    }
    double lo = 0.0;
    double w = 1.0;
    double b = 0.0;
    double y_pred[19717];
    double gradient_w[1] = {0.0};
    double gradient_b[1] = {0.0};
    double start_time = omp_get_wtime();
    for (int epoch = 0; epoch < 500; epoch++) {
        linear_regression(x, y_pred, w, b);
        compute_gradients(x, y_true, y_pred, gradient_w, gradient_b);

        w -= learning_rate * gradient_w[0]; // Update parameters using gradient descent
        b -= learning_rate * gradient_b[0];

        double loss_ = loss_cal(y_true, y_pred);
        //printf("Epoch %d: Loss=%.4f, b=%.4f\n", epoch + 1, loss_, b);
        lo = loss_;
    }
    printf("Loss=%.4f, b=%.4f\n", lo, b);
    double end = omp_get_wtime();
    printf("%f",end-start_time);
    return 0;
}
