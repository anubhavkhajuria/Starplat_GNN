#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-10 


double calculateMean(double *data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum / n;
}


void computeCovarianceMatrix(double **data, int numRows, int numCols, double **covMatrix) {

    double means[numCols];
    for (int j = 0; j < numCols; ++j) {
        means[j] = calculateMean(data[j], numRows);
    }


    for (int i = 0; i < numCols; ++i) {
        for (int j = i; j < numCols; ++j) { 
            double cov = 0.0;
            for (int k = 0; k < numRows; ++k) {
                cov += (data[i][k] - means[i]) * (data[j][k] - means[j]);
            }
            cov /= numRows;
            covMatrix[i][j] = cov;
            covMatrix[j][i] = cov;
        }
    }
}

void jacobiRotation(double **A, double **V, int p, int q, int n) {
    double tau = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
    double t = (tau >= 0) ? 1.0 / (tau + sqrt(1.0 + tau * tau)) : -1.0 / (-tau + sqrt(1.0 + tau * tau));
    double c = 1.0 / sqrt(1.0 + t * t);
    double s = c * t;


    double Apq = A[p][q];
    A[p][p] -= t * Apq;
    A[q][q] += t * Apq;
    A[p][q] = A[q][p] = 0.0;
    for (int i = 0; i < n; ++i) {
        if (i != p && i != q) {
            double Aip = A[i][p];
            double Aiq = A[i][q];
            A[i][p] = A[p][i] = c * Aip - s * Aiq;
            A[i][q] = A[q][i] = s * Aip + c * Aiq;
        }
    }


    for (int i = 0; i < n; ++i) {
        double Vip = V[i][p];
        double Viq = V[i][q];
        V[i][p] = c * Vip - s * Viq;
        V[i][q] = s * Vip + c * Viq;
    }
}


void eigenDecomposition(double **A, int n, double *eigenvalues, double **eigenvectors) {

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    int iter = 0;
    int maxIter = n * n * n; 
    int p, q;
    double offDiagMax;
    do {

        offDiagMax = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (fabs(A[i][j]) > offDiagMax) {
                    offDiagMax = fabs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }


        if (offDiagMax > EPSILON) {
            jacobiRotation(A, eigenvectors, p, q, n);
        }

        iter++;
    } while (offDiagMax > EPSILON && iter < maxIter);


    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = A[i][i];
    }
}

int main() {

    double data[3][5] = {
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {2.0, 3.0, 4.0, 5.0, 6.0},
        {3.0, 4.0, 5.0, 6.0, 7.0}
    };
    int numRows = sizeof(data[0]) / sizeof(double);
    int numCols = sizeof(data) / sizeof(data[0]);


    double **covarianceMatrix = malloc(numCols * sizeof(double *));
    for (int i = 0; i < numCols; ++i) {
        covarianceMatrix[i] = malloc(numCols * sizeof(double));
    }


    computeCovarianceMatrix(data, numRows, numCols, covarianceMatrix);


    double *eigenvalues = malloc(numCols * sizeof(double));
    double **eigenvectors = malloc(numCols * sizeof(double *));
    for (int i = 0; i < numCols; ++i) {
        eigenvectors[i] = malloc(numCols * sizeof(double));
    }


    eigenDecomposition(covarianceMatrix, numCols, eigenvalues, eigenvectors);


    printf("Eigenvalues:\n");
    for (int i = 0; i < numCols; ++i) {
        printf("%lf\n", eigenvalues[i]);
    }


    printf("Eigenvectors:\n");
    for (int i = 0; i < numCols; ++i) {
        for (int j = 0; j < numCols; ++j) {
            printf("%lf ", eigenvectors[i][j]);
        }
        printf("\n");
    }


    free(eigenvalues);
    for (int i = 0; i < numCols; ++i) {
        free(eigenvectors[i]);
    }
    free(eigenvectors);

    for (int i = 0; i < numCols; ++i) {
        free(covarianceMatrix[i]);
    }
    free(covarianceMatrix);

    return 0;
}

