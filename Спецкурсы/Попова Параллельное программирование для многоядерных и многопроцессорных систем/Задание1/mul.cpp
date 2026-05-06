#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

using namespace std;

void getSpace(double**& matrix, int row, int col) {
    matrix = new double*[row];
    for(int i = 0; i < row; i++) {
        matrix[i] = new double[col];
    }
}

void clean(double** matrix, int row) {
    for(int i = 0; i < row; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void readMatrix(double**& matrix, string& filename, int& row, int& col) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Can't open file" << filename << endl;
        exit(1);
    }
    int row1;
    file >> row1 >> col;
    if (row1 <=0 || col <= 0) {
        cout << "Invalid row or col values" << endl;
        exit(2);
    }

    if((row != 0 && row != row1)) {
        cout << "Matrixs have different dimensions" << endl;
        exit(4);
    }
    row = row1;

    getSpace(matrix, row, col);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (!(file >> matrix[i][j])) {
                cout << "Error: Invalid data in file" << endl;
                exit(4);
            }
        }
    }
    
    file.close();
}

void writeMatrix(double** matrix, string& filename, int row, int col) {
    ofstream file(filename);
    if (!file.is_open()) {
        cout << "Can't open file " << filename << endl;
        exit(5);
    }
    
    file << row << " " << col << endl;
    
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            file << matrix[i][j] << " ";
        }
        file << endl;
    }
    
    file.close();
}

void generateMatrix(double**& matrix, int row, int col) {
    getSpace(matrix, row, col);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            matrix[i][j] = (i + j)%100 + 3*(i+1)%1000 + 4*(j+1)%1000;
        }
    }
}

void MutrMult_omp(double** a, double** b, double**& c, int m, int n, int l) {
    getSpace(c, m, l);

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < l; j++) {
            double sum = 0;
            for(int k = 0; k < n; k++){
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    int m = 0, n = 0, l = 0;
    double **A, **B, **C;
    double mulTime;
    string matrixA = "matrixA.txt", matrixB = "matrixB.txt", matrixC = "matrixC.txt";
    if(argc >= 4) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        l = atoi(argv[3]);

        if(m <= 0 || n <= 0 || l <= 0) {
            cout << "Invalid row or col values" << endl;
            exit(2);
        }

        generateMatrix(A, m, n);
        generateMatrix(B, n, l);
    }
    else {
        readMatrix(A, matrixA, m, n);
        readMatrix(B, matrixB, n, l);
    }
    mulTime = omp_get_wtime();
    MutrMult_omp(A, B, C, m, n, l);
    mulTime = omp_get_wtime() - mulTime;
    writeMatrix(C, matrixC, m, l);
    cout << "Time: " << mulTime << endl;
    clean(A, m);
    clean(B, n);
    clean(C, m);
    return 0;
}

