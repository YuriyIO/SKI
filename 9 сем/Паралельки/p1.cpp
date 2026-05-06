#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <omp.h>
#include <cstring>
#include <cmath>

#define MAX_ELEMS_IN_ROWS 5

bool debugFlag;

class StartData {
public:
    int Nx, Ny, K1, K2, maxit;
    double eps;

    StartData(int argc, char **argv) {
        if(argc < 7) {
            std::cout << "Введите не менее 6 параметров:\n"
            "Nx, Ny - число клеток в решетке по вертикали и горизонтали\n"
            "K1, K2 - параметры для количества однопалубных и двухпалубных кораблей треугольных и четырехугольных элементов\n"
            "maxit - максимальное число итераций  метода сопряженных градиентов c диагональным предобуславливателем"
            "eps - критерий остановки (𝜀), которым определяется точность решения"
            "Введите Y, если нужна отладочная печать\n"
            "Ввод параметров осуществляется в показанном порядке"
            ""
            << std::endl;
            exit(1);
        }
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        K1 = atoi(argv[3]);
        K2 = atoi(argv[4]);
        maxit = atoi(argv[5]);
        eps = atof(argv[6]);

        debugFlag = false;
        if (argc > 7 && strcmp(argv[7], "Y") == 0) {
            debugFlag = true;
        }       

        if(Nx <= 0) {
            std::cout << "Параметр Nx должен быть положительным" << std::endl;
            exit(2);
        }
        if(Ny <= 0) {
            std::cout << "Параметр Ny должен быть положительным" << std::endl;
            exit(3);
        }

        if(K1 < 0) {
            std::cout << "Параметр K1 должен быть неотрицательным" << std::endl;
            exit(4);
        }
        if(K2 < 0) {
            std::cout << "Параметр K2 должен быть неотрицательным" << std::endl;
            exit(5);
        }
        if(K1 == 0 && K2 ==0) {
            std::cout << "Параметры K1 и K2 не могут быть одновременно равны 0" << std::endl;
            exit(6);
        }

        if(maxit < 0) {
            std::cout << "Параметр maxit должен быть неотрицательным" << std::endl;
             exit(7);
        }

        if(eps < 0) {
            std::cout << "Параметр eps должен быть неотрицательным" << std::endl;
            std::exit(8);
        }
    }
};

struct MatrixElem {
    int elem;
    int upElemNum;
    int downElemNum;
};

class StartMatrix {
public:
    int row, column, K1, K2;
    MatrixElem ** matrix;
    int numberOfElems;

    StartMatrix(const StartData &startData) {
        this->K1 = startData.K1;
        this->K2 = startData.K2;
        row = startData.Ny;
        column = startData.Nx;
        numberOfElems = 0;
        matrix = new MatrixElem*[row];
        for(int i = 0; i < row; i++) {
            matrix[i] = new MatrixElem[column];
        }

        fiilMatrix();
    }

    ~StartMatrix() {
        for(int i = 0; i < row; i++) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }

    void fiilMatrix() {
        int block = K1 + K2;

        if (K1 == 0 || K2 == 0) {
            int k = (K2 == 0 ? 1 : 2);

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < column; j++) {
                    int num = (i * column + j) * k;
                    matrix[i][j].elem = k;
                    matrix[i][j].upElemNum = num;
                    matrix[i][j].downElemNum = (k == 2 ? num + 1 : 0);
                }
            }
            numberOfElems = row * column * k;
            return;
        }

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                int num = i * column + j;
                int fullBlocks = num / block;
                int offset = num % block;

                int counter = fullBlocks * (K1 + 2 * K2)
                                + (offset < K1 ? offset : K1 + 2 * (offset - K1));

                if (offset < K1) {
                    matrix[i][j].elem = 1;
                    matrix[i][j].upElemNum = counter;
                    matrix[i][j].downElemNum = 0;
                } else {
                    matrix[i][j].elem = 2;
                    matrix[i][j].upElemNum = counter;
                    matrix[i][j].downElemNum = counter + 1;
                }
            }
        }

        numberOfElems = ((row*column)/block) * (K1 + 2*K2)
               + std::min((row*column)%block, K1)
               + std::max(0, (row*column)%block - K1) * 2;
    }

    void print() const {
        std::cout << "Matrix:" << std::endl;
        for(int i = 0; i < row; i++) {
            for(int j = 0; j < column; j++) {
                std::cout << matrix[i][j].upElemNum;
                if(matrix[i][j].downElemNum != 0) {
                    std::cout << "/" << matrix[i][j].downElemNum;
                }
                std::cout << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

};

class ELLPACK {
public:
    double* A;
    int* JA;
    int size;

    ELLPACK(const StartMatrix &startMatrix) {
        ELLPACK::A = nullptr;
        size = startMatrix.numberOfElems;

        A = new double[size*MAX_ELEMS_IN_ROWS];
        JA = new int[size*MAX_ELEMS_IN_ROWS];

        fillJA(startMatrix);
    }
    
    ~ELLPACK() {
        delete[] A;
        delete[] JA;
    }
    
    void fillJA(const StartMatrix &startMatrix) {
        int* currentPos = new int[size]();

        /*главная диагональ*/
        for(int i = 0; i < size; i++) {
            addElem(i, i, currentPos);
        }

        for(int i = 0; i < startMatrix.row; i++) {
            for(int j = 0; j < startMatrix.column; j++) {
                MatrixElem& cell = startMatrix.matrix[i][j];
                
                /*между элементами*/
                if (cell.elem == 2) {
                    addPair(cell.upElemNum, cell.downElemNum, currentPos);
                }
                
                /*правый сосед*/
                if (j + 1 < startMatrix.column) {
                    MatrixElem& right = startMatrix.matrix[i][j+1];
                    int from = (cell.downElemNum > 0 ? cell.downElemNum : cell.upElemNum);
                    int to = right.upElemNum;
                    addPair(from, to, currentPos);
                }
                
                /*нижний сосед*/
                if (i + 1 < startMatrix.row) {
                    MatrixElem& bottom = startMatrix.matrix[i+1][j];
                    int from = (cell.downElemNum > 0 ? cell.downElemNum : cell.upElemNum);
                    int to = bottom.upElemNum;
                    addPair(from, to, currentPos);
                }
            }
        }

        /*дополняем до конца последним элементом*/
        for(int i = 0; i < size; i++) {
            int elem = JA[i*MAX_ELEMS_IN_ROWS+currentPos[i]-1];
            while(currentPos[i] < MAX_ELEMS_IN_ROWS) {
                JA[i*MAX_ELEMS_IN_ROWS+currentPos[i]] = elem;
                currentPos[i]++;
            }
        }

        delete[] currentPos;
    }

    void addPair(int row, int col, int* currentPos) {
        addElem(row, col, currentPos);
        addElem(col, row, currentPos);
    }
    
    void addElem(int row, int col, int* currentPos) {
        JA[row*MAX_ELEMS_IN_ROWS+currentPos[row]] = col;
        currentPos[row]++;
    }

    void fill() {
        #pragma omp parallel for
        for(int i = 0; i < size; i++) {
            double sum = 0;
            int col = -1;
            for(int j = 1; j < MAX_ELEMS_IN_ROWS; j++) {
                if(JA[i*MAX_ELEMS_IN_ROWS+j] == col) {
                    A[i*MAX_ELEMS_IN_ROWS+j] = 0;
                    continue;
                }
                col = JA[i*MAX_ELEMS_IN_ROWS+j];
                A[i*MAX_ELEMS_IN_ROWS+j] = cos(i * col + i + col);
                sum += fabs(A[i*MAX_ELEMS_IN_ROWS+j]);
            }
            A[i*MAX_ELEMS_IN_ROWS] = 1.5 * sum;
        }
    }
    
    void print() const{
        std::cout << "ELLPACK Format:" << std::endl;
        
        std::cout << "A:" << std::endl;
        for(int i = 0; i < size; i++) {
            for(int k = 0; k < MAX_ELEMS_IN_ROWS; k++) {
                if(JA[i*MAX_ELEMS_IN_ROWS+k] != -1) {
                    std::cout << A[i*MAX_ELEMS_IN_ROWS+k] << " ";
                } else {
                    std::cout << "0 ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        
        std::cout << "JA:" << std::endl;
        for(int i = 0; i < size; i++) {
            for(int k = 0; k < MAX_ELEMS_IN_ROWS; k++) {
                std::cout << JA[i*MAX_ELEMS_IN_ROWS+k] << " ";
            }
            std::cout << std::endl;
        }
    }
};

void fillB(double* b, const int size) {
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        b[i] = sin(i);
    }
}

void varB1(int argc, char **argv);
ELLPACK generate(int argc, char **argv, int &maxit, double &eps);
void fill(ELLPACK &ell, double* b);
void solve(ELLPACK &ell, const double*b, const double eps, const int maxit, double* x, int& n, double& res, double* timeOp);
void report(const double eps, const int maxit, const int n, const int size, const double res, const double* time, const double* timeOp);

int main(int argc, char **argv) {
    varB1(argc, argv);
    return 0;
}


long long countSPMV, countDOT, countAXPY, countSPMVDiag;

void spmvCount(const ELLPACK &ell) {
    countSPMV += 2 * ell.size * MAX_ELEMS_IN_ROWS;
}

void spmvDiagCount(const int size) {
    countSPMVDiag = size;
}

void dotVecCount(const int size) {
    countDOT += 2 * size;
}

void axpyCount(const int size) {
    countAXPY += 2 * size;
}

void varB1(int argc, char **argv) {
    int maxit;
    double eps;
    double time[4];
    double timeOp[4] = {0};
    time[0] = omp_get_wtime();

    ELLPACK ell = generate(argc, argv, maxit, eps);
    time[1] = omp_get_wtime();

    double* b = new double[ell.size];
    fill(ell, b);
    time[2] = omp_get_wtime();

    double* x = new double[ell.size];
    int n = 0;
    double res = 0;
    solve(ell, b, eps, maxit, x, n, res, timeOp);
    time[3] = omp_get_wtime();

    countSPMV = countSPMVDiag = countDOT = countAXPY = 0;
    spmvCount(ell);
    spmvDiagCount(ell.size);
    dotVecCount(ell.size);
    axpyCount(ell.size);

    report(eps, maxit, n, ell.size, res, time, timeOp);
    delete[] b;
    delete[] x;
}

ELLPACK generate(int argc, char **argv, int &maxit, double &eps) {
    StartData startdata(argc, argv);
    maxit = startdata.maxit;
    eps = startdata.eps;
    StartMatrix startMatrix(startdata);
    if(debugFlag) {
        startMatrix.print();
    }
    ELLPACK ell(startMatrix);
    return ell;
}

void fill(ELLPACK &ell, double* b) {
    ell.fill();
    fillB(b, ell.size);
    if(debugFlag) {
        ell.print();
        std::cout << std::endl;

        std::cout << "b:" << std::endl;
        for(int i = 0; i < ell.size; i++) {
            std::cout << b[i] << " ";
        }
        std::cout << std::endl;
    }
}

void spmv(double *qk, const ELLPACK &ell, const double *pk, double* timeOp) {
    double start = omp_get_wtime();
    #pragma omp parallel for
    for(int i = 0; i < ell.size; i++) {
        double sum = 0;
        for(int k = 0; k < MAX_ELEMS_IN_ROWS; k++) {
            sum += ell.A[i*MAX_ELEMS_IN_ROWS+k] * pk[ell.JA[i*MAX_ELEMS_IN_ROWS+k]];
        }
        qk[i] = sum;
    }
    timeOp[0] += omp_get_wtime() - start;
}

void spmvDiag(double *zk, const ELLPACK& ell, const double *rPred, const int size, double* timeOp) {
    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        zk[i] = rPred[i]/ell.A[i*MAX_ELEMS_IN_ROWS];
    }
    timeOp[1] += omp_get_wtime() - start;
}

double dotVec(const double* x, const double* y, const int size, double* timeOp) {
    double start = omp_get_wtime();
    double res = 0;
    #pragma omp parallel for reduction(+:res)
    for (int i = 0; i < size; i++)  
        res += x[i]*y[i];
    timeOp[2] += omp_get_wtime() - start;
    return res;
}

void axpy(double* pkVec, const double* zk, const double bk , const double* pkVecPred, const int size, double* timeOp) {
    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < size; i++) 
        pkVec[i] = zk[i] + bk*pkVecPred[i];
    timeOp[3] += omp_get_wtime() - start;
}

void copyVec(double* to, const double* from, const int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) 
        to[i] = from[i];
}

void fillConstVal(double* x, const double val, const int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; i++) 
        x[i] = val;
}

void solve(ELLPACK& ell, const double* b, const double eps, const int maxit, double* x, int& n, double& res, double* timeOp) {
    int size = ell.size;
    double* rk = new double[size];
    double* zk = new double[size];
    double* pkVec = new double[size];
    double* qk = new double[size];

    double pk = 0, pkPred = 0, ak = 0, bk = 0;

    fillConstVal(x, 0, size);
    copyVec(rk, b, size);
    n = 0;

    do {
        n++;
        spmvDiag(zk, ell, rk, size, timeOp);
        pkPred = pk;
        pk = dotVec(rk, zk, size, timeOp);

        if (n == 1) {
            copyVec(pkVec, zk, size);
        } else {
            bk = pk / pkPred;
            axpy(pkVec, zk, bk, pkVec, size, timeOp);
        }

        spmv(qk, ell, pkVec, timeOp);
        ak = pk / dotVec(pkVec, qk, size, timeOp);
        axpy(x, x, ak, pkVec, size, timeOp);
        axpy(rk, rk, -ak, qk, size, timeOp);
    } while (pk > eps * eps && n < maxit);
    res = pk;

    if(debugFlag) {
        std::cout << std::endl;
        std::cout << "x: " << std::endl;
        for(int i = 0; i < size; i++) {
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
    }

    delete[] rk;
    delete[] zk;
    delete[] qk;
    delete[] pkVec;
}

void report(const double eps, const int maxit, const int n, const int size, const double res, const double* time, const double* timeOp) {    
    std::cout << std::endl;
    std::cout << "Parameter:" << std::endl;
    std::cout << "Epsilon: " << eps << std::endl;
    std::cout << "Max iterations: " << maxit << std::endl;
    std::cout << "Iterations done : " << n << std::endl;
    std::cout << "L2 norm : " << res << std::endl;
    std::cout << std::endl;
    
    std::cout << "Execution time:" << std::endl;
    std::cout << "Total time: " << time[3] - time[0] << std::endl;
    std::cout << "Generation: " << time[1] - time[0] << std::endl;
    std::cout << "Fill: " <<  time[2] - time[1] << std::endl;
    std::cout << "Solve: " << time[3] - time[2] << std::endl;
    std::cout << "SPMV: " << timeOp[0] << std::endl;
    std::cout << "SPMVDiag: " << timeOp[1] << std::endl;
    std::cout << "Dot: " << timeOp[2] << std::endl;
    std::cout << "AXPY: " << timeOp[3] << std::endl;
    std::cout << std::endl;

    std::cout << "Counters:" << std::endl;
    std::cout << "spmvCount: " << countSPMV << std::endl;
    std::cout << "spmvDiagCount: " << countSPMVDiag << std::endl;
    std::cout << "dotVecCount: " << countDOT << std::endl;
    std::cout << "axpyCount: " <<countAXPY << std::endl;
    std::cout << std::endl;
}