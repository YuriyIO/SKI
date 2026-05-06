#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <cmath>

using namespace std;

#define MAX_ELEMS_IN_ROWS 5

void printZero(const char* str, int rank) {
    if(rank != 0) 
        return;
    printf("%s\n", str);
}

class StartData {
public:
    int debugFlag;
    int Nx, Ny, Px, Py, K1, K2;
    int maxit;
    double eps;
    int rank, size;
    int bi, ei, bj, ej;
    int up, down, right, left;

    int numberOfElems;

    StartData(const int rank,const int size, int argc, char **argv) {
        this->rank = rank;
        this->size = size;
        up = down = right = left = -1;

        init(argc, argv);
        setIJ();
        setNeighbors();
        int block = K1 + K2;
        numberOfElems = ((Ny*Nx)/block) * (K1 + 2*K2)
               + min((Ny*Nx)%block, K1)
               + max(0, (Ny*Nx)%block - K1) * 2;
    }

    void init(int argc, char **argv) {
        if(argc < 7) {
            printZero(
                "Введите не менее 6 параметров:\n"
                "Nx, Ny - число клеток в решетке по вертикали и горизонтали\n"
                "Px, Py - количество процессов по горизонтали и вертикали\n"
                "K1, K2 - число однопалубных и двупалубных клеток\n"
                "Введите Y, если нужна отладочная печать\n"
                "Ввод параметров осуществляется в показанном порядке",
                rank
            );
            MPI_Finalize();
            exit(0);
        }
    
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Px = atoi(argv[3]);
        Py = atoi(argv[4]);
        K1 = atoi(argv[5]);
        K2 = atoi(argv[6]);
        maxit = 1000;
        eps = 0.001;

        debugFlag = 0;
        if (argc > 7 && strcmp(argv[7], "Y") == 0) {
            debugFlag = 1;
        }   
        
        if(Nx <= 0) {
            printZero("Параметр Nx должен быть положительным", rank);
            MPI_Finalize();
            exit(1);
        }
        if(Ny <= 0) {
            printZero("Параметр Ny должен быть положительным", rank);
            MPI_Finalize();
            exit(1);
        }
        
        if(Px * Py != size){
            printZero("Ошибка: Px * Py должно равняться общему количеству процессов", rank);
            MPI_Finalize();
            exit(1);
        }
        if(Px <= 0) {
            printZero("Параметр Px должен быть положительным", rank);
            MPI_Finalize();
            exit(1);
        }
        if(Py <= 0) {
            printZero("Параметр Py должен быть положительным", rank);
            MPI_Finalize();
            exit(1);
        }

        if(K1 < 0) {
            printZero("Параметр K1 должен быть неотрицательным", rank);
            MPI_Finalize();
            exit(1);
        }
        if(K2 < 0) {
            printZero("Параметр K2 должен быть неотрицательным", rank);
            MPI_Finalize();
            exit(1);
        }
        if(K1 == 0 && K2 == 0) {
            printZero("Параметры K1 и K2 не могут быть одновременно равны 0", rank);
            MPI_Finalize();
            exit(1);
        }
    }

    void setIJ() {
        int baseX = Nx / Px;
        int baseY = Ny / Py;
        int remainX = Nx % Px;
        int remainY = Ny % Py;
        int rowNum = rank / Px;
        int colNum = rank %  Px;

        bi = baseY * rowNum + (rowNum < remainY ? rowNum : remainY);
        ei = bi + baseY + (rowNum < remainY ? 1 : 0) - 1;

        bj = baseX * colNum + (colNum < remainX ? colNum : remainX);
        ej = bj + baseX + (colNum < remainX ? 1 : 0) - 1;
    }

    void setNeighbors() {
        int colNum = rank % Px;
        int rowNum = rank / Px;
        
        if(rowNum > 0) {
            up = rank - Px;
        }
        if(rowNum < (Py - 1)) {
            down = rank + Px;
        }
        if(colNum > 0) {
            left = rank - 1;
        }
        if(colNum < (Px - 1)) {
            right = rank + 1;
        }
    }

    void print() const{
        if (rank == 0) {
            printf("Nx = %d, Ny = %d\n", Nx, Ny);
            printf("Px = %d, Py = %d\n", Px, Py);
            printf("K1 = %d, K2 = %d\n", K1, K2);
            printf("Debug flag = %d\n\n", debugFlag);
        }
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        for (int proc = 0; proc < size; proc++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == proc) {
                printf("proc: %d i: %d, %d j:%d, %d\n", rank, bi, ei, bj, ej);
                printf("proc: %d up: %d, down: %d, left:%d, right %d\n", rank, up, down, left, right);
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        printZero("\n", rank);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

struct MatrixElem{
    int upGlobal, downGlobal;
    int upLocal, downLocal;
};

class StartMatrix {
public:
    struct MatrixElem* matrix;
    int localRow, globalRow;
    int localCol, globalCol;
    int numberOfInnerElems;
    int numberOfElems;

    int leftStartLocalNums, rightStartLocalNums, downStartLocalNums;
    
    StartMatrix(const StartData &data){
        localRow = data.ei - data.bi + 1 + 2;
        localCol = data.ej - data.bj + 1 + 2;
        globalRow = data.bi;
        globalCol = data.bj;

        matrix = new MatrixElem[localRow*localCol];
        
        fillDistributedMatrix(data);
    }

    ~StartMatrix() {
        delete[] matrix;
    }

    void fillDistributedMatrix(const StartData &startData) {
        int block = startData.K1 + startData.K2;    
        if (startData.K1 == 0 || startData.K2 == 0) {
            int k = (startData.K2 == 0 ? 1 : 2);
            
            for (int i = 1; i < localRow - 1; i++) {
                for (int j = 1; j < localCol - 1; j++) {
                    int iGlobal = globalRow + i - 1;
                    int jGlobal = globalCol + j - 1;
                    int numGlobal = iGlobal * startData.Nx + jGlobal;
                    
                    matrix[i*localCol+j].upGlobal = numGlobal;
                    matrix[i*localCol+j].downGlobal = (k == 2 ? numGlobal + 1 : 0);
                }
            }
        } else {
            for (int i = 1; i < localRow - 1; i++) {
                for (int j = 1; j < localCol - 1; j++) {
                    int iGlobal = globalRow + i - 1;
                    int jGlobal = globalCol + j - 1;
                    int numGlobal = iGlobal * startData.Nx + jGlobal;
                    int globalFullBlocks = numGlobal / block;
                    int globalOffset = numGlobal % block;
                    int counterGlobal = globalFullBlocks * (startData.K1 + 2 * startData.K2) +
                                    (globalOffset < startData.K1 ? globalOffset : startData.K1 + 2 * (globalOffset - startData.K1));

                    if (globalOffset < startData.K1) {
                        matrix[i*localCol+j].upGlobal = counterGlobal;
                        matrix[i*localCol+j].downGlobal = 0;
                    } else {
                        matrix[i*localCol+j].upGlobal = counterGlobal;
                        matrix[i*localCol+j].downGlobal = counterGlobal + 1;
                    }

                }
            }
        }

        fillNeighbors(startData);
        initLocalIndexes(startData);
    }

    void fillNeighbors(const StartData &data) {
        int innerRows = data.ei - data.bi + 1;
        int innerCols = data.ej - data.bj + 1;
        
        auto fillElement = [&](const int i, const int j, const int iGlobal, const int jGlobal) {
            const int numGlobal = iGlobal * data.Nx + jGlobal;
            
            if (data.K1 == 0 || data.K2 == 0) {
                int k = (data.K2 == 0 ? 1 : 2);
                matrix[i*localCol+j].upGlobal = numGlobal;
                matrix[i*localCol+j].downGlobal = (k == 2 ? numGlobal + 1 : 0);
            } else {
                int globalFullBlocks = numGlobal / (data.K1 + data.K2);
                int globalOffset = numGlobal % (data.K1 + data.K2);
                int counterGlobal = globalFullBlocks * (data.K1 + 2 * data.K2) +
                                (globalOffset < data.K1 ? globalOffset : data.K1 + 2 * (globalOffset - data.K1));

                if (globalOffset < data.K1) {
                    matrix[i*localCol+j].upGlobal = counterGlobal;
                    matrix[i*localCol+j].downGlobal = 0;
                } else {
                    matrix[i*localCol+j].upGlobal = counterGlobal;
                    matrix[i*localCol+j].downGlobal = counterGlobal + 1;
                }
            }
        };

        if(data.up != -1) {
            for (int j = 1; j < localCol - 1; j++) {
                fillElement(0, j, globalRow - 1, globalCol + j - 1);
            }
        } else {
            for (int j = 0; j < localCol; j++) {
                matrix[j].upGlobal = -1;
                matrix[j].downGlobal = 0;
            }
        }
        
        if(data.down != -1) {
            for (int j = 1; j < localCol - 1; j++) {
                fillElement(localRow-1, j, globalRow + innerRows, globalCol + j - 1);
            }
        } else {
            for (int j = 1; j < localCol-1; j++) {
                matrix[(localRow-1)*localCol+j].upGlobal = -1;
                matrix[(localRow-1)*localCol+j].downGlobal = 0;
            }
        }
        
        if(data.left != -1) {
            for (int i = 1; i < localRow - 1; i++) {
                fillElement(i, 0, globalRow + i - 1, globalCol - 1);
            }
        } else {
            for (int i = 1; i < localRow-1; i++) {
                matrix[i*localCol].upGlobal = -1;
                matrix[i*localCol].downGlobal = 0;
            }
        }
        
        if(data.right != -1) {
            for (int i = 1; i < localRow - 1; i++) {
                fillElement(i, localCol-1, globalRow + i - 1, globalCol + innerCols);
            }
        } else {
            for (int i = 1; i < localRow-1; i++) {
                matrix[i*localCol+localCol-1].upGlobal = -1;
                matrix[i*localCol+localCol-1].downGlobal = 0;
            }
        }
        
        matrix[0] = {-1, 0, -1, 0};
        matrix[localCol-1] = {-1, 0, -1, 0};
        matrix[(localRow-1)*localCol] = {-1, 0, -1, 0};
        matrix[(localRow-1)*localCol+localCol-1] = {-1, 0, -1, 0};
    }

    void initLocalIndexes(const StartData &startData) {
        int localNum = 0;
        for(int i = 1; i < localRow-1; i++) {
            for(int j = 1; j < localCol-1; j++) {
                MatrixElem& cell = matrix[i*localCol+j];
                cell.upLocal = localNum;
                localNum++;
                if(cell.downGlobal != 0) {
                    cell.downLocal = localNum;
                    localNum++;
                }
                else {
                    cell.downLocal = 0;
                }
            }
        }
        numberOfInnerElems = localNum;

        if(startData.up != -1) {
            for(int j = 1; j < localCol-1; j++) {
                MatrixElem& cell = matrix[0*localCol+j];
                cell.upLocal = localNum;
                localNum++;
                if(cell.downGlobal != 0) {
                    cell.downLocal = localNum;
                    localNum++;
                }
                else {
                    cell.downLocal = 0;
                }
            }
        } 
        else {
            for(int j = 1; j < localCol-1; j++) {
                MatrixElem& cell = matrix[0*localCol+j];
                cell.upLocal = -1;
                cell.downLocal = 0;
            }
        } 

        leftStartLocalNums = localNum;
        if(startData.left != -1) {
            for(int i = 1; i < localRow-1; i++) {
                MatrixElem& cell = matrix[i*localCol];
                cell.upLocal = localNum;
                localNum++;
                if(cell.downGlobal != 0) {
                    cell.downLocal = localNum;
                    localNum++;
                }
                else {
                    cell.downLocal = 0;
                }
            }
        }
        else {
            for(int i = 1; i < localRow-1; i++) {
                MatrixElem& cell = matrix[i*localCol];
                cell.upLocal = -1;
                cell.downLocal = 0;
            }
        }

        rightStartLocalNums = localNum;
        if(startData.right != -1) {
            for(int i = 1; i < localRow-1; i++) {
                MatrixElem& cell = matrix[i*localCol+localCol-1];
                cell.upLocal = localNum;
                localNum++;
                if(cell.downGlobal != 0) {
                    cell.downLocal = localNum;
                    localNum++;
                }
                else {
                    cell.downLocal = 0;
                }
            }
        }
        else {
            for(int i = 1; i < localRow-1; i++) {
                MatrixElem& cell = matrix[i*localCol+localCol-1];
                cell.upLocal = -1;
                cell.downLocal = 0;
            }
        }

        downStartLocalNums = localNum;
        if(startData.down != -1) {
            for(int j = 1; j < localCol-1; j++) {
                MatrixElem& cell = matrix[(localRow-1)*localCol+j];
                cell.upLocal = localNum;
                localNum++;
                if(cell.downGlobal != 0) {
                    cell.downLocal = localNum;
                    localNum++;
                }
                else {
                    cell.downLocal = 0;
                }
            }
        }
        else {
            for(int j = 1; j < localCol-1; j++) {
                MatrixElem& cell = matrix[(localRow-1)*localCol+j];
                cell.upLocal = -1;
                cell.downLocal = 0;
            }
        }

        numberOfElems = localNum;
    }

    void print(const StartData &data) const{
        for (int proc = 0; proc < data.size; proc++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (data.rank == proc) {
                printf("=== Process %d ===\n", data.rank);
                for (int i = 1; i < localRow - 1; i++) {
                    for (int j = 1; j < localCol - 1; j++) {
                        struct MatrixElem* elem = &matrix[i*localCol+j];
                        printf("%d", elem->upGlobal);
                        if(elem->downGlobal != 0) {
                            printf("/%d", elem->downGlobal);
                        }
                        printf(" ");
                    }
                    printf("\n");
                }
                printf("\n");
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void printWithNeighbors(const StartData & data) const{
        for (int proc = 0; proc < data.size; proc++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (data.rank == proc) {
                printf("=== Process %d Global Indexes ===\n", data.rank);
                for (int i = 0; i < localRow; i++) {
                    for (int j = 0; j < localCol; j++) {
                        struct MatrixElem* elem = &matrix[i*localCol+j];
                        if (i == 0 || i == localRow-1 || j == 0 || j == localCol-1) {
                            printf("\x1b[33m");
                        }
                        printf("%d", elem->upGlobal);
                        if(elem->upGlobal != -1 && elem->downGlobal != 0) {
                            printf("/%d", elem->downGlobal);
                        }
                        if (i == 0 || i == localRow-1 || j == 0 || j == localCol-1) {
                            printf("\x1b[0m");
                        }
                        printf("\t");
                    }
                    printf("\n");
                }
                printf("\n");
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void printLocalWithNeighbors(const StartData & data) const{
        for (int proc = 0; proc < data.size; proc++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (data.rank == proc) {
                printf("=== Process %d Local Indexes ===\n", data.rank);
                for (int i = 0; i < localRow; i++) {
                    for (int j = 0; j < localCol; j++) {
                        struct MatrixElem* elem = &matrix[i*localCol+j];
                        if (i == 0 || i == localRow-1 || j == 0 || j == localCol-1) {
                            printf("\x1b[33m");
                        }
                        printf("%d", elem->upLocal);
                        if(elem->upLocal != -1 &&  elem->downLocal != 0) {
                            printf("/%d", elem->downLocal);
                        }
                        if (i == 0 || i == localRow-1 || j == 0 || j == localCol-1) {
                            printf("\x1b[0m");
                        }
                        printf("\t");
                    }
                    printf("\n");
                }
                printf("\n");
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
};

class ELLPACK {
public:
    double* A;
    int* JA;
    int numberOfElems;
    int emptyElem;
    int numberOfInnerElems;
    
    int* localToGlobal;

    ELLPACK(const StartData &startData, const StartMatrix &startMatrix) {
        A = nullptr;
        emptyElem = startMatrix.numberOfElems;
        numberOfElems = startMatrix.numberOfElems;
        numberOfInnerElems = startMatrix.numberOfInnerElems;

        A = new double[numberOfInnerElems*MAX_ELEMS_IN_ROWS];
        JA = new int[numberOfInnerElems*MAX_ELEMS_IN_ROWS];
        localToGlobal = new int[numberOfElems];

        fillLocalToGlobal(startMatrix); 
        
        fillJA(startMatrix);
    }
    
    ~ELLPACK() {
        delete[] A;
        delete[] JA;
        delete[] localToGlobal;
    }

    void fillJA(const StartMatrix &startMatrix) {
        int* currentPos = new int[numberOfInnerElems]();
        /*главная диагональ*/

        for(int i = 0; i < numberOfInnerElems; i++) {
            addDiagElem(i, i, currentPos);
        }

        for(int i = 0; i < startMatrix.localRow-1; i++) {
            for(int j = 0; j < startMatrix.localCol-1; j++) {
                MatrixElem& cell = startMatrix.matrix[i*startMatrix.localCol+j];

                if(cell.upGlobal == -1) 
                    continue;
                
                /*между элементами*/
                if (cell.downGlobal != 0) {
                    addPair(cell.upLocal, cell.downLocal, currentPos, startMatrix);
                }
                
                /*правый сосед*/
                if (j + 1 < startMatrix.localCol) {
                    MatrixElem& right = startMatrix.matrix[i*startMatrix.localCol+j+1];
                    if(right.upGlobal != -1) {
                        int from = (cell.downLocal > 0 ? cell.downLocal : cell.upLocal);
                        int to = right.upLocal;
                        addPair(from, to, currentPos, startMatrix);
                    }
                }
                
                /*нижний сосед*/
                if (i + 1 < startMatrix.localRow) {
                    MatrixElem& bottom = startMatrix.matrix[(i+1)*startMatrix.localCol+j];
                    if(bottom.upGlobal != -1) {
                        int from = (cell.downLocal > 0 ? cell.downLocal : cell.upLocal);
                        int to = bottom.upLocal;
                        addPair(from, to, currentPos, startMatrix);
                    }
                }
            }
        }
        
        for(int i = 0; i < numberOfInnerElems; i++) {
            int elem = JA[i*MAX_ELEMS_IN_ROWS+currentPos[i]-1];
            while(currentPos[i] < MAX_ELEMS_IN_ROWS) {
                JA[i*MAX_ELEMS_IN_ROWS+currentPos[i]] = elem;
                currentPos[i]++;
            }
        }

        delete[] currentPos;
    }

    void addPair(int row, int col, int* currentPos, const StartMatrix &startMatrix) {
        addElem(row, col, currentPos);
        addElem(col, row, currentPos);
    }

    void addElem(int row, int col, int* currentPos) {
        if(row >= numberOfInnerElems)
            return;
        JA[row*MAX_ELEMS_IN_ROWS+currentPos[row]] = col;
        currentPos[row]++;
    }
    
    void addDiagElem(int row, int col, int* currentPos) {
        JA[row*MAX_ELEMS_IN_ROWS+currentPos[row]] = col;
        currentPos[row]++;
    }

    void fillLocalToGlobal(const StartMatrix &startMatrix) {
        for(int i = 0; i < startMatrix.localRow; i++) {
            for(int j = 0; j < startMatrix.localCol; j++) {
                MatrixElem& cell = startMatrix.matrix[i*startMatrix.localCol+j];
                if(cell.upGlobal == -1) 
                    continue;
                localToGlobal[cell.upLocal] = cell.upGlobal;
                if(cell.downGlobal != 0) {
                    localToGlobal[cell.downLocal] = cell.downGlobal;
                }
            }
        }
    }

    void fill(StartMatrix &startMatrix) {
        for(int i = 0; i < numberOfInnerElems; i++) {
            double sum = 0.0;
            int ind;
            int globalRow = localToGlobal[i], globalCol = -1, col = -1;
            for(int j = 1; j < MAX_ELEMS_IN_ROWS; j++) {
                if(JA[i*MAX_ELEMS_IN_ROWS+j] == col) {
                    A[i*MAX_ELEMS_IN_ROWS+j] = 0;
                    continue;
                }
                col = JA[i*MAX_ELEMS_IN_ROWS+j];
                A[i*MAX_ELEMS_IN_ROWS+j] = cos(globalRow * localToGlobal[col] + globalRow + localToGlobal[col]);
                sum += fabs(A[i*MAX_ELEMS_IN_ROWS+j]);
            }
            A[i*MAX_ELEMS_IN_ROWS] = 1.5 * sum;
        }
    }

    void printJA(const StartData &data) const{
        for (int proc = 0; proc < data.size; proc++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (data.rank == proc) {
                printf("=== Process %d JA ===\n", data.rank);
                for(int i = 0; i < numberOfInnerElems; i++) {
                    printf("%2d: [", i);
                    for(int k = 0; k < MAX_ELEMS_IN_ROWS; k++) {
                        printf("%2d", JA[i*MAX_ELEMS_IN_ROWS+k]);
                        if(k < MAX_ELEMS_IN_ROWS - 1) printf(" ");
                    }
                    printf("]\n");
                }
                printf("\n");
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void printLocalToGlobal(const StartData &data) const{
        for (int proc = 0; proc < data.size; proc++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (data.rank == proc) {
                printf("=== Process %d LocalToGlobal ===\n", data.rank);
                printf("Local:  ");
                for(int i = 0; i < numberOfElems; i++) {
                    printf("%3d ", i);
                }
                printf("\nGlobal: ");
                for(int i = 0; i < numberOfElems; i++) {
                    printf("%3d ", localToGlobal[i]);
                }
                printf("\n\n");
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    void printA(const StartData &data) const{
        for (int proc = 0; proc < data.size; proc++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (data.rank == proc) {
                printf("=== Process %d A Matrix ===\n", data.rank);
                for(int i = 0; i < numberOfInnerElems; i++) {
                    for(int k = 0; k < MAX_ELEMS_IN_ROWS; k++) {
                        printf("%9.6f ", A[i*MAX_ELEMS_IN_ROWS+k]);
                    }
                    printf("\n");
                }
                printf("\n");
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
};

int max(int a, int b) {
    return a > b ? a : b;
}

class Comm {
public:
    int* neighbourBoundary[4]; /*up, left, right, down*/
    int neighbourBoundarySize[4]; /*up, left, right, down*/
    int* myBoundary[4]; 
    int myBoundarySize[4];
    int neighbors[4]; /*up, left, right, down*/


    Comm(const StartData &startData, const StartMatrix &startMatrix) {
        neighbors[0] = startData.up;
        neighbors[1] = startData.left;
        neighbors[2] = startData.right;
        neighbors[3] = startData.down;

        neighbourBoundarySize[0] = (neighbors[0] != -1) ? (startMatrix.leftStartLocalNums - startMatrix.numberOfInnerElems) : 0;
        neighbourBoundarySize[1] = (neighbors[1] != -1) ? (startMatrix.rightStartLocalNums - startMatrix.leftStartLocalNums) : 0;
        neighbourBoundarySize[2] = (neighbors[2] != -1) ? (startMatrix.downStartLocalNums - startMatrix.rightStartLocalNums) : 0;
        neighbourBoundarySize[3] = (neighbors[3] != -1) ? (startMatrix.numberOfElems - startMatrix.downStartLocalNums) : 0;

        int maxElems = 2*max(startMatrix.localCol-2, startMatrix.localRow-2);
        for (int i = 0; i < 4; i++) {
            neighbourBoundary[i] = nullptr;
            myBoundary[i] = nullptr;
            myBoundarySize[i] = 0;
            if (neighbourBoundarySize[i] > 0) {
                neighbourBoundary[i] = new int[neighbourBoundarySize[i]];
                myBoundary[i] = new int[maxElems];
                initNeighbourBoundary(i, startMatrix);
                initMyBoundary(i, startMatrix);
            }
        }
    }

    ~Comm() {
        for (int i = 0; i < 4; i++) {
            delete[] neighbourBoundary[i];
            delete[] myBoundary[i];
        }
    }

    void initMyBoundary(int boundary, const StartMatrix &startMatrix) {
        switch (boundary) {
            case 0:
            {
                MatrixElem& lastUpCell = startMatrix.matrix[1*startMatrix.localCol+startMatrix.localCol-2];
                myBoundarySize[boundary] = lastUpCell.upLocal+1;
                if(lastUpCell.downLocal > 0) {
                    myBoundarySize[boundary] ++;
                }
                for(int i = 0; i < myBoundarySize[boundary]; i++) {
                    myBoundary[0][i] = i;
                }
            }
                break;
            case 1:
                for (int i = 1; i < startMatrix.localRow - 1; i++) {
                    MatrixElem& cell = startMatrix.matrix[i*startMatrix.localCol + 1];
                    myBoundary[1][myBoundarySize[boundary]++] = cell.upLocal;
                    if(cell.downLocal != 0) {
                        myBoundary[1][myBoundarySize[boundary]++] = cell.downLocal;
                    }
                }
                break;
            case 2:
                for (int i = 1; i < startMatrix.localRow - 1; i++) {
                    MatrixElem& cell = startMatrix.matrix[i*startMatrix.localCol+startMatrix.localCol-2];
                    myBoundary[2][myBoundarySize[boundary]++] = cell.upLocal;
                    if(cell.downLocal != 0) {
                        myBoundary[2][myBoundarySize[boundary]++] = cell.downLocal;
                    }
                }
                break;
            case 3:
            {
                int start = startMatrix.matrix[(startMatrix.localRow-2)*startMatrix.localCol+1].upLocal;
                MatrixElem& lastDownCell = startMatrix.matrix[(startMatrix.localRow-2)*startMatrix.localCol+startMatrix.localCol-2];
                myBoundarySize[boundary] = lastDownCell.upLocal+1 - start;
                if(lastDownCell.downLocal > 0) {
                    myBoundarySize[boundary] ++;
                }
                for(int i = 0; i < myBoundarySize[boundary]; i++) {
                    myBoundary[3][i] = start + i;
                }
            }
                break;
        }
    }

    void initNeighbourBoundary(int boundaryType, const StartMatrix &startMatrix) {
        switch (boundaryType) {
            case 0: /*up*/ 
                for(int i = startMatrix.numberOfInnerElems; i < startMatrix.leftStartLocalNums; i++) {
                    neighbourBoundary[0][i-startMatrix.numberOfInnerElems] = i;
                }
                break;
            case 1: /*left*/ 
                for(int i = startMatrix.leftStartLocalNums; i < startMatrix.rightStartLocalNums; i++) {
                    neighbourBoundary[1][i-startMatrix.leftStartLocalNums] = i;
                }
                break;
            case 2: /*right*/ 
                for(int i = startMatrix.rightStartLocalNums; i < startMatrix.downStartLocalNums; i++) {
                    neighbourBoundary[2][i-startMatrix.rightStartLocalNums] = i;
                }
                break;
            case 3: /*down*/ 
                for(int i = startMatrix.downStartLocalNums; i < startMatrix.numberOfElems; i++) {
                    neighbourBoundary[3][i-startMatrix.downStartLocalNums] = i;
                }
                break;
        }
    }

    void exchangeBoundaryData(double* pkVek, const int numberOfInnerElems) {
        double* sendBuffers[4];
        double* recvBuffers[4];
        MPI_Request requests[8];
        int requestCount = 0;

        for (int i = 0; i < 4; i++) {
            if (neighbors[i] != -1) {
                sendBuffers[i] = new double[myBoundarySize[i]];
                recvBuffers[i] = new double[neighbourBoundarySize[i]];

                for (int j = 0; j < myBoundarySize[i]; j++) {
                    sendBuffers[i][j] = pkVek[myBoundary[i][j]];
                }

                MPI_Isend(sendBuffers[i], myBoundarySize[i], MPI_DOUBLE, neighbors[i], 0, MPI_COMM_WORLD, &requests[requestCount++]);
                MPI_Irecv(recvBuffers[i], neighbourBoundarySize[i], MPI_DOUBLE, neighbors[i], 0, MPI_COMM_WORLD, &requests[requestCount++]);
            }
        }

        MPI_Waitall(requestCount, requests, MPI_STATUSES_IGNORE);

        int ghostOffset = numberOfInnerElems;
        for (int i = 0; i < 4; i++) {
            if (neighbors[i] != -1) {
                for (int j = 0; j < neighbourBoundarySize[i]; j++) {
                    pkVek[ghostOffset++] = recvBuffers[i][j];
                }
                delete[] sendBuffers[i];
                delete[] recvBuffers[i];
            }
        }
    }

    void print(const StartData &startData) const {
        for (int proc = 0; proc < startData.size; proc++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (startData.rank == proc) {
                printf("=== Process %d COMM ===\n", startData.rank);
                printf("Neighbors: up=%d, left=%d, right=%d, down=%d\n", 
                       neighbors[0], neighbors[1], neighbors[2], neighbors[3]);
                printf("neighbourBoundarySize: up=%d, left=%d, right=%d, down=%d\n", 
                       neighbourBoundarySize[0], neighbourBoundarySize[1], neighbourBoundarySize[2], neighbourBoundarySize[3]);
                       printf("myBoundarySize: up=%d, left=%d, right=%d, down=%d\n\n", 
                       myBoundarySize[0], myBoundarySize[1], myBoundarySize[2], myBoundarySize[3]);
                fflush(stdout);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
};

int fillB(double* b, const StartMatrix &startMatrix, const int size) {
    int len = 0;
    int indexM = 0, indexB = 0;
    for (int i = 1; i < startMatrix.localRow - 1; i++) {
        for (int j = 1; j < startMatrix.localCol - 1; j++) {
            MatrixElem& cell = startMatrix.matrix[i*startMatrix.localCol+j];
            b[indexB] = sin(cell.upGlobal);
            indexB++;
            if(cell.downGlobal != 0) {
                b[indexB] = sin(cell.downGlobal);
                indexB++;
            }
        }
    }
    for(int i = indexB; i < size; i++) {
        b[i] = 0;
    }
    return indexB;
}

void printLocalSize(const double* b, const StartData &startData, const ELLPACK & ell) {
    int indB = 0, indGL = 0;
    while(indGL < startData.numberOfElems) {
        if(indB < ell.numberOfInnerElems && indGL == ell.localToGlobal[indB]) {
            cout << indGL << ": " << b[indB] << endl;
            fflush(stdout);
            indB++;
        }
        indGL++;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(startData.rank == 0)
        cout << endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

void spmv(double *qk, const ELLPACK &ell, const int size, const double *pk) {
    for(int i = 0; i < size; i++) {
        double sum = 0;
        for(int k = 0; k < MAX_ELEMS_IN_ROWS; k++) {
            sum += ell.A[i*MAX_ELEMS_IN_ROWS+k] * pk[ell.JA[i*MAX_ELEMS_IN_ROWS+k]];
        }
        qk[i] = sum;
    }
}

void spmvDiag(double *zk, const ELLPACK& ell, const double *rPred, const int size) {
    for (int i = 0; i < size; i++) {
        zk[i] = rPred[i]/ell.A[i*MAX_ELEMS_IN_ROWS];
    }
}

double dotVec(const double* x, const double* y, const int size) {
    double res = 0;
    for (int i = 0; i < size; i++)  
        res += x[i]*y[i];
    return res;
}

void axpy(double* pkVec, const double* zk, const double bk , const double* pkVecPred, const int size) {
    for (int i = 0; i < size; i++) 
        pkVec[i] = zk[i] + bk*pkVecPred[i];
}

void copyVec(double* to, const double* from, const int size) {
    for (int i = 0; i < size; i++) 
        to[i] = from[i];
}

void fillConstVal(double* x, const double val, const int size) {
    for (int i = 0; i < size; i++) 
        x[i] = val;
}

void solve(const StartData &startData, const ELLPACK& ell, Comm& comm, const int numberOfInnerElems, const int numberOfElems, const double* b, double* x, int& n, double& res, double* timeOp) {
    double* rk = new double[numberOfInnerElems];
    double* zk = new double[numberOfInnerElems];
    double* pkVec = new double[numberOfElems];
    double* qk = new double[numberOfInnerElems];
    double timeStart;

    double pk = 0, pkPred = 0, ak = 0, bk = 0;

    fillConstVal(x, 0, numberOfInnerElems);
    copyVec(rk, b, numberOfInnerElems);
    n = 0;

    do {
        n++;

        timeStart = MPI_Wtime();
        spmvDiag(zk, ell, rk, numberOfInnerElems);
        timeOp[1] += MPI_Wtime() - timeStart;

        pkPred = pk;

        timeStart = MPI_Wtime();
        double pkLocal = dotVec(rk, zk, numberOfInnerElems);
        MPI_Allreduce(&pkLocal, &pk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        timeOp[2] += MPI_Wtime() - timeStart;

        if (n == 1) {
            copyVec(pkVec, zk, numberOfInnerElems);
        } else {
            bk = pk / pkPred;

            timeStart = MPI_Wtime();
            axpy(pkVec, zk, bk, pkVec, numberOfInnerElems);
            timeOp[3] += MPI_Wtime() - timeStart;
        }

        timeStart = MPI_Wtime();
        comm.exchangeBoundaryData(pkVec, numberOfInnerElems);
        spmv(qk, ell, numberOfInnerElems, pkVec);
        timeOp[0] += MPI_Wtime() - timeStart;

        timeStart = MPI_Wtime();
        double localDot = dotVec(pkVec, qk, numberOfInnerElems);
        double globalDot;
        MPI_Allreduce(&localDot, &globalDot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        timeOp[2] += MPI_Wtime() - timeStart;
        ak = pk / globalDot;

        timeStart = MPI_Wtime();
        axpy(x, x, ak, pkVec, numberOfInnerElems);
        axpy(rk, rk, -ak, qk, numberOfInnerElems);
        timeOp[3] += MPI_Wtime() - timeStart;
        
    } while (pk > startData.eps * startData.eps && n < startData.maxit);
    res = pk;

    delete[] rk;
    delete[] zk;
    delete[] qk;
    delete[] pkVec;
}

void report(const StartData &startData, const int n, const int size, const double res, const double* time, const double* timeOp) {
    if(startData.rank != 0) {
        return;
    }

    cout << endl;
    cout << "Parameter:" << endl;
    cout << "Epsilon: " << startData.eps << endl;
    cout << "Max iterations: " << startData.maxit << endl;
    cout << "Iterations done: " << n << endl;
    cout << "L2 norm: " << res << endl;
    cout << endl;
    
    cout << "Execution time:" << endl;
    cout << "Total time: " << time[3] - time[0] << endl;
    cout << "Generation: " << time[1] - time[0] << endl;
    cout << "Fill: " <<  time[2] - time[1] << endl;
    cout << "Solve: " << time[3] - time[2] << endl;
    cout << "SPMV: " << timeOp[0] << endl;
    cout << "SPMVDiag: " << timeOp[1] << endl;
    cout << "Dot: " << timeOp[2] << endl;
    cout << "AXPY: " << timeOp[3] << endl;
    cout << endl;
}

int main(int argc, char** argv) {

    int size;
    int rank;

    MPI_Init(&argc, &argv);               
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double time[4];
    time[0] = MPI_Wtime();

    /*START GENERATE*/

    StartData startData(rank, size, argc, argv);
    if(startData.debugFlag)
        startData.print();

    StartMatrix startMatrix(startData);
    if(startData.debugFlag) {
        startMatrix.printWithNeighbors(startData);
        startMatrix.printLocalWithNeighbors(startData);
    }

    ELLPACK ell(startData, startMatrix);
    if(startData.debugFlag) {
        ell.printJA(startData);
    }

    /*COMMUNICATION*/

    Comm comm(startData, startMatrix);
    if(startData.debugFlag) {
        comm.print(startData);
    }

    /*COMMUNICATION*/
    
    /*FINISH GENERATE*/
    time[1] = MPI_Wtime();

    /*START FILL*/

    double* b = new double[startMatrix.numberOfElems];
    ell.fill(startMatrix);
    int len = fillB(b, startMatrix, startMatrix.numberOfElems);
    if(startData.debugFlag) {
        ell.printA(startData);
        MPI_Barrier(MPI_COMM_WORLD);
        if(startData.rank == 0)
            cout << "b\nnum: val" << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        printLocalSize(b, startData, ell);
    }
    
    /*FINISH FILL*/
    time[2] = MPI_Wtime();

    /*START SOLVE*/

    double* x = new double[startMatrix.numberOfInnerElems];
    int n = 0;
    double res;
    double timeOp[4] = {0};

    solve(startData, ell, comm, startMatrix.numberOfInnerElems, startData.numberOfElems, b, x, n, res, timeOp);
    if(startData.debugFlag) {
        if(startData.rank == 0)
            cout << "x\nnum: val" << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        printLocalSize(x, startData, ell);
    }
    time[3] = MPI_Wtime();

    /*FINISH SOLVE*/

    report(startData, n, startMatrix.numberOfInnerElems, res, time, timeOp);

    delete[] b;
    delete[] x;

    MPI_Finalize();
    
    return 0;
}
