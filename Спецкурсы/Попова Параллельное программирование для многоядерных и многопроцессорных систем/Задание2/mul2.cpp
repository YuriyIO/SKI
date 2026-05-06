#include <mpi.h>
#include <iostream>
#include <cstdlib>

using namespace std;

void generateMatrix(const char* filename, int startRow, int endRow, int startCol, int endCol, int Ny, int rank)
{
    int numRow = endRow - startRow + 1;
    int numCol = endCol - startCol + 1;
    double* A = new double[numRow * numCol];
    for (int i = 0; i < numRow; i++)
    {
        for (int j = 0; j < numCol; j++)
        {
            A[i * numCol + j] = rank;
        }
    }

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_Datatype filetype;
    MPI_Type_vector(numRow, numCol, Ny, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);
    MPI_Offset offset = startRow * Ny + startCol;
    MPI_File_set_view(fh, offset * sizeof(double), MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, A, numRow * numCol, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_Type_free(&filetype);
    MPI_File_close(&fh);

    delete[] A;
}

void generateVector(const char* filename, int startIdx, int numElements, int rank)
{
    double* x = new double[numElements];
    for (int i = 0; i < numElements; i++)
    {
        x[i] = rank;
    }

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, startIdx * sizeof(double), MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, x, numElements, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    delete[] x;
}

void readMatrix(const char* fileA, double* A, int startRow, int endRow, int startCol, int endCol, int Ny)
{
    int numRow = endRow - startRow + 1;
    int numCol = endCol - startCol + 1;

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, fileA, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_Datatype filetype;
    MPI_Type_vector(numRow, numCol, Ny, MPI_DOUBLE, &filetype);
    MPI_Type_commit(&filetype);
    MPI_Offset offset = startRow * Ny + startCol;
    MPI_File_set_view(fh, offset * sizeof(double), MPI_DOUBLE, filetype, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, A, numRow * numCol, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_Type_free(&filetype);
    MPI_File_close(&fh);
}

void readVector(const char* fileX, double* x, int startIdx, int numElements)
{
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, fileX, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, startIdx * sizeof(double), MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, x, numElements, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

void setIJ(int Nx, int Ny, int Px, int Py, int rank, int& startRow, int& endRow, int& startCol, int& endCol)
{
    int rowNum = rank / Py;
    int colNum = rank % Py;

    int baseX = Nx / Px;
    int baseY = Ny / Py;
    int remX = Nx % Px;
    int remY = Ny % Py;

    startRow = baseX * rowNum + (rowNum < remX ? rowNum : remX);
    endRow = startRow + baseX + (rowNum < remX ? 1 : 0) - 1;

    startCol = baseY * colNum + (colNum < remY ? colNum : remY);
    endCol = startCol + baseY + (colNum < remY ? 1 : 0) - 1;
}

void matrixVectorMultiply(double* A, double* x, double* b, int numRow, int numCol)
{
    for (int i = 0; i < numRow; i++)
    {
        b[i] = 0.0;
        for (int j = 0; j < numCol; j++)
        {
            b[i] += A[i * numCol + j] * x[j];
        }
    }
}

void writeResult(const char* fileB, double* b, int startIdx, int numElements, MPI_Comm comm)
{
    MPI_File fh;
    MPI_File_open(comm, fileB, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, startIdx * sizeof(double), MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, b, numElements, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Px = atoi(argv[3]);
    int Py = atoi(argv[4]);

    const char* fileA = argv[5];
    const char* fileX = argv[6];
    const char* fileB = argv[7];

    int generate_data = (argc > 8) ? atoi(argv[8]) : 0;

    if (Px * Py != size)
    {
        if (rank == 0)
        {
            cout << "Px * Py != MPI size" << endl;
        }
        MPI_Finalize();
        return 0;
    }

    int startRow, endRow, startCol, endCol;
    setIJ(Nx, Ny, Px, Py, rank, startRow, endRow, startCol, endCol);

    int numRow = endRow - startRow + 1; 
    int numCol = endCol - startCol + 1;

    if (generate_data)
    {
        generateMatrix(fileA, startRow, endRow, startCol, endCol, Ny, rank);
        generateVector(fileX, startCol, numCol, rank);
    }

    double* A = new double[numRow * numCol];
    double* x = new double[numCol];
    double* b_partial = new double[numRow];
    double* b_final = new double[numRow];
    double timeOP[3];

    double start = MPI_Wtime();
    readMatrix(fileA, A, startRow, endRow, startCol, endCol, Ny);
    readVector(fileX, x, startCol, numCol);
    timeOP[0] = MPI_Wtime() - start;

    start = MPI_Wtime();
    matrixVectorMultiply(A, x, b_partial, numRow, numCol);
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank / Py, rank, &row_comm);
    MPI_Reduce(b_partial, b_final, numRow, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    MPI_Comm_free(&row_comm);
    timeOP[1] = MPI_Wtime() - start;

    start = MPI_Wtime();
    MPI_Comm col0_comm;
    int color = (rank % Py == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &col0_comm);
    if (color == 0)
    {
        writeResult(fileB, b_final, startRow, numRow, col0_comm);
        MPI_Comm_free(&col0_comm);
    }
    timeOP[2] = MPI_Wtime() - start;
    
    if (rank == 0) {
        cout << "Read time: "    << timeOP[0] << " sec" << endl;
        cout << "Compute time: " << timeOP[1] << " sec" << endl;
        cout << "Write time: "   << timeOP[2] << " sec" << endl;
    }

    delete[] A;
    delete[] x;
    delete[] b_partial;
    delete[] b_final;

    MPI_Finalize();
    return 0;
}
