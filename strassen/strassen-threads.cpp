#include "include/threadpool.h"

#include <omp.h>

#include <cstdio>
#include <string>
#include <ctime>

using namespace std;

// threshold - пороговое значение размера матриц,
// после которого вызывается стандартный алгоритм
int threshold;

// печать матрицы
void showMatrix(double **matrix, int nRows, int nCols)
{
    if (matrix != NULL)
        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
                printf(" %8.3f", matrix[i][j]);

            printf("\n");
        }
    printf("\n");
}

template <typename T>
T **createMatrix(int nRows, int nCols)
{
    T **data = new T*[nRows];
    T *buffer = new T[nRows * nCols];

    for (int i = 0; i < nRows; i++)
    {
        data[i] = buffer;
        buffer += nCols;
    }

    return data;
}

template <typename T>
void deleteMatrix(T** matrix)
{
    delete[] * matrix;
    delete[] matrix;
}

// последовательное умножение матриц
// A [m x p], B [p x n] -> C[m x n]
void seqMatMult(int m, int n, int p, int** A, int** B, int** C)
{
    int i, j, k;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            C[i][j] = 0;

    for (i = 0; i < m; i++)
        for (k = 0; k < p; k++)
            for (j = 0; j < n; j++)
                C[i][j] += A[i][k] * B[k][j];
}

// копирует nRows строк матрицы Y начиная со строки m и столбца n
void copyQtrMatrix(int **X, int nRows, int **Y, int m, int n)
{
    for (int i = 0; i < nRows; i++)
        X[i] = &Y[m + i][n];
}

// сложение матриц
void addMatBlocks(int **T, int m, int n, int **X, int **Y)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            T[i][j] = X[i][j] + Y[i][j];
}

// вычитание матриц
void subMatBlocks(int **T, int m, int n, int **X, int **Y)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            T[i][j] = X[i][j] - Y[i][j];
}

void strassen_omp(int size, int **A, int **B, int **C)
{
    // если размер меньше порогового,
    // выполняем обычное последовательное умножение
    if (size <= threshold)
        seqMatMult(size, size, size, A, B, C);

    else
    {
        int n = size / 2;

        int **M1 = createMatrix<int>(n, n);
        int **M2 = createMatrix<int>(n, n);
        int **M3 = createMatrix<int>(n, n);
        int **M4 = createMatrix<int>(n, n);
        int **M5 = createMatrix<int>(n, n);
        int **M6 = createMatrix<int>(n, n);
        int **M7 = createMatrix<int>(n, n);

        int **AM1 = createMatrix<int>(n, n);
        int **BM1 = createMatrix<int>(n, n);
        int **AM2 = createMatrix<int>(n, n);
        int **BM3 = createMatrix<int>(n, n);
        int **BM4 = createMatrix<int>(n, n);
        int **AM5 = createMatrix<int>(n, n);
        int **AM6 = createMatrix<int>(n, n);
        int **BM6 = createMatrix<int>(n, n);
        int **AM7 = createMatrix<int>(n, n);
        int **BM7 = createMatrix<int>(n, n);

        int **A11 = new int*[n];
        int **A12 = new int*[n];
        int **A21 = new int*[n];
        int **A22 = new int*[n];

        int **B11 = new int*[n];
        int **B12 = new int*[n];
        int **B21 = new int*[n];
        int **B22 = new int*[n];

        int **C11 = new int*[n];
        int **C12 = new int*[n];
        int **C21 = new int*[n];
        int **C22 = new int*[n];

        copyQtrMatrix(A11, n, A, 0, 0);
        copyQtrMatrix(A12, n, A, 0, n);
        copyQtrMatrix(A21, n, A, n, 0);
        copyQtrMatrix(A22, n, A, n, n);

        copyQtrMatrix(B11, n, B, 0, 0);
        copyQtrMatrix(B12, n, B, 0, n);
        copyQtrMatrix(B21, n, B, n, 0);
        copyQtrMatrix(B22, n, B, n, n);

        copyQtrMatrix(C11, n, C, 0, 0);
        copyQtrMatrix(C12, n, C, 0, n);
        copyQtrMatrix(C21, n, C, n, 0);
        copyQtrMatrix(C22, n, C, n, n);

// все операции с подматрицами выполняются в отдельных потоках
#pragma omp task
        {
            // M1 = (A11 + A22) * (B11 + B22)
            addMatBlocks(AM1, n, n, A11, A22);
            addMatBlocks(BM1, n, n, B11, B22);
            strassen_omp(n, AM1, BM1, M1);
        }

#pragma omp task
        {
            // M2 = (A21 + A22) * B11
            addMatBlocks(AM2, n, n, A21, A22);
            strassen_omp(n, AM2, B11, M2);
        }

#pragma omp task
        {
            // M3 = A11 * (B12 - B22)
            subMatBlocks(BM3, n, n, B12, B22);
            strassen_omp(n, A11, BM3, M3);
        }

#pragma omp task
        {
            // M4 = A22 * (B21 - B11)
            subMatBlocks(BM4, n, n, B21, B11);
            strassen_omp(n, A22, BM4, M4);
        }

#pragma omp task
        {
            // M5 = (A11 + A12) * B22
            addMatBlocks(AM5, n, n, A11, A12);
            strassen_omp(n, AM5, B22, M5);
        }

#pragma omp task
        {
            // M6 = (A21 - A11) * (B11 + B12)
            subMatBlocks(AM6, n, n, A21, A11);
            addMatBlocks(BM6, n, n, B11, B12);
            strassen_omp(n, AM6, BM6, M6);
        }

#pragma omp task
        {
            // M7 = (A12 - A22) * (B21 + B22)
            subMatBlocks(AM7, n, n, A12, A22);
            addMatBlocks(BM7, n, n, B21, B22);
            strassen_omp(n, AM7, BM7, M7);
        }
#pragma omp taskwait
// необходимо дождаться выполнения подзадач

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                C11[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
                C12[i][j] = M3[i][j] + M5[i][j];
                C21[i][j] = M2[i][j] + M4[i][j];
                C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
            }

        deleteMatrix<int>(M1);
        deleteMatrix<int>(M2);
        deleteMatrix<int>(M3);
        deleteMatrix<int>(M4);
        deleteMatrix<int>(M5);
        deleteMatrix<int>(M6);
        deleteMatrix<int>(M7);

        deleteMatrix<int>(AM1);
        deleteMatrix<int>(BM1);
        deleteMatrix<int>(AM2);
        deleteMatrix<int>(BM3);
        deleteMatrix<int>(BM4);
        deleteMatrix<int>(AM5);
        deleteMatrix<int>(AM6);
        deleteMatrix<int>(BM6);
        deleteMatrix<int>(AM7);
        deleteMatrix<int>(BM7);

        delete[] A11; delete[] A12; delete[] A21; delete[] A22;
        delete[] B11; delete[] B12; delete[] B21; delete[] B22;
        delete[] C11; delete[] C12; delete[] C21; delete[] C22;
    }
}

void strassen(int size, int **A, int **B, int **C)
{
#pragma omp parallel
    {
#pragma omp single
        {
            strassen_omp(size, A, B, C);
        }
    }
}

void strassen_threads(int size, int **A, int **B, int **C, int nThreads)
{
    // если размер меньше порогового,
    // выполняем обычное последовательное умножение
    if (size <= threshold)
        seqMatMult(size, size, size, A, B, C);

    else
    {
        // создаем пул потоков
        ThreadPool pool(nThreads);
        int n = size / 2;

        int **M1 = createMatrix<int>(n, n);
        int **M2 = createMatrix<int>(n, n);
        int **M3 = createMatrix<int>(n, n);
        int **M4 = createMatrix<int>(n, n);
        int **M5 = createMatrix<int>(n, n);
        int **M6 = createMatrix<int>(n, n);
        int **M7 = createMatrix<int>(n, n);

        int **AM1 = createMatrix<int>(n, n);
        int **BM1 = createMatrix<int>(n, n);
        int **AM2 = createMatrix<int>(n, n);
        int **BM3 = createMatrix<int>(n, n);
        int **BM4 = createMatrix<int>(n, n);
        int **AM5 = createMatrix<int>(n, n);
        int **AM6 = createMatrix<int>(n, n);
        int **BM6 = createMatrix<int>(n, n);
        int **AM7 = createMatrix<int>(n, n);
        int **BM7 = createMatrix<int>(n, n);

        int **A11 = new int*[n];
        int **A12 = new int*[n];
        int **A21 = new int*[n];
        int **A22 = new int*[n];

        int **B11 = new int*[n];
        int **B12 = new int*[n];
        int **B21 = new int*[n];
        int **B22 = new int*[n];

        int **C11 = new int*[n];
        int **C12 = new int*[n];
        int **C21 = new int*[n];
        int **C22 = new int*[n];

        copyQtrMatrix(A11, n, A, 0, 0);
        copyQtrMatrix(A12, n, A, 0, n);
        copyQtrMatrix(A21, n, A, n, 0);
        copyQtrMatrix(A22, n, A, n, n);

        copyQtrMatrix(B11, n, B, 0, 0);
        copyQtrMatrix(B12, n, B, 0, n);
        copyQtrMatrix(B21, n, B, n, 0);
        copyQtrMatrix(B22, n, B, n, n);

        copyQtrMatrix(C11, n, C, 0, 0);
        copyQtrMatrix(C12, n, C, 0, n);
        copyQtrMatrix(C21, n, C, n, 0);
        copyQtrMatrix(C22, n, C, n, n);

// добавляем задачи в пул потоков
pool.enqueue([AM1, n, A11, A22, BM1, B11, B22, M1, nThreads]
        {
            // M1 = (A11 + A22) * (B11 + B22)
            addMatBlocks(AM1, n, n, A11, A22);
            addMatBlocks(BM1, n, n, B11, B22);
            strassen_threads(n, AM1, BM1, M1, nThreads);
});

pool.enqueue([AM2, n, A21, A22, B11, M2, nThreads]
{
            // M2 = (A21 + A22) * B11
            addMatBlocks(AM2, n, n, A21, A22);
            strassen_threads(n, AM2, B11, M2, nThreads);
});

pool.enqueue([BM3, n, B12, B22, A11, M3, nThreads]
{
            // M3 = A11 * (B12 - B22)
            subMatBlocks(BM3, n, n, B12, B22);
            strassen_threads(n, A11, BM3, M3, nThreads);
});

pool.enqueue([BM4, n, B21, B11, A22, M4, nThreads]
{
            // M4 = A22 * (B21 - B11)
            subMatBlocks(BM4, n, n, B21, B11);
            strassen_threads(n, A22, BM4, M4, nThreads);
});

pool.enqueue([AM5, n, A11, A12, B22, M5, nThreads]
{
            // M5 = (A11 + A12) * B22
            addMatBlocks(AM5, n, n, A11, A12);
            strassen_threads(n, AM5, B22, M5, nThreads);
});

pool.enqueue([AM6, n, A21, A11, BM6, M6, B11, B12, nThreads]
{
            // M6 = (A21 - A11) * (B11 + B12)
            subMatBlocks(AM6, n, n, A21, A11);
            addMatBlocks(BM6, n, n, B11, B12);
            strassen_threads(n, AM6, BM6, M6, nThreads);
});

pool.enqueue([AM7, n, A12, A22, BM7, M7, B21, B22, nThreads]
{
            // M7 = (A12 - A22) * (B21 + B22)
            subMatBlocks(AM7, n, n, A12, A22);
            addMatBlocks(BM7, n, n, B21, B22);
            strassen_threads(n, AM7, BM7, M7, nThreads);
});
        pool.Stop(); // необходимо дождаться выполнения всех задач

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                C11[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
                C12[i][j] = M3[i][j] + M5[i][j];
                C21[i][j] = M2[i][j] + M4[i][j];
                C22[i][j] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
            }

        deleteMatrix<int>(M1);
        deleteMatrix<int>(M2);
        deleteMatrix<int>(M3);
        deleteMatrix<int>(M4);
        deleteMatrix<int>(M5);
        deleteMatrix<int>(M6);
        deleteMatrix<int>(M7);

        deleteMatrix<int>(AM1);
        deleteMatrix<int>(BM1);
        deleteMatrix<int>(AM2);
        deleteMatrix<int>(BM3);
        deleteMatrix<int>(BM4);
        deleteMatrix<int>(AM5);
        deleteMatrix<int>(AM6);
        deleteMatrix<int>(BM6);
        deleteMatrix<int>(AM7);
        deleteMatrix<int>(BM7);

        delete[] A11; delete[] A12; delete[] A21; delete[] A22;
        delete[] B11; delete[] B12; delete[] B21; delete[] B22;
        delete[] C11; delete[] C12; delete[] C21; delete[] C22;
    }
}

int main(int argc, char **argv)
{
    threshold = atoi(argv[1]);
    int size = atoi(argv[2]);
    int nThreads = atoi(argv[3]);

    double t1, t2;

    int **A = createMatrix<int>(size, size);
    int **B = createMatrix<int>(size, size);
    int **C = createMatrix<int>(size, size);
    int **D = createMatrix<int>(size, size);

    srand((unsigned)time(0));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
        {
            B[i][j] = 0;
            A[i][j] = (rand() % 100) - 50;
        }

    // единичная матрица
    for (int i = 0; i < size; ++i)
        B[i][i] = 1;

    omp_set_num_threads(nThreads);
    t1 = omp_get_wtime();
    strassen(size, A, B, C);
    t2 = omp_get_wtime();
    printf("\nwall clock time (omp) = %f\n", t2 - t1);

    t1 = omp_get_wtime();
    strassen_threads(size, A, B, D, nThreads);
    t2 = omp_get_wtime();
    printf("wall clock time (threads) = %f\n", t2 - t1);

    // проверка
    bool flag = true;
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            if (D[i][j] != A[i][j]) flag = false;

    if (flag) printf("\ncorrect: matrix size = %d, # of threads = %d\n", size, omp_get_max_threads());
    else printf("\nerror: matrix size = %d, # of threads = %d\n", size, omp_get_max_threads());

    deleteMatrix<int>(A);
    deleteMatrix<int>(B);
    deleteMatrix<int>(C);
    deleteMatrix<int>(D);

    return 0;
}
