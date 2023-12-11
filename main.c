#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define N 2048
#define nGenerations 2000

void alocarMatriz(float ***grid)
{
    *grid = (float **)malloc(N * sizeof(float *));

    for (int i = 0; i < N; i++)
    {
        (*grid)[i] = (float *)malloc(N * sizeof(float));
    }
}

void desalocarMatriz(float ***grid)
{
    for (int i = 0; i < N; i++)
    {
        free((*grid)[i]);
    }
    free(*grid);
}

void zerarMatriz(float ***grid)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            (*grid)[i][j] = 0;
        }
    }
}

void glider(float ***grid)
{
    int lin = 1, col = 1;
    (*grid)[lin][col + 1] = 1.0;
    (*grid)[lin + 1][col + 2] = 1.0;
    (*grid)[lin + 2][col] = 1.0;
    (*grid)[lin + 2][col + 1] = 1.0;
    (*grid)[lin + 2][col + 2] = 1.0;
}

void rPentomino(float ***grid)
{
    int lin = 10;
    int col = 30;
    (*grid)[lin][col + 1] = 1.0;
    (*grid)[lin][col + 2] = 1.0;
    (*grid)[lin + 1][col] = 1.0;
    (*grid)[lin + 1][col + 1] = 1.0;
    (*grid)[lin + 2][col + 1] = 1.0;
}

void printarMatriz(float **grid)
{

    int i, j;

    for (i = 0; i < 50; i++)
    {
        for (j = 0; j < 50; j++)
        {

            printf("[%2.f]", grid[i][j]);
        }
    }
}

float mediaVivos(float ***grid, int i, int j)
{
    float totalVizinhos = 0.0;
    int offsets[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

    for (int k = 0; k < 8; ++k)
    {
        int px = (i + offsets[k][0] + N) % N; // Adicionado +N para garantir que o valor não seja negativo
        int py = (j + offsets[k][1] + N) % N;

        totalVizinhos += (*grid)[px][py];
    }

    return totalVizinhos * 0.125; // 1/8 = 0.125, assim evitamos a divisão
}

void trocarMatriz(float ***grid, float ***newGrid)
{

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {

            (*grid)[i][j] = (*newGrid)[i][j];
        }
    }
}

int somarVivos(float ***grid)
{

    int totalVivos = 0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {

            if ((*grid)[i][j] != 0.0)
            {
                totalVivos += 1;
            }
        }
    }

    return totalVivos;
}

void geracao(float ***grid, float ***newGrid, int start_row, int end_row)
{
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int totalVizinhos = 0;
            int aux, aux2;

            for (aux = -1; aux <= 1; aux++)
            {
                for (aux2 = -1; aux2 <= 1; aux2++)
                {
                    if (aux != 0 || aux2 != 0)
                    {
                        int px = i + aux, py = j + aux2;

                        // Ajuste para tratar as bordas
                        if (px < 0)
                            px = 0;
                        if (px >= end_row)
                            px = end_row - 1;
                        if (py < 0)
                            py = 0;
                        if (py >= N)
                            py = N - 1;

                        if ((*grid)[px][py] != 0.0)
                        {
                            totalVizinhos += 1;
                        }
                    }
                }
            }

            // Resto da lógica
            if (totalVizinhos < 2 || totalVizinhos > 3)
            {
                (*newGrid)[i][j] = 0;
            }
            else if (totalVizinhos == 3)
            {
                (*newGrid)[i][j] = ((*grid)[i][j] == 1) ? 1 : mediaVivos(grid, i, j);
            }
            else if (totalVizinhos == 2)
            {
                (*newGrid)[i][j] = ((*grid)[i][j] != 0) ? 1 : 0;
            }
        }
    }
}

int main(int argc, char **argv)
{
    float **grid, **newGrid;
    struct timeval inicioTotal, finalTotal, inicioLaco, finalLaco;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Calcula quantas linhas cada processo deve lidar

    int extra_rows = 2; // Uma linha extra para cada borda
    int rows_per_process = N / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;

    gettimeofday(&inicioTotal, NULL);

    alocarMatriz(&grid);
    alocarMatriz(&newGrid);

    zerarMatriz(&grid);
    zerarMatriz(&newGrid);

    glider(&grid);
    rPentomino(&grid);

    gettimeofday(&inicioLaco, NULL);

    for (int gen = 0; gen < nGenerations; gen++)
    {
        // Exchange boundary rows at the start of each generation
        if (rank != 0)
        {
            // Send top boundary to previous process and receive from previous process
            MPI_Sendrecv(grid[start_row], N, MPI_FLOAT, rank - 1, 0,
                         grid[start_row - 1], N, MPI_FLOAT, rank - 1, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank != size - 1)
        {
            // Send bottom boundary to next process and receive from next process
            MPI_Sendrecv(grid[end_row - 1], N, MPI_FLOAT, rank + 1, 1,
                         grid[end_row], N, MPI_FLOAT, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        geracao(&grid, &newGrid, start_row, end_row);
        trocarMatriz(&grid, &newGrid);
        zerarMatriz(&newGrid);

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
        {
            int resultado = somarVivos(&grid);
            // printf("\nRESULTADO DA GERACAO(%i) = %i", gen + 1, resultado);
        }
    }

    gettimeofday(&finalLaco, NULL);

    long long tempoLaco = (finalLaco.tv_sec - inicioLaco.tv_sec) * 1000 + (finalLaco.tv_usec - inicioLaco.tv_usec) / 1000;

    printf("\n-----Tempo total do laco -----\n");
    printf("%lld milisegundos \n", tempoLaco);
    printf("------------------------------\n");

    desalocarMatriz(&grid);
    desalocarMatriz(&newGrid);

    gettimeofday(&finalTotal, NULL);

    long long tempoTotal = (finalTotal.tv_sec - inicioTotal.tv_sec) * 1000 + (finalTotal.tv_usec - inicioTotal.tv_usec) / 1000;

    printf("---Tempo total do programa ---\n");
    printf("%lld milisegundos \n", tempoTotal);
    printf("------------------------------\n");

    MPI_Finalize();
    return 0;
}
