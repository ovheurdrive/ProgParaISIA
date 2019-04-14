# Rapport Programmation Parallèle TP 3

## I - Introduction

### Materiel

CPU :
- Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz
- 8 Cores

GPU :
- Nvidia Geforce GTX 970M
- 1280 Cuda Cores
- GPU Clock 924MHz + Boost
- Memory : 3024MiB GDDR5, 2500 MHz

### Folders

- `src`: codes sources
- `include`: headers
- `bench`: graphiques et log de benchmark
- `exe` et `obj` : binaires
- `images`: images sources et résultats

### Config

Pour réaliser ce TP, if faut une version récente de GCC supportant les dernière instructions OpenMP, une carte graphique Nvidia avec un pilote graphique à jour (390.116 sur GeForce GTX 970M dans le cas de ma machine). Il faut egalement installer le paquet `nvidia-cuda-toolkit` sur Linux qui fournit la dernière version de cuda disponible pour le Driver installé (9.1 à l'heure actuelle).

```C
#include "cuda_runtime.h"
#include "cuda.h"
#include "omp.h"
#include "device_launch_parameters.h"
```
Build : `make clean && make`

Usage : `./tpcuda.run [path to pgm image] [number of iterations]`

Fichiers : `main.(cu/cuh), utils.(cu/h), dotp.(cu/cuh), sobel.(cu/cuh), transpo.(cu/cuh), histo.(cu/cuh), container.(cu/cuh), ios.(cu/cuh), pixmap_io.(cu/cuh) and global_paramters.cuh`

Le compilateur utilisé est `nvcc` (Nvidia C Compiler) avec les options de compilation suivantes :
`-ccbin g++-6 -gencode arch=compute_50,code=sm_50 -I /usr/local/cuda/include -std=c++11 -Xcompiler -fopenmp -O3 -Iinclude`


### Fonctionnement

Dans ce TP, nous allons paralléliser des programmes de traitements d'images sur GPU en utilisant Nvidia CUDA Toolkit 9.1.

Pour utiliser cuda, il faut importer les headers suivants:
```C
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
```
Il faut utiliser l'extension `.cu` au lieu de `.cpp` et `.cuh` au lieu de `.h`.

Pour executer du code sur le gpu, on doit créer un fonction appelée `kernel` qu'on déclare ainsi:

```C
__global__ void kernel(args){}
```
Dans ce kernel, on va utiliser plusieurs nouveaux mots clés :

- `__shared__ type variable[size]` : Déclaration d'un tableau dans la mémoire partagée du GPU
- `blockIdx`, `blockDim`, `threadIdx` : Variables permettant au kernel de connaitre des informations positionnelles dans le GPU
- `atomidAdd(&acc, val)` : ajoute la valeur `val` à l'accumulateur `acc` de manière atomique (thread safe)
- `__syncthreads()` : fonction pour synchroniser les threads d'un block

Ce kernel sera appelé avec des blocks contenant chacun un certain nombre de threads. Ces deux paramètres sont passé au kernel ainsi et sont de type `dim3`:

```C
// nombre d'argument de 1 à 3 pour le nombre de dimensions
dim3 blocks(args); // ex: block(10,10)
dim3 threads(args2); // threads(32,32)
kernel<<<blocks,threads>>>(args);
```

Les données envoyés au kernel (si elles ne sont pas de types atomiques comme `int` ou `unsigned`) doivent être alloués dans le mémoire du GPU en utilisant les fonctions suivantes:

```C
// On alloue srcCuda et resCuda dans la mémoire du GPU
cudaMalloc(&srcCuda, size);
cudaMalloc(&rescuda, size);

// On copie nos données dans cette variable
cudaMemcpy(srcCuda, source, size, cudaMemcpyHostToDevice);

// On appelle le kernel avec n blocks de k threads
kernel<<<n,k>>>(srcCuda, resCuda, other_args...);

// On récupère le résultat dans une variable CPU
cudaMemcpy(res, resCuda, size, cudaMemcpyDeviceToHost);

// On libère la mémoire du GPU
cudaFree(srcCuda);
cudaFree(resCuda);
```

Pour le premier exercice, on utilise un PRNG pour remplir les vecteurs.

```C
srand(time(NULL));

float rand_float(){
    return (float)((rand() % 360) - 180.0);
}
```
Pour les exercices suivants on peut utiliser les images présentes dans le dossier `images` dont les résultats modifiés sont sauvegardés dans le dossier `images/Résultats`.

Pour chaque fonction, le temps d'execution est mesuré en utilisant la fonction `omp_get_wtime()`.

Les becnhmark effectués qui ont aidé à générer les graphiques sont fournis avec ces même graphiques dans le dossier `bench`.

Tout les codes CPU sont parallélisés sur 8 threads en utilisant OpenMP.

Afin de profilé l'utilisation du GPU, je vais utiliser `nvprof` (l'outils en cli derrière `nvvp` qiu est moins visuel mais avec des résultats plus simples à visualiser considérant le fait que j'itère beaucoup de fois et que le profiler fait le programme complet, les résultats ne sont donc pas très visuels)

## II - Produit Scalaire de 2 vecteurs

Dans le code CUDA, on utilise une variable locale qui va contenir le produit partiel, et on utilise `atomicAdd` pour ajouter ce resultat de manière thread safe au résultat global.

### Code source (src/1-dotp/dotp.cu)

```C
__global__ void gpu_dotp_kernel(int size, float* vec1, float* vec2, float* res){

    float cache = 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i < size ){
        cache = vec1[i]*vec2[i];
    }

    atomicAdd(res, cache);
}

float cpu_dotp(float* vec1, float* vec2, int size){
    int res = 0;
    #pragma omp parallel num_threads(8)
    {
        #pragma for reduction(+:res) schedule(auto)
        for( int i = 0; i < size; i++ ){
            res+= vec1[i]*vec2[i];
        }
    }
    return res;
}
```

### Performances

Benchmark sur 1000 itérations:

| Data Size   | GPU Kernel Acceleration | GPU Total Acceleration |
|-------------|-------------------------|------------------------|
| 16 x 16     |        0.135448         |        0.0407969       |
| 32 x 32     |        0.497353         |        0.142652        |
| 64 x 64     |        1.99547          |        0.431701        |
| 128 x 128   |        7.60698          |        0.948662        |
| 256 x 256   |        28.2657          |        1.28158         |
| 512 x 512   |        87.3893          |        1.39062         |
| 1024 x 1024 |        180.303          |        1.50325         |
| 2048 x 2048 |        905.526          |        1.46826         |
| 4096 x 4096 |        2814.33          |        1.32986         |

On peut observer que le kernel accélere de manière exponentielle en fonction de la taille de la donnée, mais que le temps d'exeuction total reste relativement borné:

![](bench/dotp_accel.png)

Cela s'explique simplement quand on regarde le profil d'execution GPU:
```console
GPU activities:   65.94%  54.4803s      1000  54.480ms  54.480ms  54.490ms  gpu_dotp_kernel(int, float*, float*, float*)
                  34.05%  28.1340s      2000  14.067ms  13.018ms  32.907ms  [CUDA memcpy HtoD]
                   0.00%  1.2704ms      1000  1.2700us  1.2160us  2.9120us  [CUDA memcpy DtoH]
     API calls:   88.55%  82.9214s      3000  27.640ms  13.142ms  54.622ms  cudaMemcpy
                  10.47%  9.80090s      3000  3.2670ms  180.48us  6.8577ms  cudaFree
                   0.92%  862.85ms      3000  287.62us  95.322us  96.786ms  cudaMalloc
                   0.06%  57.378ms      1000  57.377us  46.292us  251.46us  cudaLaunch
                   0.00%  2.1361ms      1000  2.1360us     900ns  15.531us  cudaConfigureCall
                   0.00%  1.6680ms      4000     417ns     132ns  14.679us  cudaSetupArgument
                   0.00%  562.96us        94  5.9880us     908ns  218.99us  cuDeviceGetAttribute
                   0.00%  101.57us         1  101.57us  101.57us  101.57us  cuDeviceTotalMem
                   0.00%  70.909us         1  70.909us  70.909us  70.909us  cuDeviceGetName
                   0.00%  6.6220us         3  2.2070us     991ns  3.2050us  cuDeviceGetCount
                   0.00%  3.9140us         2  1.9570us  1.2370us  2.6770us  cuDeviceGet
```

On observe en effet que les call API `cudaMemcpy` prennent la grande majorité du temps d'execution total. On est borné en execution par les accès mémoire.

![](bench/dotp_logtime.png)

On voit ici le log10 des temps d'execution en fonction de la taille des données: on observe bien une sorte de barrière à 10e-9s qui est le temps d'accès à la mémoire.


### Remarques

Parfois le résultat du CPU et du GPU étaient différents :

```console
Dotp vec size : 16777216
Image Dimensions: 720x480
Exercice 1 : Dot Product
====================================
Cpu dot prod: -368561
Gpu dot prod: -369235
Absolute Error: 674

Dotp vec size : 4194304
Image Dimensions: 720x480
Exercice 1 : Dot Product
====================================
Cpu dot prod: -4.59882e+06
Gpu dot prod: -4.59886e+06
Absolute Error: 43

Dotp vec size : 1048576
Image Dimensions: 720x480
Exercice 1 : Dot Product
====================================
Cpu dot prod: 1.5272e+07
Gpu dot prod: 1.52721e+07
Absolute Error: 81
```

Je ne saurai pas expliquer d'ou proviens cette différence de résultat, mais cela reste un problème. (Peut être des erreurs de conversion de types ou d'arrondi de nombre à virgule flottante)


## III - Filtre de Sobel

*Ce code a été fourni par le professeur M. Cabaret*

### Code Source (src/2-sobel/sobel.cu)

```C
__global__ void gpu_sobel_kernel_naive(u_char *Source, u_char *Resultat, unsigned width, unsigned height) {
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    u_char val;
    int globalIndex = i*width+j;
    if ((i==0)||(i>=height-1)||(j==0)||(j>=width-1)) {Resultat[globalIndex]=0;}
    else {
        val  = std::abs(Source[(i-1)*width+(j-1)] + Source[(i-1)*width+(j)] + Source[(i-1)*width+(j+1)] -\
                       (Source[(i+1)*width+(j-1)] + Source[(i+1)*width+(j)] + Source[(i+1)*width+(j+1)]));
        Resultat[globalIndex]  = val + std::abs(Source[(i-1)*width+(j-1)] + Source[(i)*width+(j-1)] + Source[(i+1)*width+(j-1)] -\
                                             (Source[(i-1)*width+(j+1)] + Source[(i)*width+(j+1)] + Source[(i+1)*width+(j+1)]));

    }
}

__global__ void gpu_sobel_kernel_shared(u_char *Source, u_char *Resultat, unsigned width, unsigned height) {
    __shared__ u_char tuile[BLOCKDIM_X][BLOCKDIM_Y];
    
    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = blockIdx.y*(BLOCKDIM_Y-2) + y;
    int j = blockIdx.x*(BLOCKDIM_X-2) + x;
    
    int globalIndex = i*width+j;

    if ((i==0)||(i>=height-1)||(j==0)||(j>=width-1)) {}
    else {            
        //mainstream    
        tuile[x][y] = Source[globalIndex];
        __syncthreads();

        u_char val;
        if ((x>0)&&(y>0)&&(x<BLOCKDIM_X-1)&&(y<BLOCKDIM_Y-1)) {
            val = std::abs(tuile[x-1][y-1] + tuile[x-1][y] + tuile[x-1][y+1] -\
                          (tuile[x+1][y-1] + tuile[x+1][y] + tuile[x+1][y+1]));
            Resultat[globalIndex]  = val + std::abs(tuile[x-1][y-1] + tuile[x][y-1] + tuile[x+1][y-1] -\
                                                   (tuile[x-1][y+1] + tuile[x][y+1] + tuile[x+1][y+1]));
        }
    }    
}

void cpu_sobel(u_char **Source, u_char **Resultat, unsigned width, unsigned height) {
    #pragma omp parallel for num_threads(8)
    for (auto i = 1; i < height-1; i++) {
        for (auto j = 1; j < width-1; j++) {
            if ((i==0)||(i==height-1)||(j==0)||(j==width-1)) {Resultat[i][j]=0;}
            else {
                Resultat[i][j]  = std::abs(Source[i-1][j-1] + Source[i-1][j] + Source[i-1][j+1] - (Source[i+1][j-1] + Source[i+1][j] + Source[i+1][j+1]));
                Resultat[i][j] += std::abs(Source[i-1][j-1] + Source[i][j-1] + Source[i+1][j-1] - (Source[i-1][j+1] + Source[i][j+1] + Source[i+1][j+1]));
            }
        }
    }
}
```

### Performances

Les performances sont mesurées sur 1000 itérations.
L'échelle de droite des graphiques représente le log2 de la taille des images (width*height).

![](bench/sobel_exec.png)

On peut observer que le temps d'execution du cpu est constant peut importe la taille de l'image, mais on observe comme précédemment que le kernel est bien plus rapide sur des données de tailles plus grandes.

Cependent on observe toujours le bottlebeck à 10e-9s qui est le temps d'accès mémoire.

En regardant les statistiques du GPU données par nvprof, on obtient par exemple sur `Drone_Huge.png`:

```console
==24064== Profiling application: exe/tpcuda.run ./images/Drone_huge.pgm 1000
==24064== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   38.80%  4.05111s      2000  2.0256ms  1.9781ms  5.4397ms  [CUDA memcpy DtoH]
                   38.33%  4.00210s      2000  2.0010ms  1.9278ms  5.3910ms  [CUDA memcpy HtoD]
                   12.08%  1.26123s      1000  1.2612ms  1.2507ms  1.2710ms  gpu_sobel_kernel_shared(unsigned char*, unsigned char*, unsigned int, unsigned int)
                   10.79%  1.12621s      1000  1.1262ms  1.1202ms  1.1406ms  gpu_sobel_kernel_naive(unsigned char*, unsigned char*, unsigned int, unsigned int)
      API calls:   78.15%  11.0169s      4000  2.7542ms  1.9893ms  7.2746ms  cudaMemcpy
                   13.29%  1.87395s      3000  624.65us  120.33us  1.1149ms  cudaFree
                    8.08%  1.13874s      3000  379.58us  102.15us  114.38ms  cudaMalloc
                    0.44%  62.497ms      2000  31.248us  14.094us  427.70us  cudaLaunch
                    0.01%  2.0914ms      8000     261ns      98ns  19.927us  cudaSetupArgument
                    0.01%  1.9170ms      2000     958ns     275ns  13.884us  cudaConfigureCall
                    0.00%  430.74us        94  4.5820us     563ns  180.21us  cuDeviceGetAttribute
                    0.00%  77.678us         1  77.678us  77.678us  77.678us  cuDeviceTotalMem
                    0.00%  47.486us         1  47.486us  47.486us  47.486us  cuDeviceGetName
                    0.00%  4.0300us         3  1.3430us     736ns  2.4360us  cuDeviceGetCount
                    0.00%  2.2380us         2  1.1190us     741ns  1.4970us  cuDeviceGet
```

On voit tout de suite que les temps de transfert de données d'accès à la mémoire sont le bottleneck de ce programme.

Cependant, on peut tout de même observer une accélération significative de l'execution du programme:

![](bench/sobel_accel.png)

(Les échelles sont en log10, donc on obtient des accélération réelles entre 20 et 40, voir `bench/sobel.csv`)

### Résultats

*Example sur mona_lisa.pgm*

| CPU | GPU | GPU Shared |
|-|-|-|
|![](images/Resultats/Sobel_cpu.png) | ![](images/Resultats/Sobel_gpu.png) | ![](images/Resultats/Sobel_gpu_shared.png) |
 
On observe des lignes noires sur certains bords qui sont dues au fait que il n'y a pas assez de block pour couvrir l'intégralité de l'image (à 
modifier dans le code dans la déclaration de `dim3 blocks`)

Le résultat en shared présente aussi des résultats étranges sur les bords (pixels blancs/noirs) qui sont dues au fait qu'on utilise de la mémoire partagée.

## IV - Transposée d'une image

Pour la transposée, je me suis inspiré, du code de sobel, et ait appliqué un algorithme de transposition d'abord naif (inversion hauteur/largeur), mais cet algorithme ne permet pas des accès mémoire coalescent.

La solution pour remédier à cela est d'uiliser la mémoire partagée du GPU pour créer une table intermédiaire dans laquel on effectue la transposition, puis on écrit de manière coalescente le résultat final dans le tableau.

### Code Source (src/3-transpo/transpo.cu)

```C
__global__ void gpu_transpo_kernel_naive(u_char *Source, u_char *Resultat, unsigned width, unsigned height){
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    if ((i<0)||(i>=height)||(j<0)||(j>=width)) {}
    else {
        Resultat[j*height + i]  = Source[i*width + j];
    }
}

__global__ void gpu_transpo_kernel_shared(u_char *Source, u_char *Resultat, unsigned width, unsigned height) {
    __shared__ u_char tuile[BLOCKDIM_X][BLOCKDIM_Y+1];
    
    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = blockIdx.y*(BLOCKDIM_Y) + y;
    int j = blockIdx.x*(BLOCKDIM_X) + x;
    

    if ((i<0)||(i>=height)||(j<0)||(j>=width)) {}
    else {            
        tuile[y][x] = Source[i*width + j];
        __syncthreads();
        int i = blockIdx.y*(BLOCKDIM_Y) + x;
        int j = blockIdx.x*(BLOCKDIM_X) + y;
        Resultat[j*height + i] = tuile[x][y];
    }    
}

void cpu_transpo(u_char **Source, u_char **Resultat, unsigned width, unsigned height){
    #pragma omp parallel for num_threads(8)
    for (unsigned i = 0; i < height; i++) {
        for (unsigned j = 0; j < width; j++) {
            Resultat[j][i]  = Source[i][j];
        }
    }
}
```

### Performances

J'ai fait tourner le benchmark sur 1000 itérations. 

![](bench/transpo_exec.png)

On observe un comportement étrange sur l'image `Carre.pgm` : Le temps d'execution total descend bien en dessous de la barrière habituelle des 10e-9s.

Je n'ai pas pu vérifier l'origine de ces valeurs car le profiler a dit être incapable de mesurer précisément le temps d'execution (invalid timestamps).

Pour toutes les autres images, on observe toujours cette limitation liée aux accès mémoire.

Le profiling du GPU donne des résultats comme les suivants (`Drone.pgm` et `Drone_huge.pgm`)

```console
==7605== Profiling application: exe/tpcuda.run ./images/Drone.pgm 1000
==7605== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   35.74%  80.503ms      1000  80.503us  79.489us  82.145us  gpu_transpo_kernel_naive(unsigned char*, unsigned char*, unsigned int, unsigned int)
                   26.71%  60.174ms      2000  30.086us  27.616us  113.31us  [CUDA memcpy DtoH]
                   26.40%  59.458ms      2000  29.728us  27.456us  106.85us  [CUDA memcpy HtoD]
                   11.15%  25.125ms      1000  25.124us  24.896us  25.408us  gpu_transpo_kernel_shared(unsigned char*, unsigned char*, unsigned int, unsigned int)
      API calls:   51.75%  411.17ms      4000  102.79us  27.077us  449.38us  cudaMemcpy
                   30.61%  243.15ms      3000  81.050us  3.1720us  116.36ms  cudaMalloc
                   13.73%  109.07ms      3000  36.355us  4.0450us  587.67us  cudaFree
                    3.52%  27.951ms      2000  13.975us  7.5780us  350.43us  cudaLaunch
                    0.21%  1.6391ms      8000     204ns     124ns  4.1170us  cudaSetupArgument
                    0.10%  814.17us      2000     407ns     198ns  11.630us  cudaConfigureCall
                    0.06%  492.20us        94  5.2360us     704ns  183.26us  cuDeviceGetAttribute
                    0.01%  99.941us         1  99.941us  99.941us  99.941us  cuDeviceTotalMem
                    0.01%  66.492us         1  66.492us  66.492us  66.492us  cuDeviceGetName
                    0.00%  6.1370us         3  2.0450us  1.0470us  3.7180us  cuDeviceGetCount
                    0.00%  3.0400us         2  1.5200us     974ns  2.0660us  cuDeviceGet


==7442== Profiling application: exe/tpcuda.run ./images/Drone_huge.pgm 1000
==7442== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
GPU activities:   35.29%  4.10725s      2000  2.0536ms  1.9472ms  5.8023ms  [CUDA memcpy HtoD]
                  35.16%  4.09172s      2000  2.0459ms  1.9795ms  5.8641ms  [CUDA memcpy DtoH]
                  22.31%  2.59662s      1000  2.5966ms  2.5888ms  2.6033ms  gpu_transpo_kernel_naive(unsigned char*, unsigned char*, unsigned int, unsigned int)
                   7.25%  843.36ms      1000  843.36us  840.75us  845.58us  gpu_transpo_kernel_shared(unsigned char*, unsigned char*, unsigned int, unsigned int)
    API calls:    79.75%  12.2407s      4000  3.0602ms  2.0172ms  8.4083ms  cudaMemcpy
                  12.31%  1.88959s      3000  629.86us  143.18us  1.1208ms  cudaFree
                   7.46%  1.14505s      3000  381.68us  102.78us  111.44ms  cudaMalloc
                   0.45%  68.481ms      2000  34.240us  16.491us  446.39us  cudaLaunch
                   0.01%  2.1771ms      2000  1.0880us     369ns  22.550us  cudaConfigureCall
                   0.01%  1.8696ms      8000     233ns     101ns  19.375us  cudaSetupArgument
                   0.00%  362.64us        94  3.8570us     476ns  146.40us  cuDeviceGetAttribute
                   0.00%  56.635us         1  56.635us  56.635us  56.635us  cuDeviceTotalMem
                   0.00%  38.231us         1  38.231us  38.231us  38.231us  cuDeviceGetName
                   0.00%  2.9330us         3     977ns     513ns  1.7480us  cuDeviceGetCount
                   0.00%  1.5280us         2     764ns     529ns     999ns  cuDeviceGet
```

On observe bien que le kernel shared est plus rapide de manière relativement significative par rapport au kernel non coalescent.

![](bench/transpo_accel.png)

### Resultas

Voisi un exemple des résultats obtenus (`coins.pgm` et `Drone.pgm`)

| CPU | GPU | GPU Shared | GPU Shared (Drone.pgm) |
|-|-|-|-|
|![](images/Resultats/Transpo_cpu.png) | ![](images/Resultats/Transpo_gpu.png) | ![](images/Resultats/Transpo_gpu_shared.png) | ![](images/Resultats/Transpo_Drone.png)|

On observe des problème sur l'algorithme shared avec l'image `coins.pgm` : Une partie des pixels/blocks semble ne pas avoir été récupéré correctement.

Mais en appliquant exactement le même code sur l'image `Drone.pgm`, on observe plus ces problèmes.

Je ne saurais pas en expliquer la source.


## V - Histogramme d'une image en niveau de gris

Pour réaliser l'histogramme d'une imageen niveau de gris, La méthode naive est que chaque thread fait un `atomicAdd` de son pixel dans le tableau, mais il y'a donc une forte concurrence sur cette opération avec beaucoups de lock potentiels.

Une solution plus intelligente, serait de créer un histogramme shared dans chaque block qu'on initialise à 0. Ensuite, chaque block va remplir son histogramme avec l'ensemble de pixels qui lui sont attribués avec des `atomicAdd`, puis une fois que tout les threads du block ont terminé, on `atomicAdd` les histogrammes partiel dans l'histogramme final. 

### Code Source (src/4-histo/histo.cu)

```C
__global__ void gpu_histo_kernel_naive(u_char* Source, int *res, unsigned height, unsigned width){
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    if ((i<0)||(i>=height)||(j<0)||(j>=width)) {}
    else {
        u_char val = Source[i*width+j];
        atomicAdd(&res[val],1);
    }
}

__global__ void gpu_histo_kernel_shared(u_char* Source, int *res, unsigned height, unsigned width){
    __shared__ int hist[256];

    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    int index = threadIdx.x * BLOCKDIM_X + threadIdx.y;

    if( index < 256) {
        hist[index] = 0;
    }
    __syncthreads();


    if ((i<0)||(i>=height) || (j<0) || (j>=width)) {}
    else {
        atomicAdd(&hist[Source[i*width+j]], 1);
        __syncthreads();
        if( index < 256)
            atomicAdd(&res[index], hist[index]);
    }
}

void cpu_histo(u_char** Source, int (*res)[256], unsigned height, unsigned width){
    #pragma omp parallel for num_threads(8)
    for( int i = 0; i < height; i++){
        for( int j = 0; j < width; j++){
            #pragma omp atomic
            (*res)[Source[i][j]]++;
        }
    }
}
```

### Performances

Voici les résultats du benchmark sur 1000 itérations.

![](bench/histo_exec.png)

Les résultats sont assez similaire aux autresproblèmes, cependant on observe un différence vraiment notable entre l'algorithme naif et l'algorithme partagé. En effet ce dernier  est en general deux fois plus rapide que l'algorithme naif.

On peut observer cela de manière flagrante avec le profiler,  le kernel naif est en mmoyenne deux fois plus lent que le kernel partagé (`Drone.pgm` et `Drone_huge.pgm`) :

```console
==2647== Profiling application: exe/tpcuda.run ./images/Drone_huge.pgm 1000
==2647== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   47.76%  6.8572ms         1  6.8572ms  6.8572ms  6.8572ms  gpu_histo_kernel_naive(unsigned char*, int*, unsigned int, unsigned int)
                   29.41%  4.2220ms         2  2.1110ms  2.0843ms  2.1378ms  [CUDA memcpy HtoD]
                   22.81%  3.2743ms         1  3.2743ms  3.2743ms  3.2743ms  gpu_histo_kernel_shared(unsigned char*, int*, unsigned int, unsigned int)
                    0.02%  2.6880us         2  1.3440us  1.3120us  1.3760us  [CUDA memcpy DtoH]
      API calls:   88.95%  132.41ms         3  44.136ms  11.018us  132.27ms  cudaMalloc
                    9.87%  14.690ms         4  3.6724ms  2.1762ms  6.9099ms  cudaMemcpy
                    0.69%  1.0203ms         3  340.11us  11.173us  849.99us  cudaFree
                    0.32%  478.04us        94  5.0850us     458ns  196.66us  cuDeviceGetAttribute
                    0.08%  119.19us         1  119.19us  119.19us  119.19us  cuDeviceGetName
                    0.05%  70.939us         1  70.939us  70.939us  70.939us  cuDeviceTotalMem
                    0.04%  54.589us         2  27.294us  21.592us  32.997us  cudaLaunch
                    0.00%  6.6360us         3  2.2120us     800ns  4.8290us  cuDeviceGetCount
                    0.00%  5.1970us         2  2.5980us     610ns  4.5870us  cuDeviceGet
                    0.00%  2.2050us         8     275ns     136ns     633ns  cudaSetupArgument
                    0.00%  2.0530us         2  1.0260us  1.0100us  1.0430us  cudaConfigureCall

==2665== Profiling application: exe/tpcuda.run ./images/Drone.pgm 1000
==2665== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   54.21%  197.03us         1  197.03us  197.03us  197.03us  gpu_histo_kernel_naive(unsigned char*, int*, unsigned int, unsigned int)
                   29.64%  107.71us         1  107.71us  107.71us  107.71us  gpu_histo_kernel_shared(unsigned char*, int*, unsigned int, unsigned int)
                   15.52%  56.417us         2  28.208us  27.329us  29.088us  [CUDA memcpy HtoD]
                    0.63%  2.3040us         2  1.1520us     992ns  1.3120us  [CUDA memcpy DtoH]
      API calls:   98.73%  107.69ms         3  35.896ms  6.4140us  107.56ms  cudaMalloc
                    0.44%  475.55us        94  5.0590us     733ns  185.95us  cuDeviceGetAttribute
                    0.43%  466.50us         4  116.63us  38.438us  208.68us  cudaMemcpy
                    0.23%  255.31us         3  85.103us  9.4790us  139.56us  cudaFree
                    0.08%  82.162us         1  82.162us  82.162us  82.162us  cuDeviceTotalMem
                    0.05%  50.476us         2  25.238us  12.574us  37.902us  cudaLaunch
                    0.05%  50.337us         1  50.337us  50.337us  50.337us  cuDeviceGetName
                    0.00%  4.1420us         3  1.3800us     845ns  2.3330us  cuDeviceGetCount
                    0.00%  2.4760us         2  1.2380us     854ns  1.6220us  cuDeviceGet
                    0.00%  1.8330us         8     229ns     134ns     462ns  cudaSetupArgument
                    0.00%  1.7740us         2     887ns     815ns     959ns  cudaConfigureCall
```

On observe pour la première fois que le `cudaMemcpy` n'est pas le bottleneck du programme car on ne copie qu'un tableau de [256*int].

Les bottleneck ici sont vraiment le kernel d'un côté et les allocations mémoire (qui ne sont pas comptées dans le benchmark d'execution) de l'autre.

![](bench/histo_accel.png)

Ici on observe bien une correlation entre l'accélération naif/shared du kernel , et celui du programme entier.

### Résultats

Voici deux histogrammes obtenus avec ces programmes:

|Histogram| Image|
|-|-|
| ![](bench/mona_lisa_histo.png) | ![](images/mona_lisa.png) |
| ![](bench/drone_histo.png) | ![](images/Drone.png) |

On voit bien que les 3 courbes sont supperposées, donc l'erreur des programme semble nulle à priori.

## VI - Remarques Générales

### Sur les benchmarks

Je n'ai pas effectué de mesures de performances sur des tailles de blocs/threads différents par manque de temps, mais on observe que (dans une cert aine limite) l'augmentation de la dimension des blocks à tendance à accelérer les algorithmes GPU.

### Sur les résultats

On observe que sur certains algorithmes, il y'a une quantité de problèmes/erreurs non négligeables (dotp, sobel, transpo), et je n'ai pas pu en isoler la cause exacte.