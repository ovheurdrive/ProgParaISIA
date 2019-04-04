# Rapport Programmation Parallèle TP 1


## I - Introduction

## Config
Pour réaliser ce TP, il faut se munir de GCC (7.3.0) et d'un processeur supportant les instructions AVX et AVX2.

Le fichier de header doit contenir (afin de pouvoir utiliser les instructions SIMD)
```C
#include <immintrin.h>
```

## Fonctionnement

Pour chaque exercice du TP, il y'a au moins une fonction scalaire (`exercice#_scal`) et une fonction vectorielle (`exercice#_vec`)

Les fonctions sont appelées depuis main dans un code de benchmark qui calcuel le temp moyen d'execution sur `MAX_ITER` itérations sur des vecteurs de taille `VEC_SIZE`

Tout les prototypes et les variables préprocesseurs sont définies dans le fichier header associé.

Les vecteurs d'input sont les mêmes pour tout les exercices et sont remplis de manière pseudo aléatoire par le PRNG suivant:
```C
srand(time(NULL));

float rand_float(){
    return (float)((rand() % 360) - 180.0);
}
```

Toutes les variables d'input (`res`, `vec1`, `vec2`, etc...) sont de type `float *` si non initialisées dans le code fourni dans le rapport.

Usage:  `make clean && make && ./simd.run`

Options de compilation standard: ` -O3 -std=c++17 -march=native`

---

## II - Moyenne de 3 Vecteurs

### Code Scalaire (variables: 3 vecteurs sources, un vecteur resultat):
```C
for ( int i = 0; i < VEC_SIZE; i++){
    res[i] = (vec1[i] + vec2[i] + vec3[i])/3;
}
```

### Code Vectoriel:
```C
__m256 d = _mm256_set1_ps(3);
for( int i = 0; i < VEC_SIZE; i+=8){
    __m256 a = _mm256_loadu_ps(&vec1[i]);
    __m256 b = _mm256_loadu_ps(&vec2[i]);
    __m256 c = _mm256_loadu_ps(&vec3[i]);

    __m256 s = _mm256_add_ps(a,b);
    s = _mm256_add_ps(s,c);
    s = _mm256_div_ps(s,d);

    _mm256_storeu_ps(&res[i], s);
}
```
### Nouvelles instructions SIMD: 

- `_mm256_set1_ps(float)`: initialise toutes les valeur d'un registre xmm à un float
- `_mm256_loadu_ps(float*)`: initialise le registre xmm à partir d'un tableau de [8xfloat]
- `_mm256_add_ps(__m256,__m256)`: ajoute deux registres xmm
- `_mm256_div_ps(__m256,__m256)`: divise deux registres xmm
- `_mm256_storeu_ps(float*,__m256)`: stock les valeur d'un registre xmm dans un tableau de [8xfloat]

### Performances (temps (en s) moyen par point sur 1000 itérations - Machine sb123):

<center>

|                        | Vectoriel | Scalaire  |
|------------------------|-----------|-----------|
|Vectorisation invisible |5.41732e-10|5.19041e-10|
|Vectorisation forcée    |5.25086e-10|8.22129e-10|
|Sans Vectorisation      |5.37963e-10|5.17184e-10|
</center>

---
## III - Produit Scalaire de 2 Vecteurs

### Code scalaire:
```C
float res = 0;
for ( int i = 0; i < VEC_SIZE; i++){
    res += vec1[i] * vec2[i];
}
return res;
```

### Code Vectoriel:

J'ai réalisé deux version Vectorielles pour ce problème de produit scalaire.

La première réalise la multiplication, puis un horizontal add et enfin extrait les resultats.

```C
float res = 0;
for( int i = 0; i < VEC_SIZE; i+=8){
    __m256 a = _mm256_loadu_ps(&vec1[i]);
    __m256 b = _mm256_loadu_ps(&vec2[i]);

    __m256 s = _mm256_mul_ps(a,b);

    s = _mm256_hadd_ps(s, s);
    s = _mm256_hadd_ps(s, s);

    __m128 hi = _mm256_extractf128_ps(s,0);
    __m128 lo = _mm256_extractf128_ps(s,1);
    
    res += _mm_cvtss_f32(hi) + _mm_cvtss_f32(lo);
}
return res;
```

Le seconde utilise la fonction SIMD dot product.

```C
float res = 0;
__m256 resv = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0); 

for (int i = 0; i < VEC_SIZE; i+=8) {    
    __m256 a = _mm256_loadu_ps(&vec1[i]);
    __m256 b = _mm256_loadu_ps(&vec2[i]);

    __m256 dotprod = _mm256_dp_ps(a, b, 241);

    resv = _mm256_add_ps(resv, dotprod);
}    
float t[8];
_mm256_storeu_ps(t, resv);
res = t[0] + t[4];
return res;
```

### Nouvelles instructions:

- `_mm256_mul_ps(__m256,__m256)`: multiplie deux registres xmm
- `_mm256_hadd_ps(__m256,__m256)`: Ajoute deux a deux horizontalement les valeurs de deux registres xmm et les store dans un seul
- `_mm256_extractf128_ps(__m256,int) -> __m128`: extrait une des deux lignes 128bits d'un registres xmm 256 bits
- `_mm_cvtss_f32(__m128)`: extrait la première valeur du registre xmm et la store dans un float
- `_mm256_dp_ps(__m256,__m256,int)`: realise le dot product de deux registres xmm deux à deux et store le resultat selon un entier de controle.

### Performances (temps (en s) moyen par point sur 1000 itérations - Machine personnelle -> i7-6700HQ):

<center>

|                        | Vectoriel V1 | Vectoriel V2 | Scalaire  |
|------------------------|--------------|--------------|-----------|
|Vectorisation invisible |4.19215e-10   |3.55027e-10   |1.36934e-09|
|Vectorisation forcée    |4.17045e-10   |3.64738e-10   |1.41293e-09|
|Sans Vectorisation      |4.26542e-10   |3.61731e-10   |1.36677e-09|
</center>

### Remarques:  
   J'ai pu remarquer que certaines valeurs initiales du PRNG causaient des erreurs de calculs -> les trois codes ont un résultat différent.  
   Pour trouver la source du problème, j'ai dérandomisé les données pour voir si l'erreur disparaissait (et elle l'a fait).  
   Ensuite j'ai utilisé le code randomisé jusqu'à obtenir une seed du PRNG qui causait l'erreur et ait pu observer qu'elle se reproduisait systématiquement.  
   En divisant la taille des données par deux, l'erreur disparait (et se trouve donc dans la deuxième partie des données)

   Seed utilisée: `time_t error_seed = 1549875661;`

---
## IV - Recherche de maximum/minimum d'un vecteur
### Remarque préliminaire:  
J'ai créé un type `minmax` afin de renvoyer les deux résultats:
```C
typedef struct minmax {
    float min;
    float max;
} minmax;
```

### Code scalaire:
```C
minmax res = { 1000, -1000 };
for (int i = 0; i < VEC_SIZE; i++ ) {
    if (vec[i] < res.min){
        res.min = vec[i];
    }
    if (vec[i] > res.max){
        res.max = vec[i];
    }
}
return res;
```

### Code vectoriel:  
Le principe est de faire des opération de min/max de manière vectorielles pour obtenir le vecteur min/max et à la fin extrait de manière scalaire le résultat du vecteur.

```C
minmax res = { 1000, -1000 };
// Load base vectors
__m256 cmp_max = _mm256_loadu_ps(&vec[0]);
__m256 cmp_min = _mm256_loadu_ps(&vec[0]);

for ( int i = 8; i < VEC_SIZE; i +=8 ){
    // Compare following vectors using min/max    
    __m256 val = _mm256_loadu_ps(&vec[i]);
    cmp_max = _mm256_max_ps(cmp_max, val);
    cmp_min = _mm256_min_ps(cmp_min, val);
}
// Retrieve vectors
float minv[8];
float maxv[8];
_mm256_storeu_ps(&minv[0], cmp_min);
_mm256_storeu_ps(&maxv[0], cmp_max);

// Scalar extraction
for (int i = 0; i < 8; i++ ) {
    if (minv[i] < res.min){
        res.min = minv[i];
    }
    if (maxv[i] > res.max){
        res.max = maxv[i];
    }
}
return res; 
```

### Nouvelles instructions:
- `_mm256_max_ps(__m256,__m256)`: compare les éléments deux deux vecteurs et renvoie un vecteur avec le max des éléments 2 à 2.
- `_mm256_min_ps(__m256,__m256)`: compare les éléments deux deux vecteurs et renvoie un vecteur avec le min des éléments 2 à 2.


### Performances (temps (en s) moyen par point sur 1000 itérations - Machine personnelle -> i7-6700HQ):

<center>

|                        | Vectoriel | Scalaire  |
|------------------------|-----------|-----------|
|Vectorisation invisible |2.14183e-10|1.28897e-09|
|Vectorisation forcée    |2.11412e-10|1.30522e-09|
|Sans Vectorisation      |2.14767e-10|1.30877e-09|
</center>

---
## V - Filtrage Gaussien

Le code utilisé ici effectue un filtrage gaussien (1-2-1) 1D => seul les extrèmes `0` et `VEC_SIZE-1` sont considérés comme des "bords".

### Code Scalaire:
```C
res[0] = (2*src[0] + src[1])/3;
for ( int i = 1; i < VEC_SIZE-1; i++){
    res[i] = (src[i-1] + 2*src[i] + src[i+1])/4;
}
res[VEC_SIZE-1] = (src[VEC_SIZE-2] + 2*src[VEC_SIZE-1])/3;
```

### Code vectoriel:
```C
// First vector (extreme case 1)
 __m256 a = _mm256_set_ps(src[6], src[5], src[4], src[3], src[2], src[1], src[0], 0.0);
__m256 b = _mm256_loadu_ps(&src[0]);
__m256 c = _mm256_loadu_ps(&src[1]);
__m256 d = _mm256_set_ps(4,4,4,4,4,4,4,3);

__m256 r = _mm256_add_ps(_mm256_add_ps(a,c), _mm256_add_ps(b,b));
r = _mm256_div_ps(r,d);
_mm256_storeu_ps(&res[0], r);

// Middle of vector
for(int i = 8; i < VEC_SIZE-8; i+=8){
    a = _mm256_loadu_ps(&src[i-1]);
    b = _mm256_loadu_ps(&src[i]);
    c = _mm256_loadu_ps(&src[i+1]);
    d = _mm256_set1_ps(4);

    r = _mm256_add_ps(_mm256_add_ps(a,c), _mm256_add_ps(b,b));
    r = _mm256_div_ps(r,d);
    _mm256_storeu_ps(&res[i], r);
}

// Last vector (extreme case 2)
a = _mm256_loadu_ps(&src[VEC_SIZE-9]);
b = _mm256_loadu_ps(&src[VEC_SIZE-8]);
c = _mm256_set_ps(0.0, src[VEC_SIZE-1], src[VEC_SIZE-2], src[VEC_SIZE-3], src[VEC_SIZE-4], src[VEC_SIZE-5], src[VEC_SIZE-6], src[VEC_SIZE-7]);
d = _mm256_set_ps(3,4,4,4,4,4,4,4);

r = _mm256_add_ps(_mm256_add_ps(a,c), _mm256_add_ps(b,b));
r = _mm256_div_ps(r,d);
_mm256_storeu_ps(&res[VEC_SIZE-8], r);
```

### Remarques:  
Le code Vectoriel est assez naïf et surement pas très optimisé, je pense qu'on pourrait le modifier en ne loadant que `vec[i]` et `vec[i+1]` et en faisant des permutations internes pour gerer les extremités de chaque vecteurs.

### Nouvelles instructions:
- `_mm256_set_ps([8xfloat])`: initalise un vecteur xmm avec 8 valeurs flottantes spécifiées.

### Performances (temps (en s) moyen par point sur 1000 itérations - Machine personnelle -> i7-6700HQ):

<center>

|                        | Vectoriel | Scalaire  |
|------------------------|-----------|-----------|
|Vectorisation invisible |4.42253e-10|4.32967e-10|
|Vectorisation forcée    |4.22387e-10|4.18147e-10|
|Sans Vectorisation      |4.33953e-10|8.42021e-10|
</center>

---
## VI - Valeur Absolue

### Code Scalaire:

Pour le code scalaire, j'ai fait en sorte qu'il soit possible de l'auto-vectoriser en utilisant le fait que les `float` utilisent un bit de signe. Je n'utilise ni `fabs` dont j'ignore si l'auto-vectorisation est possible, ni de conditions qui l'empècherait.

```C
for(int i = 0; i < VEC_SIZE; i++){
    int casted = *(int*)(&src[i]);
    casted &= 0x7fffffff;
    res[i] = *(float*)&casted;
    // or (with math.h)
    // res[i] = fabs(src[i]);
}
```
(Après test il semblerait que `fabs` soit auto-vectorisable)

### Code Vectoriel:

Version 1:  
```C
for( int i = 0; i < VEC_SIZE; i+=8){
    __m256 v = _mm256_loadu_ps(&src[i]);
    v = _mm256_andnot_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)), v);
    _mm256_storeu_ps(&res[i], v);
}

```
Version 2: 
```C
for( int i = 0; i < VEC_SIZE; i+=8){
    __m256 v = _mm256_loadu_ps(&src[i]);
    v = _mm256_max_ps(-v, v); // seems like gcc has builtins allowing us to do that
    _mm256_storeu_ps(&res[i], v);
}``````
```

### Nouvelles instructions:
- `_mm256_andnot_ps(__m256, __m256)`: Effectue l'opération `NOt` sur la première opérande puis effectue le `AND` binaire avec la deuxième.
- `_mm256_castsi256_ps(__m256i)`: Permet de caster un epi32 en ps (Cette opération n'effectue aucun changement et ne sert que pour la compilation)
- `_mm256_set1_epi32(int)`: intialise toutes les valeurs d'un registre xmm à un entier 32 bits [8xint].

### Performances (temps (en s) moyen par point sur 1000 itérations - Machine personnelle -> i7-6700HQ):

Code vectoriel testé :  `exercice5_vec(float*,float*)`  

<center>

|                        | Vectoriel | Scalaire  |
|------------------------|-----------|-----------|
|Vectorisation invisible |4.48051e-10|4.55274e-10|
|Vectorisation forcée    |4.55267e-10|4.56563e-10|
|Sans Vectorisation      |4.41499e-10|8.11731e-10|
</center>

---
## Remarques globales:

J'ai découvert après la rédaction de la majorité du code et de ce rapport que GCC a beaucoup de fonction builtins qui nous simplifient la vie en SIMD:

```C
void test(){
    __m256 a = _mm256_set1_ps(3);
    a = a - 1;
    a = a + 8;
    __m256 b = _mm256_set_ps(1,2,3,5,8,9,7,7);
    __m256 c = _mm256_set_ps(8,7,5,57,9,5,4,8);
    a = a * b;
    a = a / c;
    float d[8];
    _mm256_storeu_ps(&d[0], a);
    for(int i = 0; i < 8; i++)
        printf("%f\n", d[i]);
}
```

Output:
```
8.750000
17.500000
18.000000
8.888889
0.877193
6.000000
2.857143
1.250000
```