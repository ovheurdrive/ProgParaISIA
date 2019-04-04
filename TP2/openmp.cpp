
#include "openmp.h"

int main(int argc, char** argv){
    int n_threads = 8;
    omp_set_num_threads(n_threads);
    double stats[4][POW_MAX][3];
    int j =0;
    for( int i = DIV_MAX; i >= DIV_MIN; i=i/2, j++){
        std::cout << "Dataset size: " << VEC_SIZE/i << std::endl;
        run_exercices(VEC_SIZE/i, n_threads, &stats, j);
    }
    #ifdef SAVE_BENCH
    create_stats_csv(stats);
    #endif

    /* Exercice 4 */
    /* Soble filter exercice : We dont change the data size as we check on an image sample */

    std::cout << "Exercice 4:\n====================\n";
    if( argc > 1 ){
        double start, end, omp_time, seq_time;
        unsigned int height, width;

        std::string image_filename(argv[1]);

        get_source_params(image_filename, &height, &width);
        std::cout << width << " " << height << std::endl;
        u_char **Source, **Resultat;

        image<u_char> imgSource(height, width, &Source);
        image<u_char> imgResultat(height, width, &Resultat);
        
        auto fail = init_source_image(Source, image_filename, height, width);
        if (fail) {
            std::cout << "Chargement impossible de l'image" << std::endl;
            return 0;
        }

        start = omp_get_wtime();
        for( int j = 0; j< MAX_ITER; j++){
            exercice4_seq(Source, Resultat, height, width);
        }
        end = omp_get_wtime();
        seq_time = (end-start)/(MAX_ITER*height*width);
        std::cout << "Seq Time: " << seq_time << std::endl;

        #ifdef SAVE_IMG
        image_filename=std::string("Sobel_seq.pgm");
        save_gray_level_image(&imgResultat, image_filename, height, width);
        #endif

        start = omp_get_wtime();
        for( int j = 0; j< MAX_ITER; j++){
            exercice4_omp(Source, Resultat, height, width);
        }
        end = omp_get_wtime();
        omp_time = (end-start)/(MAX_ITER*height*width);
        std::cout << "OMP Time: " << omp_time << " Acceleration Ratio: " << seq_time/omp_time  << std::endl << std::endl;


        #ifdef SAVE_IMG
        image_filename=std::string("Sobel_para.pgm");
        save_gray_level_image(&imgResultat, image_filename, height, width);
        #endif
    }
    else {
        std::cout << "No argument passed. Skipping..." << std::endl << std::endl;
    }
    return 0;

}

int run_exercices(int data_size, int n_threads, double p_stats[][4][POW_MAX][3] , int iter){
    // Setup
    double* vec1 = (double*)malloc(data_size*sizeof(double));
    double* vec2 = (double*)malloc(data_size*sizeof(double));
    double* res1 = (double*)calloc(data_size, sizeof(double));
    double* res2 = (double*)calloc(data_size, sizeof(double));
    int* list = (int*)malloc(data_size*sizeof(int));
    int* sublist1 = (int*)calloc(data_size, sizeof(int));
    int* sublist2 = (int*)calloc(data_size, sizeof(int));

    for( int i = 0; i < data_size; i++) {
        vec1[i] = rand_double();
        vec2[i] = rand_double();
        list[i] = rand_int();
    }
    double start, end;
    double omp_time, seq_time;

    #ifdef EX1
    /* Exercice 1 */
    std::cout << "Exercice 1:\n====================\n";

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        exercice1_seq(vec1,vec2,res2,data_size);
    }
    end = omp_get_wtime();
    seq_time = (end-start)/(MAX_ITER);
    (*p_stats)[0][iter][0] = seq_time;
    std::cout << "Seq Time: " << seq_time << std::endl;

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        exercice1_omp(vec1,vec2,res1,data_size);
    }
    end = omp_get_wtime();
    omp_time = (end-start)/(MAX_ITER);
    (*p_stats)[0][iter][1] = omp_time;
    std::cout << "OMP Time: " << omp_time << " Acceleration Ratio: " << seq_time/omp_time  << std::endl << std::endl;

    (*p_stats)[0][iter][2] = 0;
    for(int i = 0; i< data_size; i++){
        if(res1[i] != res2[i]){
            std::cout << "Vector different" << std::endl;
        }
    }
    #endif

    double res3 = 0;
    double res4 = 0;
    double res5 = 0;
    #ifdef EX2
    /* Exercice 2 */
    std::cout << "Exercice 2:\n====================\n";
    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        res5 = exercice2_seq(vec1,vec2,data_size,false);
    }
    end = omp_get_wtime();
    seq_time = (end-start)/(MAX_ITER);
    (*p_stats)[1][iter][0] = seq_time;
    std::cout << "Seq Time: " << seq_time << std::endl;

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        res3 = exercice2_omp_no_red(vec1,vec2,data_size);
    }
    end = omp_get_wtime();
    omp_time = (end-start)/(MAX_ITER);
    (*p_stats)[1][iter][1] = omp_time;
    std::cout << "OMP Time ( no reduction ): " << omp_time << " Acceleration Ratio: " << seq_time/omp_time  << std::endl;

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        res4 = exercice2_omp_red(vec1,vec2,data_size,false,iter);
    }
    end = omp_get_wtime();
    omp_time = (end-start)/(MAX_ITER);
    (*p_stats)[1][iter][2] = omp_time;
    std::cout << "OMP Time ( with reduction ): " << omp_time << " Acceleration Ratio: " << seq_time/omp_time  << std::endl << std::endl;

    if(res3 != res5 || res4 != res5){
        std::cout << "Dot prod different" << std::endl;
    }
    #endif

    #ifdef EX3
    /* Exercice 3 */
    std::cout << "Exercice 3:\n====================\n";
    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        res5 = exercice2_seq(vec1,vec2,data_size,true);
    }
    end = omp_get_wtime();
    seq_time = (end-start)/(MAX_ITER);
    (*p_stats)[2][iter][0] = seq_time;
    std::cout << "Seq Time: " << seq_time << std::endl;

    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        res4 = exercice2_omp_red(vec1,vec2,data_size,true,iter);
    }
    end = omp_get_wtime();
    omp_time = (end-start)/(MAX_ITER);
    (*p_stats)[2][iter][1] = omp_time;
    std::cout << "OMP Time ( with reduction ): " << omp_time << " Acceleration Ratio: " << seq_time/omp_time  << std::endl << std::endl;

    (*p_stats)[2][iter][2] = 0;
    if(res4 != res5){
        std::cout << "Dot prod different" << std::endl;
    }
    #endif

    #ifdef EX5
    /* Exercice 5 */
    std::cout << "Exercice 5:\n====================\n";
    int counter1 = 0;
    int counter2 = 0;
    start = omp_get_wtime();
    for( int j = 0; j< MAX_ITER; j++){
        counter1 = 0;
        exercice5_seq(list, sublist2, &counter1, data_size);
    }
    end = omp_get_wtime();
    seq_time = (end-start)/(MAX_ITER);
    (*p_stats)[3][iter][0] = seq_time;
    std::cout << "Seq Time: " << seq_time << std::endl;

    start = omp_get_wtime();
    for( int j = 10; j< MAX_ITER; j++){
        counter2 = 0;
        exercice5_omp(list, sublist1, &counter2, n_threads, data_size);
    }
    end = omp_get_wtime();
    omp_time = (end-start)/(MAX_ITER);
    (*p_stats)[3][iter][1] = omp_time;
    std::cout << "OMP Time: " << omp_time << " Acceleration Ratio: " << seq_time/omp_time  << std::endl << std::endl;

    (*p_stats)[3][iter][2] = 0;
    if( counter1 != counter2 ){
        std::cout << "Sublist different" << std::endl;
    }
    #endif


    free(vec1);
    free(vec2);
    free(res1);
    free(res2);
    free(list);
    free(sublist1);
    free(sublist2);
    return 0;
}


void exercice1_omp(double* vec1, double* vec2, double* res, int size){
    #pragma omp parallel for
    for ( int i = 0; i < size; i++){
        res[i] = (vec1[i] + vec2[i])/2;
    }
}

void exercice1_seq(double* vec1, double* vec2, double* res, int size){
    for ( int i = 0; i < size; i++){
        res[i] = (vec1[i] + vec2[i])/2;
    }
}

double exercice2_omp_no_red(double* vec1, double* vec2, int size){
    double res[8] = {0,0,0,0,0,0,0,0};
    #pragma omp parallel for
    for ( int k = 0; k < 8; k++ ){
        for ( int i = 0; i < size/8; i++){
            res[k] += vec1[i+k*size/8] * vec2[i+k*size/8];
        }
    }
    return res[0]+res[1]+res[2]+res[3]+res[4]+res[5]+res[6]+res[7];
    
}

double exercice2_omp_red(double* vec1, double* vec2, int size, bool imbalanced,int iter){
    double res = 0;
    #pragma omp parallel
    {
        #pragma omp for reduction(+:res) schedule(auto)
        for ( int i = 0; i < size; i++){
            if(imbalanced == false)
                res += vec1[i] * vec2[i];
            else {
                if(i < size/20 || i > size-size/20)
                    res += vec1[i] * vec2[i];
            }
        }
    }
    return res;
}

double exercice2_seq(double* vec1, double* vec2, int size, bool imbalanced){
    double res = 0;
    for ( int i = 0; i < size; i++){
        if(imbalanced == false)
                res += vec1[i] * vec2[i];
        else {
            if((i < size/20) || (i > size-size/20)){
                res += vec1[i] * vec2[i];
            }
        }
    }
    return res;
}

void exercice4_omp(u_char **Source, u_char **Resultat, unsigned int height, unsigned int width){
    #pragma omp parallel for
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

void exercice4_seq(u_char **Source, u_char **Resultat, unsigned int height, unsigned int width){
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

void exercice5_seq(int* list, int* even_sublist, int* counter, int size){
    for(int i = 0; i < size; i++){
        if(list[i] % 2 == 0){
            even_sublist[*counter] = list[i]; 
            (*counter)++;
        }
    }
}

void exercice5_omp(int* list, int* even_sublist, int* counter, int n_threads, int size){
    *counter = 0;
    // We split the source list in n_threads batchs of equal sizes
    int* tmp_sublist = (int*)calloc(size, sizeof(int));
    int counters[n_threads];
    int batch_size = size/n_threads;

    // Each thread process its batch and store it in a reserved 
    // space of the destination list and increment a counter for each 
    // thread
    #pragma omp parallel for
    for( int i = 0; i < n_threads; i++){
        counters[i] = 0;
        for(int j = 0; j < batch_size; j++){
            if( list[i*batch_size + j] % 2 == 0 ){
                tmp_sublist[i*batch_size+j] = list[i*batch_size+j];
                counters[i]++;
            }
        }
    }

    // We merge the sublists into one
    for(int k=0; k < n_threads; k++){
        for(int l=0; l < counters[k]; l++){
            even_sublist[(*counter)+l] = tmp_sublist[k*batch_size+l]; 
        }
        (*counter)+= counters[k];
    }
    free(tmp_sublist);
}


double rand_double(){
    return (double)((rand() % 360) - 180.0);
}

int rand_int(){
    return (int)((rand() % 360) - 180);
}