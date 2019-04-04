#include "csv.h"

void create_stats_csv(double stats[4][POW_MAX][3]){

    for(int i = 0; i < 4; i++){ 
        #ifndef EX1
        if( i == 0 )
            continue;
        #endif
        #ifndef EX2
        if( i == 1 )
            continue;
        #endif
        #ifndef EX3
        if( i == 2 )
            continue;
        #endif
        #ifndef EX5
        if( i == 3 )
            continue;
        #endif

        printf("\n Creating Exercice%d.csv file", (i+1 == 4) ? i+2 : i+1 );
        FILE *fp;
        char filename[50];
        sprintf(filename, "Exercice%d.csv", (i+1 == 4) ? i+2 : i+1);

        fp=fopen(filename,"w+");

        fprintf(fp, "Data size;Seq;Para1;Para2\n");

        int iter = 0;
        for(int j = DIV_MAX; j >= DIV_MIN; j=j/2, iter++){
            fprintf(fp,"%d;", VEC_SIZE/j);
            fprintf(fp, "%0.14f;%0.14f;%0.14f;\n", stats[i][iter][0], stats[i][iter][1], stats[i][iter][2]);

        }
        fclose(fp);
        printf("\n %s file created\n\n", filename);
    }
}