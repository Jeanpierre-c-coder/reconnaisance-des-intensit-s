/** ***********************************************************
\file  init_masque_pt_carac.c 
\brief 

Author: xxxxxxxx
Created: XX/XX/XXXX
Modified:
- author: C.Giovannangeli
- description: specific file creation
- date: 11/08/2004

Theoritical description:
 - \f$  LaTeX equation: none \f$  

Description: 
    Definit le masque utilise pour l'extraction de points carac 
        l       : taille du masque                              
        theta1  : coefficient de la gaussienne partie pos       
        theta2  : coeff de la 2emme gaussienne partie neg       
    Operateur forme par la difference de 2 gaussiennes          

Macro:
-Taille_Max_Tableau

Local variables:
-none

Global variables:
-none

Internal Tools:
-none

External Tools: 
-none

Links:
- type: algo / biological / neural
- description: none/ XXX
- input expected group: none/xxx
- where are the data?: none/xxx

Comments:

Known bugs: none (yet!)

Todo:see author for testing and commenting the function

http://www.doxygen.org
************************************************************/
#include <libx.h>
#include <stdlib.h>
#include <math.h>

#include "include/macro.h"

float **init_masque_pt_carac(int l, float theta2, float theta1)
{
    float **tableau;
    int i, j;
    float a1, a2, d, d1, d2, a3, a4;

    /* unused :
       int x,y,p,q,s,fin;
       float v,m;
     */

    if (l > Taille_Max_Tableau)
    {
        printf("Erreur : largeur du masque demandee trop grande\n");
        tableau = 0;
        return (tableau);
    }

    /*
       cree_tableau_2d(tableau,float,2*l,2*l); 
     */

    tableau = (float **) calloc(2 * l, sizeof(float *));
    if (tableau == NULL)
    {
        fprintf(stderr, "allocation memoire impossible -> ARRET\n");
        exit(-1);
    }
    for (i = 0; i < 2 * l; i++)
    {
        tableau[i] = (float *) calloc(2 * l, sizeof(float));
        if (tableau[i] == NULL)
        {
            fprintf(stderr, "allocation memoire impossible -> ARRET\n");
            exit(-1);
        }
    }

    a3 = (2. * theta1 * theta1);
    a4 = (2. * theta2 * theta2);
    a1 = 1. / (2. * M_PI * theta1 * theta1);    /*M.M. en 2D c'est ca le facteur de normalisation... */
    a2 = 1. / (2. * M_PI * theta2 * theta2);

    for (j = -l + 1; j < l; j++)
    {
        for (i = -l + 1; i < l; i++)
        {
            d = (float) (i * i + j * j);
            d1 = exp(-d / a3);
            d2 = exp(-d / a4);
            tableau[l + i][l + j] = ((a1 * d1 - a2 * d2));
        }
    }

    return (tableau);
}
