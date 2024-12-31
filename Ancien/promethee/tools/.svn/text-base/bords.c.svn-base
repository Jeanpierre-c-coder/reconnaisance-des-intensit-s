/** ***********************************************************
\file  bords.c 
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
 Traitement des Bords : initialisation a zero de N lignes et colonnes ==> N = BORDS
 
Macro:
-none 

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

void bords(float *tab, int Bord, int nbpixel, int nx, int ny)
{
    int i, j;

    for (i = 0; i < Bord * nx; i++)
    {
        tab[i] = 0.;
        tab[nbpixel - i - 1] = 0.;
    }
    for (i = Bord; i < ny; i++)
    {
        for (j = 0; j < Bord; j++)
        {
            tab[i * nx + j] = 0.;
            tab[i * nx - 1 - j] = 0.;
        }
    }
}
