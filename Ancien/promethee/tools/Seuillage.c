/** ***********************************************************
\file  Seuillage.c 
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
  Fonction de seuillage du gradient

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
#include <math.h>

void Seuillage(float Seuil, float *Psrce, unsigned char *Pdest, unsigned N)
{
    unsigned char Gr;
    unsigned long Compt;
    for (Compt = 0; Compt < N; Compt++)
    {
        Gr = (unsigned char) ceil(Psrce[Compt]);
        Pdest[Compt] = (Gr < (unsigned char) Seuil) ? 0 : 1;    /*modif de Braik B. 1->255 */
    }
}
