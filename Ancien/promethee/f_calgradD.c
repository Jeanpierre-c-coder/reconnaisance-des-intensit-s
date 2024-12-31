/** ***********************************************************
\ file  f_calgradD.c 
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

Macro:
-BORDS 

Local variables:
-float s_alpha;
-float seuil
-int Intensity,R,O,M,Q,C,L

Global variables:
-none

Internal Tools:
-mamphiJ()
-Seuillage()
-bords()
-filtre_ghv()
-fh_ghv()
-fv_ghv()

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
#include <Struct/prom_images_struct.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tools/include/macro.h"
#include "tools/include/local_var.h"
#include "tools/include/Seuillage.h"
#include "tools/include/bords.h"
/*#include "net_message_debug_dist.h"*/

/*#define DEBUG*/
void filtre_ghv(float *image, float alpha, int nx, int ny);
void fh_ghv(float *image, float alpha, int nx, int ny);
void fv_ghv(float *image, float alpha, int nx, int ny);
void mamphiJ(float *ik, float *in, unsigned Nx, unsigned Ny);

typedef struct data
{
  float s_alpha;
  float seuil;
  int Intensity;
  int R;
  int O;
  int M;
  int C;
  int Q;
  int L;
} MyData;

void filtre_ghv(float *image, float alpha, int nx, int ny)
{
    int ix, iy, pnt;
    float y, yy, yyy, x, xx;
    float a;

    a = exp(-alpha);
    for (iy = 0; iy < ny; iy++)
    {
        y = yy = yyy = 0.;
        x = xx = 0.;
        for (ix = 0; ix < nx; ix++)
        {
            pnt = iy * nx + ix;
            xx = x;
            x = image[pnt];
            yyy = yy;
            yy = y;
            y = (1 - a) * (1 - a) * (x + xx) + 2 * a * yy - a * a * yyy;
            image[pnt] = y;
        }

        y = yy = yyy = 0.;
        x = xx = 0.;
        for (ix = nx - 1; ix >= 0; ix--)
        {
            pnt = iy * nx + ix;
            xx = x;
            x = image[pnt];
            yyy = yy;
            yy = y;
            y = (1 - a * a) * x + (2 * a * yy - a * a * yyy);
            image[pnt] = y;
        }
    }

    for (ix = 0; ix < nx; ix++)
    {
        y = yy = yyy = 0.;
        x = xx = 0.;
        for (iy = 0; iy < ny; iy++)
        {
            pnt = iy * nx + ix;
            xx = x;
            x = image[pnt];
            yyy = yy;
            yy = y;
            y = (1 - a) * (1 - a) * (x + xx) + 2 * a * yy - a * a * yyy;
            image[pnt] = y;
        }

        y = yy = yyy = 0.;
        x = xx = 0.;
        for (iy = ny - 1; iy >= 0; iy--)
        {
            pnt = iy * nx + ix;
            xx = x;
            x = image[pnt];
            yyy = yy;
            yy = y;
            y = (1 - a * a) * x + (2 * a * yy - a * a * yyy);
            image[pnt] = (y / 4) * (1 - a) / (1 + a);
        }
    }
}

void fh_ghv(float *image, float alpha, int nx, int ny)
{
    int ix, iy, pnt;
    float y, yy, yyy, x, xx, xxx;
    float a;

    a = exp(-alpha);
    for (iy = 0; iy < ny; iy++)
    {
        y = yy = yyy = 0.;
        x = xx = 0.;
        for (ix = 0; ix < nx; ix++)
        {
            pnt = iy * nx + ix;
            xx = x;
            x = image[pnt];
            yyy = yy;
            yy = y;
            y = (1 - a) * (1 - a) * xx + 2 * a * yy - a * a * yyy;
            image[pnt] = y;
        }

        y = yy = yyy = 0.;
        x = xx = xxx = 0.;
        for (ix = nx - 1; ix >= 0; ix--)
        {
            pnt = iy * nx + ix;
            xxx = xx;
            xx = x;
            x = image[pnt];
            yyy = yy;
            yy = y;
            y = (1 - a * a) * (xxx - x) + (2 * a * yy - a * a * yyy);
            image[pnt] = y;
        }
    }
}

void fv_ghv(float *image, float alpha, int nx, int ny)
{
    int ix, iy, pnt;
    float y, yy, yyy, x, xx, xxx;
    float a;

    a = exp(-alpha);

    for (ix = 0; ix < nx; ix++)
    {
        y = yy = yyy = 0.;
        x = xx = 0.;
        for (iy = 0; iy < ny; iy++)
        {
            pnt = iy * nx + ix;
            xx = x;
            x = image[pnt];
            yyy = yy;
            yy = y;
            y = (1 - a) * (1 - a) * xx + 2 * a * yy - a * a * yyy;
            image[pnt] = y;
        }

        y = yy = yyy = 0.;
        x = xx = xxx = 0.;
        for (iy = ny - 1; iy >= 0; iy--)
        {
            pnt = iy * nx + ix;
            xxx = xx;
            xx = x;
            x = image[pnt];
            yyy = yy;
            yy = y;
            y = (1 - a * a) * (xxx - x) + (2 * a * yy - a * a * yyy);
            image[pnt] = y;
        }
    }
}

void mamphiJ(float *ik, float *in, unsigned Nx, unsigned Ny)
{
    int sign;
    unsigned indice, indice1, indice2, indice3, indice4;
    unsigned kn;
    register int i, j;
    float *a;
    float angle;
    float g1, g2, u, ux, uy;

    kn = Nx * Ny;
    if ((a = (float *) calloc(kn, sizeof(float))) == NULL)
    {
        printf("ALLOCATION IMPOSSIBLE ...! \n");
        exit(1);
    };

    /*
     * calcul de l'amplitude
     */
    for (i = 0; i < kn; ++i)
        a[i] = sqrt(ik[i] * ik[i] + in[i] * in[i]);
    /*
     * suppression de non maximum
     */
    for (i = 1; i <= Ny - 2; ++i)
        for (j = 1; j <= Nx - 2; ++j)
        {
            indice = i * Nx + j;
            if (isdiff(ik[indice], 0))
                angle = in[indice] / ik[indice];
            else
                angle = 2.;

            sign = 1;
            if (angle < 0)
                sign = -1;
            ux = ik[indice];
            if (ux < 0)
                ux = -ux;
            uy = in[indice];
            if (uy < 0)
                uy = -uy;
            ik[indice] = 0;
            switch (sign)
            {
            case -1:
                if (angle < -1)
                {
                    u = a[indice] * uy;
                    indice1 = (i - 1) * Nx + j + 1;
                    indice2 = indice1 - 1;
                    g1 = ux * a[indice1] + (uy - ux) * a[indice2];
                    if (u < g1)
                    {
                        ik[indice] = 0;
                        continue;
                    }
                    indice3 = (i + 1) * Nx + j - 1;
                    indice4 = indice3 + 1;
                    g2 = ux * a[indice3] + (uy - ux) * a[indice4];
                    if (u <= g2)
                    {
                        ik[indice] = 0;
                        continue;
                    }
                }
                else
                {
                    u = a[indice] * ux;
                    indice1 = (i - 1) * Nx + j + 1;
                    indice2 = indice1 + Nx;
                    g1 = uy * a[indice1] + (ux - uy) * a[indice2];
                    if (u < g1)
                    {
                        ik[indice] = 0;
                        continue;
                    }
                    indice3 = (i + 1) * Nx + j - 1;
                    indice4 = indice3 - Nx;
                    g2 = uy * a[indice3] + (ux - uy) * a[indice4];
                    if (u <= g2)
                    {
                        ik[indice] = 0;
                        continue;
                    }
                }
                break;
            case 1:
                if (angle >= 1)
                {
                    u = a[indice] * uy;
                    indice1 = (i + 1) * Nx + j + 1;
                    indice2 = indice1 - 1;
                    g1 = ux * a[indice1] + (uy - ux) * a[indice2];
                    if (u < g1)
                    {
                        ik[indice] = 0;
                        continue;
                    }
                    indice3 = (i - 1) * Nx + j - 1;
                    indice4 = indice3 + 1;
                    g2 = ux * a[indice3] + (uy - ux) * a[indice4];
                    if (u <= g2)
                    {
                        ik[indice] = 0;
                        continue;
                    }
                }
                else
                {
                    u = a[indice] * ux;
                    indice1 = (i - 1) * Nx + j - 1;
                    indice2 = indice1 + Nx;
                    g1 = uy * a[indice1] + (ux - uy) * a[indice2];
                    if (u < g1)
                    {
                        ik[indice] = 0;
                        continue;
                    }
                    indice3 = (i + 1) * Nx + j + 1;
                    indice4 = indice3 - Nx;
                    g2 = uy * a[indice3] + (ux - uy) * a[indice4];
                    if (u <= g2)
                    {
                        ik[indice] = 0;
                        continue;
                    }
                }
                break;
            }                   /*  fin du switch */

            ik[indice] = a[indice];
        }                       /* fin du for */

    for (j = 0; j < Nx; ++j)
    {
        ik[j] = 0;
        ik[Nx + j] = 0;
        indice = (Ny - 1) * Nx + j;
        ik[indice] = 0;
        indice -= Nx;
        ik[indice] = 0;
    }
    for (i = 0; i < Ny; ++i)
    {
        indice = i * Nx;
        ik[indice] = 0;
        ++indice;
        ik[indice] = 0;
        indice = (i + 1) * Nx - 1;
        ik[indice] = 0;
        --indice;
        ik[indice] = 0;
    }
    for (i = 0; i <= ((Ny - 1) * (Nx + 1)); ++i)
        in[i] = a[i];

    free(a);
    return;
}

void function_calgradD(int Gpe)
{
  
  char *string = NULL, *ST;
  unsigned char *im;
  float *lh, *lv, *tmp;
  prom_images_struct *rez_image;
  
  unsigned int i;
  unsigned int nx, ny;
  unsigned int n;
  int InputGep = 0;
  float D0 = 0., D1 = 0., S0 = 0., S1 = 0.;
  MyData *data=NULL;
#ifdef TIME_TRACE
  gettimeofday(&InputFunctionTimeTrace, (void *) NULL);
#endif
  
#ifdef DEBUG
  printf("begin f_calgradD\n");
#endif
  
  
  /**************************************************************************************/
  /********** Construction de la structure ext du groupe ********************************/
  /**************************************************************************************/
  /*    On recupere les parametre sur le lien entrant:  alpha = -Ax avec x in R+        */
  /*                                                    type = -Fx avec x in {i,c,q,m}  */
  /*    et les parametres optionnels:   lissage = -L                                    */
  /*                                    seuillage = -Sx avec x in R+                    */
  /*                                                                                    */
  /*    les parameter sont gardes dans ext->image_table[5]                              */
  /*    Le numero du groupe d'entree est garde dans ext->image_table[4]                 */
  /**************************************************************************************/
  


  
  if (def_groupe[Gpe].data == NULL)
    {
      
      data = (MyData *) malloc(sizeof(MyData));
      if (data == NULL)
	{
	  printf("error malloc in %s\n", __FUNCTION__);
	  exit(0);
	}

      /* recuperation des infos sur le gpe precedent */
      for (i = 0; i < nbre_liaison; i++)
	if ((liaison[i].arrivee == Gpe)
	    && (strcmp(liaison[i].nom, "synch") != 0))
	  {
	    InputGep = liaison[i].depart;
	    string = liaison[i].nom;
	    if (string[0] != '.')
	      break;
	  }
      
      if (def_groupe[InputGep].ext == NULL)
        {
	  printf("Gpe amonte avec ext nulle; pas de calcul de gradient\n");
	  return;
        }



      /* alocation de memoire */
      def_groupe[Gpe].ext =
	(prom_images_struct *) malloc(sizeof(prom_images_struct));
      if (def_groupe[Gpe].ext == NULL)
        {
	  printf("ALLOCATION IMPOSSIBLE ...! \n");
	  exit(-1);
        }
      
      /* recuperation d'info sur le lien */
      /* recuperation d's_alpha */
      printf("on recupere l info");
      ST = strstr(string, "-A");
      if ((ST != NULL))
        {
	  data->s_alpha = atof(&ST[2]);
#ifdef DEBUG
	  printf("s_alpha = %f\n", data->s_alpha);
#endif
        }
      else
        {
	  printf
	    ("Error in function_calgradDS,you must specified -Aalpha\n");
	  exit(-1);
        }
      /* recuperation type filtrage */
      data->Intensity = data->R = data->O = data->M = data->C = data->Q = 0;
      ST = strstr(string, "-F");
      if (ST != NULL)
	switch (ST[2])
	  {
	  case 'i':
#ifdef DEBUG
	    printf("Filtre i \n");
#endif
	    data->Intensity = 1;
	    break;
	  case 'c':
#ifdef DEBUG
	    printf("Filtre c\n");
#endif
	    data->C = 1;
	    break;
	  case 'q':
#ifdef DEBUG
	    printf("Filtre q\n");
#endif
	    data->Q = 1;
	    break;
	  case 'm':
#ifdef DEBUG
	    printf("Filtre m\n");
#endif
	    data->M = 1;
	    break;
	  default:
	    printf("Type inconnu \n\n");
	    printf("Exit in function_calgradDS, Gpe = %d\n", Gpe);
	    exit(-1);
	  }
      else
        {
	  printf("Error in function_calgradDS,you must specified -Ftype\n");
	  exit(-1);
        }
      /* recuperation info lissage */
      data->L = 1;
      ST = strstr(string, "-L");
      if (ST != NULL)
	data->L = 0;
      /* recuperation info sur le seuil */
      data->seuil = 0.0;
      ST = strstr(string, "-S");
      if (ST != NULL)
        {
	  data->seuil = atof(&ST[2]);
#ifdef DEBUG
	  printf("Seuil = %f\n", data->seuil);
#endif
        }
      
      /* recuperation des infos sur la taille */
      nx = ((prom_images_struct *) def_groupe[InputGep].ext)->sx;
      ((prom_images_struct *) def_groupe[Gpe].ext)->sx = nx;
      ny = ((prom_images_struct *) def_groupe[InputGep].ext)->sy;
      ((prom_images_struct *) def_groupe[Gpe].ext)->sy = ny;
      n = nx * ny;
      ((prom_images_struct *) def_groupe[Gpe].ext)->nb_band =
	((prom_images_struct *) def_groupe[InputGep].ext)->nb_band;
      ((prom_images_struct *) def_groupe[Gpe].ext)->image_number = 1;
      
      /* sauvgarde des infos trouves sur le lien */
      ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[5] =
	(unsigned char *) malloc(10 * sizeof(float));
      {
	float *temp =
	  (float *) (((prom_images_struct *) def_groupe[Gpe].ext)->
		     images_table[5]);
	temp[0] = (float) data->Intensity;
	temp[1] = (float) data->R;
	temp[2] = (float) data->O;
	temp[3] = (float) data->C;
	temp[4] = (float) data->Q;
	temp[5] = (float) data->L;
	temp[6] = (float) data->M;
	temp[7] = (float) data->s_alpha;
	temp[8] = (float) data->seuil;
      }
      


        /* allocation de memoire */
        ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[0] = (unsigned char *) malloc(n * sizeof(char));
        /* attention, pointeurs vers de reels */
        ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[1] = (unsigned char *) malloc(n * sizeof(float));
        ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[2] = (unsigned char *) malloc(n * sizeof(float));
        ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[3] = (unsigned char *) malloc(n * sizeof(float));
        ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[4] = (unsigned char *) malloc(1 * sizeof(int));
        if ((((prom_images_struct *) def_groupe[Gpe].ext)->images_table[0] == NULL) 
	    || (((prom_images_struct *) def_groupe[Gpe].ext)->images_table[1] == NULL)
            || (((prom_images_struct *) def_groupe[Gpe].ext)->images_table[2] == NULL)
            || (((prom_images_struct *) def_groupe[Gpe].ext)->images_table[3] == NULL)
            || (((prom_images_struct *) def_groupe[Gpe].ext)->images_table[4] == NULL))
        {
            printf("ALLOCATION IMPOSSIBLE ...! \n");
            exit(-1);
        }

        /* sauvgarde du numero du groupe amonte */
        *(int *) (((prom_images_struct *) def_groupe[Gpe].ext)->images_table[4]) = InputGep;

        rez_image = def_groupe[Gpe].ext;
 
      def_groupe[Gpe].data = (MyData *) data;
    }
  else
    {
      float *temp;
      data = (MyData *) (def_groupe[Gpe].data);
      rez_image = ((prom_images_struct *) def_groupe[Gpe].ext);
      nx = rez_image->sx;
      ny = rez_image->sy;
      n = nx * ny;
      InputGep =
	*(int *) (((prom_images_struct *) def_groupe[Gpe].ext)->
		  images_table[4]);
      temp = (float *) rez_image->images_table[5];
      data->Intensity = (unsigned char) temp[0];
      data->R = (unsigned char) temp[1];
      data->O = (unsigned char) temp[2];
      data->C = (unsigned char) temp[3];
      data->Q = (unsigned char) temp[4];
      data->L = (unsigned char) temp[5];
      data->M = (unsigned char) temp[6];
      data->s_alpha = temp[7];
      data->seuil = temp[8];
    }
#ifdef DEBUG
  printf("s_alpha=%f\n",data->s_alpha);
  printf("seuil=%f\n",data->seuil);
#endif
  /******************************************************************************************************************************/
  /************************ Traitement: calcul du gradinent  ********************************************************************/
  /******************************************************************************************************************************/
  /*    On dispose ici des parametre:   Intensity, C, Q, M. Un seul vaut 1, cela depend de l'otion -Fx avec x in {icqm}         */
  /*                                    R = O = 0.                                                                              */
  /*                                    L = 1 si option -L, et 0 sinon                                                          */
  /*                                    s_alpha = x venant de -Ax                                                               */
  /*                                    seuil = 0. ou x venant de l'otion -Sx                                                   */
  /******************************************************************************************************************************/


  /* utilisation des variables locales */
  /********************************************************************/
  /* on copie l'image d'entree dans   im=ext->image_table[0]          */
  /*                                  tmp=ext->image_table[3]         */
  /*                                  lh=ext->image_table[1]          */
  /*                                  lv=ext->image_table[2]          */
  /********************************************************************/
  
  
  im = (unsigned char *) rez_image->images_table[0];
  lh = (float *) rez_image->images_table[1];
  lv = (float *) rez_image->images_table[2];
  tmp = (float *) rez_image->images_table[3];
  
  
  memcpy(rez_image->images_table[0], ((prom_images_struct *) def_groupe[InputGep].ext)->images_table[0], n * sizeof(char));
  for (i = 0; i < n; i++)
    tmp[i] = lh[i] = lv[i] = (float) im[i];
  
  /**************************************/
  /* Lissage    si otpion -L            */
  /*            sinon traitement normal */
  /**************************************/
  if (data->L)
    {
      filtre_ghv(tmp, data->s_alpha, nx, ny);
      for (i = 1; i < n - nx; i++)
        {
	  D1 = tmp[i + nx] - tmp[i];
	  S1 = tmp[i] + tmp[i + nx];
	  lh[i - 1] = (D0 + D1);
	  lv[i - 1] = (S1 - S0);
	  S0 = S1;
	  D0 = D1;
        }
    }
  else
    { /*calcul du gradient en horizontal et vertical*/
      fh_ghv(lv, data->s_alpha, nx, ny);
      fv_ghv(lh, data->s_alpha, nx, ny);
    }
  /* O different de 0 touhours etant donne le code actuel */
  if (data->O == 1)
    {
      printf("\n\n Le resultat est de type float !!! A VOIR\n\n");
      exit(-1);
      /*strcpy(NomOUT, NomImage);strcat   (NomOUT, "-o.gdr");
	O_out(NomOUT,p_im,lh, lv); */
    }
  
  
  /***********************************************************************/
  /* amphi(lv,lh): Extraction des maxima locaux                 */
  /* ------------        en retour lv contient les maxima locaux,       */
  /*             lh contient la norme du gradient                       */
  /*********************************************************************/
  /* amphi(lv,lh,nx,ny); */
  

  /*calcul dans lv le gradient "aminci" (suivi du max) et dans lh la norme*/
  /*mamphiJ(lv, lh, nx, ny); */   /* original lounis */
  bords(lh, BORDS, n, nx, ny);
  bords(lv, BORDS, n, nx, ny);
  
  

  
  
  /* On effectue le traitement associe a l'otion -Fx qui definit le type I, C, Q, M */
  if (data->Intensity == 1)
    {
      /*strcpy(NomOUT, NomImage);strcat   (NomOUT, "-i.gdr");
	I_out(NomOUT,p_im,im,lh, lv); */
      for (i = 0; i < n; i++)
	/* a decommenter pour utiliserle code de Lounis*/
	/*im[i] = (unsigned char) floor(lh[i]);*/
	 im[i]= floor(sqrt(lh[i]*lh[i]+lv[i]*lv[i])) ;
    }
  if (data->C == 1)
    {
      /*strcpy(NomOUT, NomImage);strcat   (NomOUT, "-c.gdr");
	C_out(NomOUT,p_im,im,lv,seuil); */
      Seuillage(data->seuil, lv, im, n);
    }
  if (data->M == 1)
    {
      /*strcpy(NomOUT, NomImage);strcat   (NomOUT, "-m.gdr");
	M_out(NomOUT,p_im,im,lv); */
      for (i = 0; i < n; i++)
	im[i] = (unsigned char) floor(lv[i]);
    }




  /* R = 0 vu le code actuel */
  if (data->R == 1)
    {
      printf("\n\n Le resultat est de type float !!! A VOIR\n\n");
      exit(-1);
      /*strcpy(NomOUT, NomImage);strcat   (NomOUT, "-r.gdr");
	R_out(NomOUT,p_im,lh); */
    }
  
  
#ifdef TIME_TRACE
  gettimeofday(&OutputFunctionTimeTrace, (void *) NULL);
  if (OutputFunctionTimeTrace.tv_usec >= InputFunctionTimeTrace.tv_usec)
    {
      SecondesFunctionTimeTrace =
	OutputFunctionTimeTrace.tv_sec - InputFunctionTimeTrace.tv_sec;
      MicroSecondesFunctionTimeTrace =
	OutputFunctionTimeTrace.tv_usec - InputFunctionTimeTrace.tv_usec;
    }
  else
    {
      SecondesFunctionTimeTrace =
	OutputFunctionTimeTrace.tv_sec - InputFunctionTimeTrace.tv_sec -
	1;
      MicroSecondesFunctionTimeTrace =
	1000000 + OutputFunctionTimeTrace.tv_usec -
	InputFunctionTimeTrace.tv_usec;
    }
  sprintf(MessageFunctionTimeTrace,
	  "Time in function_calgradD\t%4ld.%06d\n",
	  SecondesFunctionTimeTrace, MicroSecondesFunctionTimeTrace);
  affiche_message(MessageFunctionTimeTrace);
#endif
  
#ifdef DEBUG
  printf("end f_calgradD\n");
#endif
}
