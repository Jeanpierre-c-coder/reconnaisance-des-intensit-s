/** ***********************************************************
\file  f_ptcg.c
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
-DEFAULT_THETA1
-DEFAULT_THETA2
-DEFAULT_TAILLEMSQ

Local variables:
-none

Global variables:
-none

Internal Tools:
-tools/include/recherche_pt_carac()
-tools/include/init_masque_pt_carac()

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

/*#define DEBUG*/
#include <libx.h>
#include <Struct/prom_images_struct.h>
#include <stdlib.h>
#include <string.h>

#include <Struct/convert.h>
#include "tools/include/macro.h"
#include <Kernel_Function/find_input_link.h>
#include <Kernel_Function/prom_getopt.h>
#include "tools/include/recherche_pt_carac.h"
#include "tools/include/init_masque_pt_carac.h"


typedef struct MyData{
  
  float theta1;
  float theta2;
  int   taille_msq;
  int   seuil;
  int   R_exclusion;
  int   mode_pano;
  int   mode_occlusion;
  float mode_minmax;
  int   mode_thread;
  float min_global_after_filtering;
  float max_global_after_filtering;
  float head;
  int InputGep;
} MyData;


void function_ptcg(int Gpe)
{
    MyData  *mydata   = NULL;
    char    *string   = NULL;
    char    param_link[256];
    
    prom_images_struct *rez_image;
    
    register int i, j;
    unsigned int nx, ny;
    unsigned int n;

      
    unsigned char *retine;
    int *retine_caract;
    float *im_tmp[2];
    float **masque_carac;
    
    int occlusion;
    int maxi;
    
  
    
#ifdef TIME_TRACE
    gettimeofday(&InputFunctionTimeTrace, (void *) NULL);
#endif
    
    dprints("====debut %s\n", __FUNCTION__);
    /* initialisation */
    if (def_groupe[Gpe].ext == NULL)
    {
      mydata = (MyData*)malloc(sizeof (MyData));
      if (mydata == NULL)
      {
        printf("pb malloc dans %s\n",def_groupe[Gpe].no_name);
        exit(0);
      } 
      
      mydata->theta2          = DEFAULT_THETA2;
      mydata->theta1          = DEFAULT_THETA1;
      mydata->seuil           = 0;
      mydata->taille_msq      = DEFAULT_TAILLEMSQ;
      mydata->R_exclusion     = 0;
      mydata->mode_pano       = 0;
      mydata->mode_minmax     = -1.;
      mydata->mode_occlusion  = 0;
      mydata->mode_thread     = 0;
      mydata->head            = 0;
      mydata->InputGep        = 0;
        
      /* recuperation des infos sur le gpe precedent */
      mydata->InputGep = liaison[find_input_link(Gpe, 0)].depart;
      string           = liaison[find_input_link(Gpe, 0)].nom;
      
      if (def_groupe[mydata->InputGep].ext == NULL)
      {
        printf("Gpe amonte avec ext nulle; pas de calcul de ptc\n");
        return;
      }

      /* alocation de memoire */  
      def_groupe[Gpe].ext = (void *) malloc(sizeof(prom_images_struct));
      if (def_groupe[Gpe].ext == NULL)
      {
        printf("ALLOCATION IMPOSSIBLE ...! \n");
        exit(-1);
      }
        
      printf("\n");
      /* recuperation d'info sur le lien : theta2, theta1, seuil, taille_msq, R_exclusion*/
     
      if (prom_getopt(string, "-T", param_link) == 2 || prom_getopt(string, "-t", param_link) == 2)
        {
            mydata->theta2 = atof(param_link);printf("theta2 = %f\n", mydata->theta2);
        }	
      
      if (prom_getopt(string, "-R", param_link) == 2 || prom_getopt(string, "-r", param_link) == 2)
        {
            mydata->theta1 = atof(param_link);        printf("theta1 = %f\n", mydata->theta1);
        }   
      if (prom_getopt(string, "-S", param_link) == 2 || prom_getopt(string, "-s", param_link) == 2)
        {
            mydata->seuil = atoi(param_link);         printf("Seuil = %d\n", mydata->seuil);
        }          
      if (prom_getopt(string, "-P", param_link) == 2 || prom_getopt(string, "-p", param_link) == 2)
        {
            mydata->taille_msq = atoi(param_link);    printf("Taille_msq = %d\n", mydata->taille_msq);
        }       
      if (prom_getopt(string, "-RE", param_link) == 2 || prom_getopt(string, "-re", param_link) == 2)
        {
            mydata->R_exclusion = atoi(param_link);   printf("Rayon exclusion = %d\n", mydata->R_exclusion);
        }
      if (prom_getopt(string, "-CP", param_link) == 1 || prom_getopt(string, "-cp", param_link) == 1)
        {
            mydata->mode_pano = 1;                    printf("Mode cam_pano actif\n");
        } 
       if (prom_getopt(string, "-V", param_link) == 1 || prom_getopt(string, "-v", param_link) == 1 )
        {
            mydata->mode_thread = 1;                  printf("Mode thread actif\n");
        }
       if (prom_getopt(string, "-MINMAX", param_link) == 2 || prom_getopt(string, "-minmax", param_link) == 2)
        {
            mydata->mode_minmax = atof(param_link);   printf("mode minmax activated: %f\n",mydata->mode_minmax);
        }
       if (prom_getopt(string, "-OCC", param_link) == 1 || prom_getopt(string, "-occ", param_link) == 1)
        {
            mydata->mode_occlusion = 1;               printf("mode occlusion\n");
        } 
          
       
      /* recuperation des infos sur la taille de l'image du groupe precedent et initialisation de l'image de ce groupe avec les infos recuperees */
      nx = ((prom_images_struct *) def_groupe[mydata->InputGep].ext)->sx;
      ((prom_images_struct *) def_groupe[Gpe].ext)->sx = nx;
      ny = ((prom_images_struct *) def_groupe[mydata->InputGep].ext)->sy;
      ((prom_images_struct *) def_groupe[Gpe].ext)->sy = ny;
      n = nx * ny;
      ((prom_images_struct *) def_groupe[Gpe].ext)->nb_band = ((prom_images_struct *) def_groupe[mydata->InputGep].ext)->nb_band;
      ((prom_images_struct *) def_groupe[Gpe].ext)->image_number = 1;
      
      printf ("%s Taille image %d x %d\n",__FUNCTION__,nx,ny);

      /* sauvgarde des infos trouves sur le lien */
      def_groupe[Gpe].data=mydata; 

         
      /* creation du masque DOG */
      ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[6] =(unsigned char *) init_masque_pt_carac(mydata->taille_msq, mydata->theta2, mydata->theta1);
      masque_carac = (float **) (((prom_images_struct *) def_groupe[Gpe].ext)->images_table[6]);

      /*Recherche valeur max pour ce filtre*/
      recherche_valeur_minmax(mydata->taille_msq,masque_carac,&mydata->min_global_after_filtering,&mydata->max_global_after_filtering);
      printf("Valeur minmax global apres filtrage= %f <-> %f \n",mydata->min_global_after_filtering,mydata->max_global_after_filtering);
      
      /* recuperation du pointeur de l'image d'entree */
      ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[4] = ((prom_images_struct *) def_groupe[mydata->InputGep].ext)->images_table[0];

      /* allocation de memoire */
      ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[0] =
          (unsigned char *) calloc(n, sizeof(char));
      /* attention, pointeurs vers des reels ou des entiers */
      ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[1] =(unsigned char *) calloc(n, sizeof(int));
      ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[2] =(unsigned char *) calloc(n, sizeof(float));
      ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[3] =(unsigned char *) calloc(n, sizeof(float));
      if ((((prom_images_struct *) def_groupe[Gpe].ext)->images_table[0] == NULL)
          || (((prom_images_struct *) def_groupe[Gpe].ext)->images_table[1] == NULL)
          || (((prom_images_struct *) def_groupe[Gpe].ext)->images_table[2] == NULL)
          || (((prom_images_struct *) def_groupe[Gpe].ext)->images_table[3] == NULL)
          || (((prom_images_struct *) def_groupe[Gpe].ext)->images_table[4] == NULL)
          || (((prom_images_struct *) def_groupe[Gpe].ext)->images_table[6] == NULL))
        {
          printf("ALLOCATION IMPOSSIBLE ...! \n");
          exit(-1);
        }

      rez_image = (prom_images_struct *) def_groupe[Gpe].ext;
    }
    else
    {
      rez_image = ((prom_images_struct *) def_groupe[Gpe].ext);
      nx = rez_image->sx;
      ny = rez_image->sy;
      n = nx * ny;     
      mydata = def_groupe[Gpe].data;  

	/*recupere le masque DOG*/
        masque_carac = (float **) rez_image->images_table[6];

	/*recupere l'image d'entree*/
        ((prom_images_struct *) def_groupe[Gpe].ext)->images_table[4] = ((prom_images_struct *) def_groupe[mydata->InputGep].ext)->images_table[0];
    }

    /* utilisation des variables locales */
    /* image de sortie */
    retine = (unsigned char *) rez_image->images_table[0];
    /*resultat d'extraction des points carcateristiques*/
    retine_caract = (int *) rez_image->images_table[1];
    /*apparament inutilise presentement...*/
    im_tmp[0] = (float *) rez_image->images_table[2];
    /*resultat de la dog*/
    im_tmp[1] = (float *) rez_image->images_table[3];
    /*masque de la dog*/
    masque_carac = (float **) rez_image->images_table[6];
    
    dprints("mode_minmax = %f \n",mydata->mode_minmax);

    if(mydata->mode_minmax<0) /*mode_minmax inactif : renormalisation par le max courant*/
    {
      /*if (mydata->mode_thread==1)*/
      /*    recherche_pt_carac_normale(rez_image->images_table[4], taille_msq, taille_msq, nx, ny,im_tmp,retine_caract, masque_carac, seuil); */
      recherche_pt_carac_thread(rez_image->images_table[4], mydata->taille_msq, mydata->taille_msq, nx, ny,im_tmp,retine_caract, masque_carac, mydata->seuil);
      /*else
        recherche_pt_carac(rez_image->images_table[4], mydata->taille_msq, mydata->taille_msq, nx, ny, im_tmp,retine_caract, masque_carac, mydata->seuil);*/
    }
    else if (mydata->mode_minmax>0) /*mode_minmax actif : le point le plus fort doit etre au moins de (vmax-vmin) / mode_minmax */
      recherche_pt_carac_fixed_minmax_thread(rez_image->images_table[4], mydata->taille_msq, mydata->taille_msq, nx, ny, im_tmp,
		                              retine_caract, masque_carac, mydata->seuil, mydata->min_global_after_filtering / mydata->mode_minmax ,mydata->max_global_after_filtering / mydata->mode_minmax );
    else  /*mode_minmax actif et automatique : le point le plus fort doit etre au moins de (vmax-vmin) / 255. */
      recherche_pt_carac_fixed_minmax(rez_image->images_table[4], mydata->taille_msq, mydata->taille_msq, nx, ny, im_tmp,
		                              retine_caract, masque_carac, mydata->seuil, mydata->min_global_after_filtering,mydata->max_global_after_filtering);
		
   maxi = n;
   for (i = maxi ;i--;)
        retine[i] = retine_caract[i] > 255 ? 255 : (retine_caract[i] < 0 ? 0 : (unsigned char)  retine_caract[i]);
    /*   retine[i]=255; */

    if ((2 * mydata->R_exclusion >= (int)nx) || (2 * mydata->R_exclusion >= (int)ny))
    {
        printf ("pb dans %s: aucun point de focalisation car R_exclusion trop grand\n", __FUNCTION__);
        exit(0);
    }
   
	/*Traitement rayon exclusion sur les bords droit et gauche */
        for (i = mydata->R_exclusion; i < ((int)ny - mydata->R_exclusion); i++)
        {
            for (j = mydata->R_exclusion;j--;)
                retine[nx * i + j] = 0;
            for (j = (int)nx - mydata->R_exclusion; j < (int)nx; j++)
                retine[nx * i + j] = 0;
        }

        if (mydata->mode_pano == 0)
        {
	    /*Traitement rayon exclusion sur les bords haut et bas */
            for (i = mydata->R_exclusion * nx; i--;)
            {
                retine[i] = 0;
            }
            for (i = ((int)ny - mydata->R_exclusion) * (int)nx; i < (int)ny * (int)nx; i++)
            {
                retine[i] = 0;
            }
        }
        else
        {
            for (j = nx; j--;)
            {
                for (i = mydata->R_exclusion + (int) ((45. / 240.) * ny); i--;)
                {
                    if (i < mydata->R_exclusion + (int) ((10. / 240.) * ny))
                    {
                        retine[nx * i + j] = 0;
                    }
                    else if ((j > (300. / 1540.) * nx)
                             && (j < ((545. / 1540.) * nx)))
                        retine[nx * i + j] = 0;
                    /*retine[nx*i+j]=0; */

                }

                for (i = ny - mydata->R_exclusion - (int) ((40. / 240.) * ny); i < (int)ny; i++)
                {
                    retine[nx * i + j] = 0;

                }
            }
        }
	
	/*if (mode_koala == 1)
        {
            for (i = 0; i < ny; i++)
            {
                for (j = R_exclusion; j < nx; j++)
                {
                    if ((j > 405) && (j < 445 + 2 * R_exclusion))
                        retine[i * nx + j] = 0;
                }
            }
        }*/
    

/*   Inhibition des points centraux: Non utilisé desormais
 if (milieu_inhibe == 1)
    {
        for (i = R_exclusion; i < (ny - R_exclusion); i++)
        {
            for (j = (nx + milieu) / 2 - R_exclusion;
                 j < (nx + milieu) / 2 + R_exclusion; j++)
                retine[nx * i + j] = 0;
        }
    }*/

    if (mydata->mode_occlusion == 1)
    {
        occlusion = mydata->head * (nx - 2 * mydata->R_exclusion) / 2. + mydata->R_exclusion;

        mydata->head = 0.;
        
        ((float *) (((prom_images_struct *) def_groupe[Gpe].ext)-> images_table[5]))[7] = mydata->head;
        for (i = occlusion;i < occlusion + (nx - 2 * mydata->R_exclusion) * (1. / 2.); i++)
            for (j = 0; j < (int)ny; j++)
            {
                retine[nx * j + i] = 0.;
            }
    }

#ifdef DEBUG
    printf("~~~~fin %s~~~~\n", __FUNCTION__);
#endif

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
    sprintf(MessageFunctionTimeTrace, "Time in function_ptcg\t%4ld.%06d\n",
            SecondesFunctionTimeTrace, MicroSecondesFunctionTimeTrace);
    affiche_message(MessageFunctionTimeTrace);
#endif
}
