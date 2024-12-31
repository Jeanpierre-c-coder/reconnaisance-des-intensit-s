/** ***********************************************************
\file  recherche_pt_carac.c 
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
   Recherche les points caracteristiques en utilisant le masque     
   * forme par la diff de 2 gaussiennes                             
   * effectue une convolution puis la recherche de max locaux       
   * les max locaux sont considere comme des points carac           
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

/*#define DEBUG */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h> 
#include "net_message_debug_dist.h"


extern void gestion_mask_signaux();

void recherche_pt_carac_lent(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil)
           /* tableaux de flottants intermediaires   */
{
    int x, y, i, j, p, q;
#ifdef VAR_2
    int qq, qqq, qqqq;
#endif
    int nbr;
    float v, vv, valeur_max, moyen;
    float rapport;
    float v1, seuil_float = 0.;

    x = xmax * ymax;
    for (p = 0; p < x; p++)
        im_fl[1][p] = 0.;

    valeur_max = 0.;

/* ImageToFileLena(im_contour,xmax,ymax,"test1.lena"); */

    seuil_float = ((float) seuil) / 255.;   /*Le seuil est mis entre 0 t 1 */

#define VAR_1

#ifdef VAR_ORIG
    for (x = l; x < xmax - l; x++)
        for (y = l; y < ymax - l; y++)
        {
            p = x + xmax * y;
            v1 = (float) im_contour[p] / 255.0;
            if (im_contour[p] > 0)
                for (j = -l + 1; j < l; j++)
                    for (i = -l + 1; i < l; i++)
                    {
                        v = tableau[l + i][l + j];
                        q = p + i + xmax * j;
                        im_fl[1][q] = im_fl[1][q] + (v * (float) im_contour[p] / 255.0);
                        if (im_fl[1][q] > valeur_max)
                            valeur_max = im_fl[1][q];
                    }
        }
#endif

#ifdef VAR_1
    for (x = l; x < xmax - l; x++)
        for (y = l; y < ymax - l; y++)
        {
            p = x + xmax * y;
            v1 = (float) im_contour[p] / 255.0;
            if (v1 > seuil_float)
                for (j = -l + 1; j < l; j++)
                    for (i = -l + 1; i < l; i++)
                    {
                        v = tableau[l + i][l + j];
                        /* if( v > 1e-10) */
                        {
                            q = p + i + xmax * j;
                            im_fl[1][q] = im_fl[1][q] + (v * v1);
                            if (im_fl[1][q] > valeur_max)
                                valeur_max = im_fl[1][q];
                        }
                    }
        }
#endif



#ifdef VAR_2
    for (x = l; x < xmax - l; x++)
        for (y = l; y < ymax - l; y++)
        {
            p = x + xmax * y;
            v1 = (float) im_contour[p] / 255.0;
            if (im_contour[p] > 0)
            {
                for (j = -l + 1; j < 0; j++)
                    for (i = -l + 1; i < 0; i++)
                    {
                        v = v1 * tableau[l + i][l + j];
                        /* if( v > 1e-10) */
                        {
                            q = p + i + xmax * j;
                            qq = p - i + xmax * j;
                            qqq = p + i - xmax * j;
                            qqqq = p - i - xmax * j;


                            im_fl[1][q] = im_fl[1][q] + v;
                            if (im_fl[1][q] > valeur_max)
                                valeur_max = im_fl[1][q];

                            im_fl[1][qq] = im_fl[1][qq] + v;
                            if (im_fl[1][qq] > valeur_max)
                                valeur_max = im_fl[1][qq];

                            im_fl[1][qqq] = im_fl[1][qqq] + v;
                            if (im_fl[1][qqq] > valeur_max)
                                valeur_max = im_fl[1][qqq];

                            im_fl[1][qqqq] = im_fl[1][qqqq] + v;
                            if (im_fl[1][qqqq] > valeur_max)
                                valeur_max = im_fl[1][qqqq];
                        }
                    }
                v = v1 * tableau[l][l];
                im_fl[1][p] = im_fl[1][p] + v;
                if (im_fl[1][p] > valeur_max)
                    valeur_max = im_fl[1][p];
            }
        }
#endif

    printf("valeur max = %f \n",valeur_max);

    rapport = 256. / valeur_max;

    nbr = xmax * ymax;

    for (p = 0; p < nbr; p++)
        im_fl[0][p] = im_fl[1][p];




    v = ((2 * l - 1) * (2 * l - 1) - 1);
    vv = 0.;


/* la boucle ne commence pas a 0 et ne finie pas a nbr pour ne pas sortir du tableau */
    for (p = (la - 1) * (1 + xmax); p < (nbr - ((1 + xmax) * (la - 1))); p++)
    {
        for (j = 1 - la; j < la; j++)
            for (i = 1 - la; i < la; i++)
            {
                q = p + i + xmax * j;
                if (im_fl[1][q] > 0)
                    vv += im_fl[1][q];

            }
        im_fl[0][p] *= v;
        im_fl[0][p] -= vv;
        im_fl[0][p] = 1 / (1 + exp(-im_fl[0][p]));
        vv = 0;
    }

    vv = 0;

    for (p = 0; p < nbr; p++)
        vv += im_fl[0][p];

    vv = vv / nbr;

    for (p = 0; p < nbr; p++)
        im_fl[0][p] = 3 / (1 + exp(-im_fl[0][p] - vv));


    /*la=2*la; */
/* la boucle ne commence pas a 0 et ne finie pas a nbr pour ne pas sortir du tableau */
    for (p = (la - 1) * (1 + xmax); p < (nbr - ((1 + xmax) * (la - 1))); p++)
    {
        for (j = 1 - la; j < la; j++)
            for (i = 1 - la; i < la; i++)
            {
                q = p + i + xmax * j;
                if ((i != 0 || j != 0) && im_fl[1][q] >= im_fl[1][p])
                {
                    im_pt_carac[p] = 0;
                    goto finie2;
                }
            }
        v = im_fl[1][p] * rapport;
        if (v < 0.)
            im_pt_carac[p] = 0;
        else if (v > 255.)
            im_pt_carac[p] = 255;
        else
            im_pt_carac[p] = (int) v;
        finie2:;
    }




    v = 0;
    y = 0;
    moyen = 0;
    i = 0;
    for (p = l; p < nbr; p++)
    {
        im_fl[1][p] = im_fl[1][p] * rapport;
        if (im_fl[1][p] > 0)
        {
            moyen += im_fl[1][p];
            i++;
        }
        if (im_pt_carac[p] != 0)
        {
            v += im_pt_carac[p];
            y++;
        }
    }
}

void competition_sur_voisinage(float valeur_max, int l, int la, int xmax, int ymax, float *im_fl1, int *im_pt_carac)
{
  int p,p1,p2,i,j,qi,j2;
  int nbr;
  float v,v1,rapport;
  int p3;
  rapport = 256. / valeur_max;
  nbr = xmax * ymax;
  l=l;
  for (p2 = (la - 1); p2 < ymax - (la - 1); p2++)
  {
    p3=p2*xmax;
    for(p1 = (la - 1); p1 < xmax - (la - 1); p1++)
    {
      p=p1+p3;
      v1=im_fl1[p];
      if(v1>0. && im_pt_carac[p]>=0)
      {
        for (j = 1 - la/2; j < la/2; j++)
        {
          j2=p+xmax * j;
          for (i = 1 - la/2; i < la/2; i++)
          {
            qi = j2+ i ;
            if (i | j)
            {
              if (im_fl1[qi] >= v1)
              {
                im_pt_carac[p] = 0;
                goto finie2;
              }
              else
              {
                  im_pt_carac[qi]=-1; /* un point dans le voisinage plus faible on le met a -1 pour accelerer la suite des calculs */
              }
            }
           }
          }
          v = v1 * rapport;
          im_pt_carac[p] = (int) v;
       }
      else im_pt_carac[p] = 0;
      finie2:;
     }
    }
    
}  

void no_competition_sur_voisinage(float valeur_max, int l, int la, int xmax, int ymax, float *im_fl1, int *im_pt_carac)
{
  int p,p1,p2;
  int nbr;
  float v,v1,rapport;
  int p3;
  rapport = 256. / valeur_max;
  nbr = xmax * ymax;
  
  for (p2 = (la - 1); p2 < ymax - (la - 1); p2++)
  {
    p3=p2*xmax;
    for(p1 = (l - 1); p1 < xmax - (l - 1); p1++)
    {
      p=p1+p3;
      v1=im_fl1[p];
      if(v1>0. && im_pt_carac[p]>=0)
      {
          v = v1 * rapport;
          im_pt_carac[p] = (int) v;
       }
      else im_pt_carac[p] = 0;
     }
  }
    
}  

void competition_sur_voisinage_pondere(float valeur_max,  int la, int xmax, int ymax, float *im_fl1, int *im_pt_carac)
{
  int p,i,j,qi,j2;
  int nbr;
  float v,v1,rapport;
  float v2,v3,ponderation;

  rapport = 256. / valeur_max;
  nbr = xmax * ymax;
  ponderation = 3./(la*(la-1)*4); /*facteur de ponderation*/
                                  /*important: difficile a manipuler*/
  
  
  for (p = (la - 1) * (1 + xmax); p < (nbr - ((1 + xmax) * (la - 1))); p++)
    {
      v2=v1=im_fl1[p];
      if(v1>0.)
	{
       	  for (j = 1 - la; j < la; j++)
	    {
	      j2=p+xmax * j;
	      for (i = 1 - la; i < la; i++)
		{
		  qi = j2+ i ;
		  if (i | j)
		    {
		      v3=im_fl1[qi];
		      if (v3 >= v1) 
			{ 
			  im_pt_carac[p] = 0;
			  goto finie2;
			}
		      else if (v3 > 0) 
			v2=v2-v3*ponderation; /*retire la moyenne*/
		      
		    }
		}
	    }
	  v = v2 * rapport; /*les points ne sont plus normalises entre 0 et 255*/
	  if (v<0) v=0;
	  im_pt_carac[p] = (int) v;
	}
      else im_pt_carac[p] = 0;
    finie2:;
    }
}  
   
void recherche_valeur_minmax(int l /*taille_masque*/ , float ** tableau /*masque*/, float * valeur_min, float * valeur_max /*valeur_resultat*/)
{
  int j,i;
  float v1;
  float v,zM=0,zm=0;
  
  *valeur_max=-100.;
  for (j = 0; j < 2*l-1; j++)
  {
    for (i = 0 ; i < 2*l-1; i++)
	  {
	    v = tableau[i][j]; 
	    
	    if(v>0)   v1=1.;  else    v1=0.;
	    zM+=  (v * v1);
	    
	    if(v>0)   v1=0.;  else    v1=1.;
	    zm+=  (v * v1);
	    /*  else if ( z < valeur_min)  valeur_min = z;*/
	  }
  }
  *valeur_max = zM;
  *valeur_min = zm;
}

inline void  analyse_un_point(unsigned char *p_contour,float  ech_gris, float seuil_float, float *p2, int xmax, int l, float **tableau, float *valeur_max)
{
  int j,i;
  float v1;
  float v,z;
  float *tab2, *q;

  v1 = (float) (*p_contour) * ech_gris;
  if (v1 > seuil_float)
    for (j = -l + 1; j < l; j++)
      {
	   q = p2 - l + 1 + xmax * j;
	   tab2=&(tableau[l + j][1]);
	   for (i = -l + 1; i < l; i++)
	    {
	     v = *tab2; 
	     z=(*q) +=  (v * v1);
	     if ( z > *valeur_max)  *valeur_max = z;
	     /*  else if ( z < valeur_min)  valeur_min = z;*/
	     q++; tab2++;
	    }
      }
}


void recherche_pt_carac_fixed_minmax(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil,float vmin,float vmax)
           /* tableaux de flottants intermediaires   */
{
  int x, y,  p, j2;

  float valeur_max, ech_gris, seuil_float = 0.;
  float *p2, *im_fl1;
  unsigned char *p_contour;

  im_fl1=im_fl[1];

  x = xmax * ymax; 
  /*  for (p = 0; p < x; p++)
      im_fl[1][p] = 0.;*/

  memset(im_fl1,0,x*sizeof(float)); /* ne marche que si 0. est 4 0 char OK norme IEEE mais sinon... */
  memset(im_pt_carac,0,x*sizeof(int));
  valeur_max = 0.;

  /* ImageToFileLena(im_contour,xmax,ymax,"test1.lena"); */

  seuil_float = ((float) seuil) / 255.;   /*Le seuil est mis entre 0 t 1 */
  ech_gris=1/ 255.0;

  /* cette solution gain +0.7 frame/s sur 640x480 on est a 8.2 f/s */
 
  for (y = l; y < ymax - l; y++)  
    {
      j2= xmax * y;
      p = l + j2;
      p2=im_fl1+p;
      p_contour=&(im_contour[p]);
      for (x = l; x < xmax - l; x++)
	{
	  /*  for(m=0;m<NB_ITER_FIXE;m++)*/
	  {
	    analyse_un_point(p_contour, ech_gris, seuil_float, p2, xmax, l, tableau, &valeur_max);
	    p2++;p_contour++;
	  }
	}
    }
  /*  printf("M=%f   m=%f \n",valeur_max,(vmax-vmin)/127.);*/
  /* la boucle ne commence pas a 0 et ne finie pas a nbr pour ne pas sortir du tableau */
 
/*1 car reetalonnage juste avant*/
  if(valeur_max<(vmax-vmin)/127.)
    memset(im_pt_carac,0,x*sizeof(int));
  else
    competition_sur_voisinage(valeur_max, l, la, xmax, ymax, im_fl1, im_pt_carac); 

/*Enfin, on renormalise entre le min et le max */
  



  /*    enregistre_image_int(im_pt_carac,xmax,ymax,"carac.lena");
	printf("modif extraction pt carac pour regarder diffusion \n");
	if(v !=0)printf("%d points caracteristiques    %f valeur moyene\n\n",y,(v/y));
	if(i != 0)printf("valeur moyene du la diffusion  %f %d points\n\n",moyen/i,i);
  */
}

void recherche_pt_carac(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil)
           /* tableaux de flottants intermediaires*/
{
  int x, y,  p, j2;


  float valeur_max, ech_gris, seuil_float = 0.;
  float *p2, *im_fl1;
  unsigned char *p_contour;

   
  im_fl1=im_fl[1];

  x = xmax * ymax; 


  memset(im_fl1,0,x*sizeof(float)); /* ne marche que si 0. est 4 0 char OK norme IEEE mais sinon... */
  memset(im_pt_carac,0,x*sizeof(float));
  valeur_max = 0.;

  /* ImageToFileLena(im_contour,xmax,ymax,"test1.lena"); */

  seuil_float = ((float) seuil) / 255.;   /*Le seuil est mis entre 0 t 1 */
  ech_gris=1/ 255.0;

  /* cette solution gain +0.7 frame/s sur 640x480 on est a 8.2 f/s */
  for (y = l; y < ymax - l; y++)  
	{
    j2= xmax * y; /*selection de la ligne*/
    p = l + j2;   /*position au premier point atteignable de la ligne courante*/
    p2=im_fl1+p;
    p_contour=&(im_contour[p]); /*se postionne sur l'image du gradient*/
    for (x = l; x < xmax - l; x++)
    {
      analyse_un_point(p_contour, ech_gris, seuil_float, p2, xmax, l, tableau, &valeur_max);
	    p2++;
      p_contour++; /* avance tout au long de la ligne*/
    }
  }

  /* la boucle ne commence pas a 0 et ne finie pas a nbr pour ne pas sortir du tableau */
 
  competition_sur_voisinage(valeur_max, l, la, xmax, ymax, im_fl1, im_pt_carac); 

  /*    enregistre_image_int(im_pt_carac,xmax,ymax,"carac.lena");
	printf("modif extraction pt carac pour regarder diffusion \n");
	if(v !=0)printf("%d points caracteristiques    %f valeur moyene\n\n",y,(v/y));
	if(i != 0)printf("valeur moyene du la diffusion  %f %d points\n\n",moyen/i,i);
  */
}


void recherche_pt_carac_normale(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil)
           /* tableaux de flottants intermediaires   */
{
  int x, y,  p, j2;
  int i,j,q;
  float v;

  float valeur_max, ech_gris, seuil_float = 0.;
  float *im_fl1;

  im_fl1=im_fl[1];

  x = xmax * ymax; 

  memset(im_fl1,0,x*sizeof(float)); /* ne marche que si 0. est 4 0 char OK norme IEEE mais sinon... */
  memset(im_pt_carac,0,x*sizeof(float));
  valeur_max = 0.;
  seuil_float = ((float) seuil) / 255.;   /*Le seuil est mis entre 0 t 1 */
  ech_gris=1/ 255.0;
  
  for(y = l-1; y <= ymax - l; y++)
  {
    
    for(x = l-1; x <= xmax - l; x++)
    {
      j2= xmax * y;     /*selection de la ligne*/    
      p = x + j2;     /*selection de la colonne -le point central- */
     
      for(j=-la+1;j<la;j++)
      {
        for(i=-l+1;i<l;i++)
        {
          q = p + i + xmax * j;   /*q represente  les positions de tous les point ou sera enregistré le resultat de la convolution */
          v=(float) im_contour[q];  /*se positionner sur l'image*/

         /* if(v>seuil) */    /*ne convoluer que les contours a forte intensité*/
          {
          im_fl1[p] = im_fl1[p] + (tableau[l+i][la+j] * v * ech_gris); /*Convolution*/
          }
        }
      }
      if(im_fl1[p]>valeur_max) valeur_max=im_fl1[p];	
      
    
    }
  }

  /* la boucle ne commence pas a 0 et ne finie pas a nbr pour ne pas sortir du tableau */
 
  competition_sur_voisinage(valeur_max, l, la, xmax, ymax, im_fl1, im_pt_carac);
 /* no_competition_sur_voisinage(valeur_max, l, la, xmax, ymax, im_fl1, im_pt_carac);*/

  /*    enregistre_image_int(im_pt_carac,xmax,ymax,"carac.lena");
	printf("modif extraction pt carac pour regarder diffusion \n");
	if(v !=0)printf("%d points caracteristiques    %f valeur moyene\n\n",y,(v/y));
	if(i != 0)printf("valeur moyene du la diffusion  %f %d points\n\n",moyen/i,i);
  */
}


typedef struct type_arg_pt_carac
{
  int y;
  int nb_bandes;
  unsigned char *im_contour;
  int l;
  int la;
  int xmax;
  int ymax; 
  float **im_fl; 
  int *im_pt_carac;
  float **tableau;
  int seuil;
  float valeur_max;
  int no_thread;

} type_arg_pt_carac;

void recherche_pt_carac_normale_thread(void *arg)
           /* tableaux de flottants intermediaires   */
{
  int x, y,  p, j2;
  int i,j,q;
  float v;

  float valeur_max, ech_gris, seuil_float = 0.;
  float *im_fl1;
  unsigned char *im_contour; int l; int la; int xmax; int ymax; float **im_fl; int *im_pt_carac; float **tableau; int seuil;
  type_arg_pt_carac *my_arg;

  gestion_mask_signaux();

  my_arg=( type_arg_pt_carac*) arg;
  y=my_arg->y; im_contour=my_arg->im_contour;
  l=my_arg->l; la=my_arg->la; xmax=my_arg->xmax; ymax=my_arg->ymax; im_fl=my_arg->im_fl; im_pt_carac=my_arg->im_pt_carac;
  tableau=my_arg->tableau; seuil=my_arg->seuil;

  im_fl1=im_fl[1];
  valeur_max = 0.;
  seuil_float = ((float) seuil) / 255.;   /*Le seuil est mis entre 0 t 1 */
  ech_gris=1/ 255.0;

    j2= xmax * y;     /*selection de la ligne*/    
    for(x = l-1; x <= xmax - l; x++)
    {
        p = x + j2;     /*selection de la colonne -le point central- */

      for(j=-la+1;j<la;j++)
      {
        for(i=-l+1;i<l;i++)
        {
          q = p + i + xmax * j;   /*q represente  les positions de tous les point ou sera enregistré le resultat de la convolution */
          v=(float) im_contour[q];  /*se positionner sur l'image*/


          im_fl1[p] = im_fl1[p] + (tableau[l+i][la+j] * v * ech_gris); /*Convolution*/
        }
      }
      if(im_fl1[p]>valeur_max) valeur_max=im_fl1[p];
    }

  my_arg->valeur_max=valeur_max;

}

/* calcule l'effet de la colonne y sur le voisinage [-l, +l] */
/* contribution du filtre tableau */
void recherche_pt_carac_ligne(void *arg)
/*(int y, unsigned char *im_contour, int l, int la, int xmax, int ymax, float **im_fl, int *im_pt_carac, float **tableau, int seuil)*/
/* tableaux de flottants intermediaires   */
{
  int x, i, j, p;

  float *p2,*q;
  int j2,j3;

  float v, valeur_max;
  float ech_gris;
  float z;
  float v1, seuil_float = 0.;
  float *im_fl1,*tab2;
  unsigned char *p_contour;
  int y; unsigned char *im_contour; int l; int la; int xmax; int ymax; float **im_fl; int *im_pt_carac; float **tableau; int seuil;
  type_arg_pt_carac *my_arg;
  
  
  gestion_mask_signaux();

  my_arg=( type_arg_pt_carac*) arg;
  y=my_arg->y; im_contour=my_arg->im_contour;
  l=my_arg->l; la=my_arg->la; xmax=my_arg->xmax; ymax=my_arg->ymax; im_fl=my_arg->im_fl; im_pt_carac=my_arg->im_pt_carac;
  tableau=my_arg->tableau; seuil=my_arg->seuil;

  im_fl1=im_fl[1];

  valeur_max = 0.;

  seuil_float = ((float) seuil) / 255.;   /*Le seuil est mis entre 0 t 1 */
  ech_gris=1/ 255.0;

  j2= xmax * y;
  p = l + j2;
  p2=im_fl1+p;
  p_contour=&(im_contour[p]);
  dprints("%s no_thread %d, y=%d  xmax=%d \n",__FUNCTION__,my_arg->no_thread,y,xmax);
  for (x = l; x < xmax - l; x++)
    {
      v1 = (float) (*p_contour) * ech_gris;
     /* v1 = (float)im_contour[j2+x]*ech_gris;*/
      if (v1 > seuil_float)
      for (j = -l + 1; j < l; j++)
      {
        j3 = xmax * j;
        /*q= & im_fl1[p-l+1+xmax*j];  */ 
        q = p2  -l + 1 + j3;  
        tab2=&(tableau[l + j][1]);
	      for (i = l+l-1; i --; ) 
       /* for (i = -l + 1; i < l; i++)*/
	      {
          v = *tab2; /*tableau[l + j][l + i];*/
         /* printf("x=%d y=%d  i=%d j=%d xmax=%d \n",x,y,i,j,xmax);*/
          z = (*q) +=  (v * v1); /* z= im_fl1[j2+x+i+xmax*j] = im_fl1[j2+x+i+xmax*j]+ tableau[l + j][l + i]*v1;*/
          if ( z > valeur_max)  valeur_max = z;
          q++; tab2++;
          
	      }
      }
    p2++;p_contour++;
    /*printf ("val max = %f, a la ligne : %d colonne : %d \n",valeur_max,y,x);*/
    }

  my_arg->valeur_max=valeur_max;
  /* pthread_exit(NULL);*/
}

/* coeur de la fonction sans optim */
/*
   for (x = l; x < xmax - l; x++)
    {
      v1 = (float)im_contour[j2+x]*ech_gris;
      if (v1 > seuil_float)
      for (j = -l + 1; j < l; j++)
      {
        j3 = xmax * j;
        tab2=&(tableau[l + j][1]);
        for (i = -l + 1; i < l; i++)
	      {
          z= im_fl1[j2+x+i+xmax*j] = im_fl1[j2+x+i+xmax*j]+ tableau[l + j][l + i]*v1;
          if ( z > valeur_max)  valeur_max = z;
	      }
      }
    }
*/





void *recherche_pt_carac_nlignes(void *arg)
{
  type_arg_pt_carac *my_arg;
  int y,k,nbre,l,i;
  float vmax;
  float valeur_max;
  pthread_mutex_t mut_bandes [1024] ;

  
  valeur_max=0.;
  my_arg=( type_arg_pt_carac*) arg;
  y=my_arg->y; 
  nbre=my_arg->nb_bandes;
  l = my_arg->l;
  
  dprints("%s thread %d y=%d, nbre=%d \n",__FUNCTION__,my_arg->no_thread,y,nbre);
  
  for (i = y-l+1;i<=y+l;i++)
    {
      pthread_mutex_lock (&mut_bandes[i]);
      /*dprints("thread %d lock=%d\n",my_arg->no_thread,i);*/
    }
  for(k=y;k<y+nbre;k++)
    {
      /*printf ("ligne : %d\tthread n° : %d\n",y,my_arg->no_thread);*/
      my_arg->y=k; /* attention y est maintenant le No de la ligne courante */
      /*Mutex*/

     recherche_pt_carac_ligne(my_arg);  /* lance le calcul a partir de la ligne k */
     pthread_mutex_unlock (&mut_bandes[k-l+1]); /*dprints ("thread %d unlocked ligne %d \n",my_arg->no_thread,k-l+1);*/
     pthread_mutex_lock (&mut_bandes[k+l+1]); /*dprints ("thread %d  locked ligne %d \n",my_arg->no_thread,k+l+1);*/
     /*recherche_pt_carac_normale_thread(my_arg );*/
       
      
      vmax=  my_arg->valeur_max;
      if(vmax>valeur_max) valeur_max=vmax;
    }
    for (i = k-l+1;i<=k+l;i++)
    {
      pthread_mutex_unlock (&mut_bandes[i]);
     /* dprints("thread %d lock=%d\n",my_arg->no_thread,i);*/
    }
  my_arg->valeur_max=valeur_max;
  return NULL;
}

#define NB_MAX_THREADS 8000
#define NB_COL_THREAD 10

void convol_thread(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil, float *valeur_max)
/* tableaux de flottants intermediaires   */
{
  int x, y, n, j;

  float vmax;
  float ech_gris;
  float seuil_float = 0.;
  float *im_fl1;
  type_arg_pt_carac arg[NB_MAX_THREADS];
  pthread_t un_thread[NB_MAX_THREADS];
  void *resultat;
  int res, nb_threads;
  int nbre;
 
  im_fl1=im_fl[1];
  x = xmax * ymax; 
  memset(im_fl1,0,x*sizeof(float)); /* ne marche que si 0. est 4 0 char OK norme IEEE mais sinon... */

  *valeur_max = 0.;
  seuil_float = ((float) seuil) / 255.;   /*Le seuil est mis entre 0 t 1 */
  ech_gris=1/ 255.0;

  nb_threads=ymax/(2*l); /*nombre de thread*/
  nbre=ymax/nb_threads ; /*nbre de lignes par bande*/
  
  dprints("nombre de threads a lancer = %d ymax=%d l=%d \n",nb_threads,ymax,l);
/*  for (y = l+y0; y < ymax - (2*l+1)*nbre; y=y+nbre*(2*l+1))*/
  /*for (y = l+y0; y < ymax - l; y=y+(2*l+1)) */
  /*for(n=nb_threads-1;n--;) *//*boucle inversee dernier thread premier a demarrer ***plus safe!!!!****/
  for(n=0;n<nb_threads;n++)
    {
      
      y=n*nbre+l-1;
      
      arg[n].y=y;arg[n].im_contour=im_contour;arg[n].l=l; arg[n].la=la; 
      arg[n].xmax=xmax; arg[n].ymax=ymax;arg[n].im_fl=im_fl;
      arg[n].im_pt_carac=im_pt_carac;arg[n].tableau=tableau; arg[n].seuil=seuil;
      arg[n].nb_bandes= nbre ;          /*nb_bandes est le nombre de ligne par bande*//*nbre*2*l*/ /*+1*/
      arg[n].no_thread=n;
      if(n==nb_threads-1) {/*printf("la derniere \n");*/ arg[n].nb_bandes = nbre - 2*l;}
      dprints("lance thread %d , traitant  %d  ligne de (%d) a (%d)\n ",arg[n].no_thread,arg[n].nb_bandes,arg[n].y,arg[n].y+arg[n].nb_bandes-1);
      
      res = pthread_create(&(un_thread[n]), NULL, recherche_pt_carac_nlignes, (void *) &(arg[n]));
  
      if (res != 0)
      { 
        kprints("fealure on thread creation \n");
        exit(1);
      }
    }
   
  	dprints("---- %d threads on ete lances ---\n",n);	
    
  for (j = 0; j < nb_threads; j++)
    {
      /*printf("j= %d \n",j); */
      res = pthread_join(un_thread[j], &resultat);
      if (res == 0)
      {        
        dprints("thread %d recueilli \n",j); 
	      vmax=  arg[j].valeur_max;
	      if(vmax>*valeur_max) *valeur_max=vmax;
	    }
      else
	    {
	     kprints("echec de pthread_join %d pour le thread %d\n", res, j);
	     exit(0);
	    }
    }
  	dprints("execution THREADS terminee \n"); 
  /* }*/

}

/* lancement de threads pas encore bien maitrise */
void recherche_pt_carac_thread(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil)
/* tableaux de flottants intermediaires   */
{
  float valeur_max;
  convol_thread(im_contour, l,la, xmax, ymax, im_fl, im_pt_carac, tableau, seuil, &valeur_max );
  
 /* printf("valeur max =%f \n",valeur_max); */
  competition_sur_voisinage(valeur_max, l, la, xmax, ymax, im_fl[1], im_pt_carac);
}

void recherche_pt_carac_fixed_minmax_thread(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil,float vmin,float vmax)
           /* tableaux de flottants intermediaires   */
{
  float valeur_max;
  int x =  0;
  
  x = xmax * ymax;
  convol_thread(im_contour, l,la, xmax, ymax, im_fl, im_pt_carac, tableau, seuil, &valeur_max );

  if(valeur_max<(vmax-vmin)/127.)
    memset(im_pt_carac,0,x*sizeof(int));
  else
    competition_sur_voisinage(valeur_max, l, la, xmax, ymax, im_fl[1], im_pt_carac); 
}
