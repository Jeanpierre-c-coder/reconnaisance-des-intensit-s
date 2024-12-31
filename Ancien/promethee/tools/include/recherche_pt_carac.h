#ifndef _CONTOURS_TOOLS_RECHERCHE_PT_CARAC_H
#define _CONTOURS_TOOLS_RECHERCHE_PT_CARAC_H

void recherche_pt_carac_thread(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil);
void recherche_pt_carac_normale(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil);
void recherche_pt_carac(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil);
void recherche_pt_carac_lent(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil);

void recherche_pt_carac_fixed_minmax(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil,float vmin,float vmax);
void recherche_valeur_minmax(int l /*taille_masque*/ , float ** tableau /*masque*/, float * valeur_min /*valeur_resultat*/,float * valeur_max);

void convol_thread(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil, float *valeur_max);
void recherche_pt_carac_fixed_minmax_thread(unsigned char *im_contour, int l, int la, int xmax,
                        int ymax, float **im_fl, int *im_pt_carac,
                        float **tableau, int seuil,float vmin,float vmax);         

#endif
