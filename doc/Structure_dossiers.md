# Structure des dossiers

08/2022 - Par Sébastien Mick

## Scripts `loop_`

Automatisent la répétition de scripts `train_` ou `replay_` avec un mélange différent des données d'entraînement.

## Scripts `replay_`

Entraînent un modèle à partir des données d'un seul sujet, avec images par ordre chronologique d'acquisition.

## Scripts `run_`

Lancent une passation expérimentale avec le robot. Bien penser à changer l'identifiant du sujet !

`SUBJECT_ID = X`

## Scripts `stat_`

Réalisent des traitements statistiques et génèrent des figures à partir de fichiers enregistrés dans le dossier `npy`

## Scripts `train_`

Entraînent un modèle à partir de données de plusieurs sujets, avec images ordonnées de façon aléatoire mais les images des mêmes sujets sont utilisées pour entraînement et évaluation.

## Autres mots-clés

* `primary`, `prim` : se rapportant aux expressions primaires/basiques
* `secondary`, `seco` : se rapportant aux expressions secondaires/élémentaires
* `mix` : se rapportant aux expressions mixtes
* `mask` : se rapportant au masquage partiel des images d'expressions élémentaires
* `single` : se rapportant à un seul sujet pour les données d'entraînement
* `multi` : se rapportant à plusieurs sujets pour les données d'entraînement
* `loo` : se rapportant à la méthode Leave-One-Out

## utils

Module Python local (d'où la présence d'un fichier `__init__.py`) rassemblant les différentes boîtes à outils intervenant dans les passations expérimentales ou les traitements hors-ligne.

## data

Contient les données, brutes et traitées, associées aux passations expérimentales, calculs hors-ligne et traitements statistiques.

### mix

Contient les images enregistrées lors de passations en mode « expressions mixtes », rangées dans les dossiers sujets étiquetés.

### npy

Contient les fichiers sérialisés utilisés pour enregistrer les matrices de confusion produites par les calculs hors-ligne, afin de faciliter les traitements statistiques et la génération de figures.

### prim

Contient les images enregistrées lors de passations en mode « expressions primaires », rangées dans les dossiers sujets étiquetés.

### seco

Contient les images enregistrées lors de passations en mode « expressions secondaires », rangées dans les dossiers sujets étiquetés.

## doc

Contient les fichiers de documentation.

## img

Contient les images générées par des scripts exécutés hors-ligne : figures, exemples de traitement visuel etc.

## Ancien

Rassemble les programmes (Python et Promethee) datant d'avant mon arrivée au laboratoire (09/2021) et à partir desquels j'ai développé ou adapté les scripts et librairies actuellement utilisées. Contient également deux fichiers ressources : un glossaire des termes se rapportant au traitement visuel, et le rapport de stage de Dorian IBERT (03/2021).

## jwhatwhere

Contient les scripts modifiés par Jordan COSIO (05/2022) pour l'intégration de la localisation (*WHERE*) dans les données d'entrée du modèle, en plus des vignettes log-polaires (*WHAT*).

## maestro-linux

Contient l'utilitaire Maestro (exécutable + dépendances) permettant de paramétrer les canaux de commande des servomoteurs avec une interface graphique.

## misc

Divers scripts n'intervenant ni dans la passation expérimentale ni dans les traitements hors-ligne, utilisés pour mener des tests ou des explorations.
