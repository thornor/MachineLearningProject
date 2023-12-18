# Projet Deep Learning

L'objectif de ce projet est de prédire le choix d'un individu en fonction de ses choix précédents et des choix effectués et du taux de réussite obtenu par des personnes 
indépendantes. Pour cela, nous allons nous baser sur le jeu de cartes à jouer et à collectionner *Magic l'Assemblée* et plus précisément sa version en ligne, *Magic Arena*. 

Nous allons nous concentrer sur un format de ce jeu appelé "Draft", dans lequel 8 joueurs vont chacun ouvrir un paquet de 15 cartes neuf issu de la même extension, en choisir une, puis passer le paquet de 14 
cartes à leur voisin de gauche pour récupérer celui donnépar leur voisin de droite. Une fois toutes les cartes sélectionnées par les joueurs, ceux-ci vont reproduire ce processus
encore 2 fois, en changeant de sens de transfert de paquet après chaque ouverture. 

L'objectif de chaque joueur est donc à la fois de prendre les meilleures cartes pour améliorer son paquet personnel tout en conservant une certaine synergie entre les cartes: une 
carte objectivement meilleure peut-être inutile si elle ne fonctionne pas avec le reste des cartes choisies, et inversement.

Les données provenant de l'application *Magic Arena* proviennent d'un mode de jeu au fonctionnement similaire, à l'exception qque toutes les cartes choisies sont virtuelles. Le 
modèle se basera sur les choix faits par les joueurs lors des drafts de l'extension la plus récente, *Les Friches d'Eldraine*, sorti début septembre. Cela permet d'avoir une base de 
données suffisament grande pour pouvoir correctement entrainer et évaluer le modèle, tout en restant raisonnable sur sa taille. 

Nous pouvons approcher ce problème comme celui d'un problème de génération de langage, où les mots seraient les cartes choisies, et les phrases l'ordre de choix.
Il existe cependant une particularité dans notre modèle par rapport à un modèle de langage génératif classique: pour chaque choix de carte (ie chaque génération de mot),
le modèle n'aura accès qu'à une liste limitée de choix, déterminée en entrée. Comme le joueur, il ne pourra choisir une carte que parmi les 14 possibles, puis les 13, etc...

Chaque ligne de la base de données initiale est constituée comme suivant:
- L'extension jouée
- Le format de jeu utilisé 
- L'identifiant du draft observé et l'heure à laquelle le joueur a participé
- Son nombre de victoires et de défaites dans ce format avant de commencer
- Le nom de la carte choisie
- Un booléen indiquant si cette carte choisie a été jouée ou non
- Une suite de booléens permettant de connaître les autres choix possibles pour le joueur 
- Une suite de booléens permettant de connaître les choix précédents du joueur
- Son taux de victoires avec le paquet formé.

Dans un premier temps, nous n'utiliserons que l'identifiant des différents drafts, ainsi que les cartes choisies. Etant donné la quantité de données présente dans cette base qui ne serons pas exploitée, il m'a paru peu pertient de créer un tenseur global, qui aurait été retravaillé. Il m'a paru préférable de construire de simples array numpy correctement dimensionnées et avec uniquement les données nécéssaires, avant de le transformer en tenseur exploitable par pytorch.

Cette fonction data_process fonctionne alors comme suit:
- Après chargement des données dans un reader, on crée une matrice copie ne gardant que les données qui nous intéressent, et dans laquelle les noms des cartes sont mis en forme en supprimant toutes les virgules, apostrophes et tirets des noms, et en remplaçant les espaces par des underscores.
- On crée une liste de "phrases", en mettant chaque choix de carte les unes après les autres, en fonction des identifiants des drafts. Une phrase type ressemble donc à ceci: [PICK_START, Expel_The_Interlopers, Chancellor_of_Tales, [...] , Hopeless_Nightmare, Plunge_into_Winter, PICK_END]
- On tokenise les différents mots de chaque phrase en fonction de leur taux d'apparition: Plus une carte est prise souvent, plus son token aura une valeur basse.
- On crée les données d'entrainement et les valeurs de référence à partir de ses phrases tokenisées sous forme de numpy array.
- On va redimensionner ces numpy arrays pour pouvoir en sortir plusieurs batchs pour une éventuelle cross validation.
- A partir des données d'entraînement redimensionnées, on va créer une nouvelle numpy array, qui va retranscrire les choix dans une matrice à 4 dimensions, ou chaque chois de carte sera représenté par un 1 dans chaque colonne aussi grande que le nombre de cartes total.
- On crée nos tenseurs (entrainement et de référence) à partir de ces données
