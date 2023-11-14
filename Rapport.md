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