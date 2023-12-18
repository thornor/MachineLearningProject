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
  
Cependant, la base de données contient également des drafts où le joueur n'est pas allé au bout, il n'a pas fait touts les choix nécéssaires pour pouvoir construire son paquet. Ceci se matérialise par des identifiants de drafts qui n'apparaissent pas suffisamant pour être exploités. Il est alors obligatiore dans notre cas de faire un tri des données, en enlevant celles qui ne correspondant pas à un draft complet.

Dans un premier temps, nous n'utiliserons que l'identifiant des différents drafts, ainsi que les cartes choisies. Etant donné la quantité de données présente dans cette base qui ne serons pas exploitée, il m'a paru peu pertient de créer un tenseur global, qui aurait été retravaillé. Il m'a paru préférable de construire de simples array numpy correctement dimensionnées et avec uniquement les données nécéssaires, avant de le transformer en tenseur exploitable par pytorch.

Cette fonction data_process fonctionne alors comme suit:
- Après chargement des données dans un reader, on crée une matrice copie ne gardant que les données qui nous intéressent, et dans laquelle les noms des cartes sont mis en forme en supprimant toutes les virgules, apostrophes et tirets des noms, et en remplaçant les espaces par des underscores.
- On crée une liste de "phrases", en mettant chaque choix de carte les unes après les autres, en fonction des identifiants des drafts. Une phrase type ressemble donc à ceci: [PICK_START, Expel_The_Interlopers, Chancellor_of_Tales, [...] , Hopeless_Nightmare, Plunge_into_Winter, PICK_END]
- On tokenise les différents mots de chaque phrase en fonction de leur taux d'apparition: Plus une carte est prise souvent, plus son token aura une valeur basse.
- On crée les données d'entrainement et les valeurs de référence à partir de ses phrases tokenisées sous forme de numpy array.
- On va redimensionner ces numpy arrays pour pouvoir en sortir plusieurs batchs pour une éventuelle cross validation.
- A partir des données d'entraînement redimensionnées, on va créer une nouvelle numpy array, qui va retranscrire les choix dans une matrice à 4 dimensions, ou chaque chois de carte sera représenté par un 1 dans chaque colonne aussi grande que le nombre de cartes total.
- On crée nos tenseurs (entrainement et de référence) à partir de ces données.

A partir de ce moment là, on peut créer un premier modèle.

On va dans un premier temps utiliser un modèle RNN simple pour obtenir des premiers résultats. Celui-ci sera suivi d'un modèle linaire afin de pouvoir exploiter les données En utilisant notre base de données à ce moment composé de 100 parties, et en faisant travailler notre modèle sur 300 epochs avec une dimension cachée de 20 et un learning rate de 0.001, on obtient ce graphe: [Graphe 1](graphes/Loss_300_epochs_hidden_dim_20.png). On retrouve la training loss en orange, et la testloss en bleu, toutes deux étant une perte cross-entropique.

Plusieurs problèmes avec ce premier graphe:
- A partir d'une soixantaine d'epochs, la test loss deveint inférieure à la training loss. Ce problème était dû au fait que la fonction de test travaillait sur la même base de données que la fonction d'entraînement. Ce problème a été réglé par la suite.
- On trouve trois pics de loss aux epochs 35, 140 et 220 (environ). Il est difficile pour moi aujourd'hui encore de comprendre l'origine de ces sauts, mais ils doivent probablement venir du manque de représentation de différentes cartes, qui pouvaient mener à de grosses différences entre la prédiction et la valeur de référence. (A ce moement-là, certaines cartes n'apparaissaient qu'une seule fois sur toute la base de donnée)

Suite à ces premiers résultats, j'ai augmenté la base de données, passant de 100 phrases à 400 phrases (ce qui complexifie l'envoi de la base de données sur Github), et j'ai voulu essayer d'appliquer une cross-validation sur ce modèle. Puisque la fonction KFolds de pytorch fonctionne avec des datasets que je n'ai pas, j'ai du essayer de construire ma propre cross validation:
- Avant la transformation en tenseur pytorch, on redimensionne les numpy array de façon à en sortir plusieurs batchs. Dans notre cas, comme notre base de données n'est pas si grande que ça, on ne fera que 5 batchs, générés à partir des phrases précédememnt tokenisées, choisies aléatoirement.
- Ensuite chaque batch sera utilisé comme test tandis que les autres seront utilisées durant l'entrâinement, dans 5 modèles indépendants

En gardant les mêmes hyper-paramètres d'entrée, et en appliquant une cross validation, on obtient les résultats suivants: [Training Loss](graphes/Training_Loss_300_epochs_hidden_dim_20_cross_val.png), [Test Loss](graphes/Test_Loss_300_epochs_hidden_dim_20_cross_val.png)

En augmentant la dimesnion cachée à 50, on obtient ces résultats: [Training Loss](graphes/Training_Loss_300_epochs_hidden_dim_50_cross_val.png), [Test Loss](graphes/Test_Loss_300_epochs_hidden_dim_50_cross_val.png)

On voit certes que le premier problème (à savoir une Test Loss inférieure à la Training loss) a été réglé, mais le deuxième problème a empiré. Toutes les pertes calculées, bien  que suivant grossièrement une trajectoire similaire, varient fortement localement. On va donc diminuer le learning rate, passant 0.001 à 0.0001. On obtient alors, avec une dimension cachée de 20, les résultats suivants:
[Training Loss](graphes/Training_Loss_300_epochs_hidden_dim_20_lr_00001.png), [Test Loss](graphes/Test_Loss_300_epochs_hidden_dim_20_lr_00001.png)

On observe que la perte finale est plus élevée que précédemment, mais on obtient des courbe plus lisses.

On va maintenant essayer de pousser notre modèle: On effectue une cross-validation à 5 batchs avec les hyper-paramètres suivants:
- Nombre d'epochs : 500
- Dimension cachée : 100
- Learning Rate : 0.0001
  
On obtient les résultats suivants :
