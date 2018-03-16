Mini rapport TP4 GPGPU - Yoan Lecoq

# Architecture

## Choix contestables

Le programme utilise l'API de `stdio.h` pour l'affichage plutôt qu'`iostream`,
pour des raisons plutôt subjectives (je trouve que le code est plus concis et
rapide à lire).

De plus, j'utilise la fonction non-standard `asprintf()`, ce qui est
d'ailleurs la seule raison de définir `_GNU_SOURCE` via le Makefile.
Encore une fois, c'est une préférence personnelle par rapport aux APIs C++.
Je sais que ce n'est pas "exception-safe", tout ça tout ça.


## Considérations

Le programme ne traite que les données RGB à 8-bit par canal.
Cela signifie aussi que L, le nombre de niveaux, est toujours mis à 256
(on pourrait l'augmenter, mais il n'y a pas d'intérêt).

Notamment, si L > 2048, le kernel de génération de la CDF
(`generate_cdf_via_inclusive_scan_histogram()`) ne marcherait plus et
nécéssiterait un refactoring.

Le programme ne considère que la CUDA Device n°0
(pour `cudaGetDeviceProperties()`).

Le code est compilé avec le niveau d'optimisation -O3.


## Code CUDA

L'essentiel du code intéressant pour ce projet est dans `src/tone_map_gpu.cu`,
dont le point d'entrée est la fonction `tone_map_gpu_rgb()` tout en bas.

Les données côté GPU sont encapsulées dans une `struct ToneMapGpu`; L'intention
n'est pas de pouvoir en instancier plusieurs (ce n'est pas correct au vu des
déclarations globales comme les textures, ou la mémoire constante), mais
de grouper l'ensemble des données en tant que paramètres implicites à des
fonctions (C'est-à-dire, via `this`) pour ne pas avoir à les répéter partout
dans le code.

Le pipeline est le suivant :

1. Allouer la mémoire GPU d'après largeur et hauteur de l'image;

2. Transférer la mémoire RGB `uchar3` de host vers device;

3. Kernel "RGB to HSV", 1 thread par pixel, utilisant une texture 2D pour
   l'entrée RGB.
   J'avais fait ce choix d'après ma présomption que le hardware était
   particulièrement optimisé pour les accès en lecture seule aux
   textures 2D. J'avais aussi lu que les textures sont un bon choix pour les
   traitements à forte localité spatiale.

4. Kernel "RGB to HSV", 1 thread par pixel, utilisant la mémoire principale pour
   l'entrée RGB.
   Ici le tableau RGB est unidimensionnel et alloué avec `cudaMalloc()`.

5. Kernel "Histogram via per-pixel global atomicAdd()"
   1 thread par pixel fait un `atomicAdd()` sur l'histogramme en mémoire
   principale. C'est l'approche la plus simple et naïve, mais étonnamment,
   elle se comporte mieux que l'approche de l'étape suivante, sur les anciens
   GPUs; et sur les GPUs récents, c'est l'inverse !

6. Kernel "Histogram using shared mem atomicAdd()"
   Chaque block accumule un histogramme en mémoire partagée avec des
   `atomicAdd()`, puis transfère celui-ci dans la mémoire principale avec
   des `atomicAdd()`.
   Cette approche n'est pas non plus très sophistiquée, mais je ne souhaite pas
   plonger plus loin dans la complexité.
   Les "shared memory atomics" sont recommendés pour les architectures à
   partir de Maxwell.

7. Kernel "CDF via inclusive scan of histogram"
   Tout est dans le nom: ce kernel calcule la Cumulative Distribution Function
   par un inclusive scan sur l'histogramme en mémoire globale.
   Il n'y a qu'un seul block, et L/2 = 128 threads.
   L'histogramme est d'abord copié en mémoire partagée, d'où il est
   traité (conceptuellement) comme un arbre binaire balancé.
   Il y a deux phases : Réduction (downsweep) et upsweep.
   Le code est basé sur le schéma de la slide 20 de la présentation suivante :
   http://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf

   Il n'y a pas de variante de ce kernel qui place l'histogramme en mémoire
   constante, parce qu'il n'y a pas de bénéfice à en tirer: chaque élément
   de l'histogramme n'est initialement lu qu'une et une seule fois, et il y a
   de toute façon besoin d'accumuler des résultats quelque part.

8. Kernel "Tone mapping, then HSV to RGB (CDF in global mem)"
   Un autre nom auto-descriptif. Le tone mapping et la conversion de HSV vers
   RGB est une opération qui se fait par pixel, donc il n'y a pas de raison
   d'en faire deux kernels séparés.
   Ici, la CDF est lue depuis la mémoire principale.

9. Kernel "Tone mapping, then HSV to RGB (CDF in constant mem)"
   Pareil que le précédent, sauf que la CDF est lue depuis la mémoire
   constante. L'intérêt, c'est que la mémoire constante est normalement
   optimisée pour des accès concurrents en lecture à des adresses
   potentiellement superposées ("broadcast").
   (aussi d'après https://stackoverflow.com/a/18021374/7972165).
   Dans notre cas, la CDF comporte L = 256 éléments, et on y accède
   aléatoirement selon chaque WxH pixel, or WxH est très souvent bien
   plus grand que L.
   Ainsi, la probabilité d'accès concurrents à des adresses similaires
   est plus grande, donc on peut s'attendre à un gain de performance en
   utilisant la mémoire constante.

10. Télécharger la mémoire RGB `uchar3` de device vers host.
11. Libérer la mémoire.


Certains kernels font exactement le même travail que leur
antécédent, mais d'une différente manière, ce qui signifie que les résultats
dudit antécédent sont écrasés.
Le but est de pouvoir observer les différences de performance entre les
techniques, mais en "production" on prendra soin de choisir celles qui
marchent le mieux pour le hardware que l'on a sous la main.
Cela signifie aussi que si l'on change le code des kernels, il faut s'assurer
manuellement que chacun continue de produire les résultats corrects
(un kernel correct peut "cacher" les résultats d'un kernel antécédent
incorrect).

Pour réduire la duplication de code, certains kernels sont "templatés" avec
un booléen. Il n'y a normalement pas de pénalités pour un `if()` sur une
constante connue à la compilation.

Les kernels acceptent des pointeurs `__restrict__` partout, dans l'espoir
d'augmenter les opportunités d'optimisation pour le compilateur, d'après notre 
connaissance du fait que nos pointeurs ne sont pas sujets à l'aliasing.
[Je me base sur cet article de Mike Acton que j'avais lu il y a qelques années](https://cellperformance.beyond3d.com/articles/2006/05/demystifying-the-restrict-keyword.html).

Les kernels sont lancés 200 fois afin de mieux évaluer leur performance.
Ce nombre peut être changé; il se trouve à un seul endroit, dans
`tone_map_gpu_rgb()`.


# Résultats

Le programme a été testé sur les GPUs suivants :
- Quadro K620 (Salle 0B002, génération Maxwell);
- GeForce GT 635M (mon PC portable, génération Fermi).

Pour facilement référencer mes kernels dans les tables, je les nomme
K0 à K6, comme suit:
- K0: RGB to HSV (RGB in 2D texture)
- K1: RGB to HSV (RGB in global mem)
- K2: Histogram via per-pixel global atomicAdd()
- K3: Histogram using shared mem atomicAdd()
- K4: CDF via inclusive scan of histogram
- K5: Tone mapping, then HSV to RGB (CDF in global mem)
- K6: Tone mapping, then HSV to RGB (CDF in constant mem)

Le temps d'exécution de chaque kernel reporté par le programme est 
la moyenne des temps d'exécution sur 200 invocations.

Les tables ci-dessous présentent les temps d'exécution en millisecondes
de chaque kernel, pour chaque image, à raison d'une table par GPU testé.

Les images testées sont les suivantes :
- Chateau (2550x1917): `images/Chateau.png`;
- Nuit (1680x1050): `images/Nuit.png`;
- Hawkes (1024x683): `images/Unequalized_Hawkes_Bay_NZ.png`;
- Paris (1000x562): `images/Paris.png`;
- Lena (512x512): `images/Lena.png`.

**Table A. Quadro K620, CUDA 5.0, arch=compute\_50, code=sm\_50**

   |  Chateau  |   Nuit   |  Hawkes  |   Paris  |   Lena
-----------------------------------------------------------
K0 |  3.729295 | --TODO-- | 0.614809 | --TODO-- | 0.211583
K1 |  3.813473 | --TODO-- | 0.632824 | --TODO-- | 0.216113
K2 |  5.233018 | --TODO-- | 0.592985 | --TODO-- | 0.182630
K3 |  1.450710 | --TODO-- | 0.195026 | --TODO-- | 0.067028
K4 |  0.006714 | --TODO-- | 0.006832 | --TODO-- | 0.006900
K5 |  3.716156 | --TODO-- | 0.459780 | --TODO-- | 0.221214
K6 |  3.481602 | --TODO-- | 0.434734 | --TODO-- | 0.208767


**Table B. GeForce GT 635M, CUDA 2.1, arch=compute\_20, code=sm\_20**

   |  Chateau  |   Nuit   |  Hawkes  |   Paris  |   Lena   
-----------------------------------------------------------
K0 | 11.139908 | 4.626625 | 1.790500 | 1.928667 | 0.730098 
K1 | 10.677858 | 4.536003 | 1.781730 | 1.853236 | 0.724311 
K2 | 14.999317 | 3.560685 | 1.482985 | 1.524979 | 0.540963 
K3 | 20.794110 | 4.948350 | 2.480159 | 1.888935 | 0.618014 
K4 |  0.008361 | 0.008672 | 0.008396 | 0.008684 | 0.008618 
K5 | 10.284321 | 3.836512 | 1.155794 | 1.114774 | 0.619960 
K6 |  9.751174 | 3.859802 | 1.178590 | 1.126635 | 0.647035 


## Observations

Les performance entre K2 et K3 varient sauvagement en fonction du GPU testé.
J'expliquerais cela par la meilleure implémentation des "shared memory atomics"
des dernières générations de GPUs.
cf. [`GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell`](https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)

K0 (texture 2D) se comporte mieux que K1 (mémoire principale) sur la
Quadro K620. C'est l'inverse sur la GeForce GT 653M, mais dans les deux cas
il n'y a pas de différence flagrante.
J'expliquerais cela par le fait qu'en fin de compte, à un thread par pixel, 
et un fetch par thread, on n'exploite pas beaucoup la localité spatiale
offerte par les textures.

Sur les deux GPUs, il semble que K6 gagne face à K5 (i.e il paraît
avantageux de lire la CDF en mémoire constante plutôt qu'en mémoire
globale), sauf, visiblement, avec la GeForce GT 635M sur les petites images.
