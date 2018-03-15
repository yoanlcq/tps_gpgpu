TODO: Plan
détaillant le pipeline de votre programme et expliquant vos choix de programmation (pourquoi tel nombre de threads, pourquoi telle zone mémoire, etc...).

# Résultats

Le programme a été testé sur une machine de la salle 1B077 (Quadro K620),
ainsi que sur mon PC portable (GeForce GT 635M).

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
- Chateau (2550x1917): `images/Chateau.png`
- Hawkes (1024x683): `images/Unequalized_Hawkes_Bay_NZ.png`
- Lena (512x512): `images/Lena.png`

**Table A. Quadro K620, CUDA 5.0, arch=compute\_50, code=sm\_50**

   |  Chateau  |  Hawkes  |   Lena
--------------------------------------
K0 |  3.729295 | 0.614809 | 0.211583
K1 |  3.813473 | 0.632824 | 0.216113
K2 |  5.233018 | 0.592985 | 0.182630
K3 |  1.450710 | 0.195026 | 0.067028
K4 |  0.006714 | 0.006832 | 0.006900
K5 |  3.716156 | 0.459780 | 0.221214
K6 |  3.481602 | 0.434734 | 0.208767


**Table B. GeForce GT 635M, CUDA 2.1, arch=compute\_20, code=sm\_20**

   |  Chateau  |  Hawkes  |   Lena
--------------------------------------
K0 | 11.139908 | 1.790500 | 0.730098
K1 | 10.677858 | 1.781730 | 0.724311
K2 | 14.999317 | 1.482985 | 0.540963
K3 | 20.794110 | 2.480159 | 0.618014
K4 |  0.008361 | 0.008396 | 0.008618
K5 | 10.284321 | 1.155794 | 0.619960
K6 |  9.751174 | 1.178590 | 0.647035

## Observations

K2 et K3 réalisent la même tâche, mais K3 accumule un 
histogramme par block avec des atomicAdd() sur de la mémoire partagée
avant d'atomicAdd() chaque élément de l'histogramme partagé dans
l'histogramme global.

Or, on observe des différences flagrantes de performance entre ces deux
kernels. J'expliquerais ça par la meilleure implémentation des "shared
memory atomics" des dernières générations de GPUs.
cf. [`GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell`](https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)

K0 ne gagne pas clairement contre K1 (i.e une texture 2D n'est pas
clairement avantageuse par rapport à la mémoire globale).
J'expliquerais cela par le fait qu'en fin de compte, à un thread par pixel, et un fetch par thread, on n'exploite pas beaucoup la localité spatiale
offerte par les textures (ni les "interpolations gratuites", etc).

Sur les deux GPUs, il semble que K6 gagne face à K5 (i.e il paraît
avantageux de lire la CDF en mémoire constante plutôt qu'en mémoire
globale), sauf, visiblement, avec la GeForce GT 635M sur des petites images !
