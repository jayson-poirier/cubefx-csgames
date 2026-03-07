
# cubefx

<img src="cubefx_eq.png" width="400">

La musique _synthwave_ est avant tout une question de son rétro-futuriste : batteries percutantes, synthés scintillants, basse grave et profonde. Pensez à l'esthétique des années 80 rencontre la production moderne.
Pour donner à l'audio ce caractère aux néons, on applique un **décalage de phase par bin** (_per-bin phase shift_) sur tout le spectre de fréquences — un effet qui colore le son avec une texture rétro caractéristique.

Votre mission : rendre ce _pipeline_ incroyablement rapide sur _CPU_ et _CUDA_, sans sacrifier la précision.

Pour cela, vous utiliserez [CubeCL](https://github.com/tracel-ai/cubecl), un _framework_ Rust pour des _kernels_ de calcul haute performance qui s'exécutent à la fois sur _CPU_ et _GPU_, vous permettant d'optimiser le parallélisme et les accès mémoire depuis une seule base de code.

## Traitement audio

Une chanson est une onde continue, aussi appelée signal. Pour la stocker sur un ordinateur, on la discrétise en échantillons (_samples_). Chaque échantillon représente l'amplitude du signal à un instant donné.

```
Onde continue :         ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿

Échantillons discrets : • • • • • • • • • • • •
                        ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
                      Points d'échantillonnage à intervalles réguliers
```

Un taux d'échantillonnage typique est de 44 100 Hz, soit 44 100 échantillons par seconde. L'audio peut être mono, stéréo ou plus, ce qui signifie que le signal comporte plusieurs canaux en parallèle.

### La transformée de Fourier

Pour travailler dans le domaine fréquentiel, on applique une transformée de Fourier.
Cette opération mathématique convertit un signal dans le domaine temporel en ses composantes fréquentielles, c'est-à-dire l'ensemble des ondes sinusoïdales à différentes fréquences dont la somme recrée le signal original.

```
Domaine temporel (signal) :   Domaine fréquentiel (spectre) :

    ∿∿                        |     |              |
   ∿  ∿                       |     |     |        |
  ∿    ∿        ──→           |     |     |   |    |
 ∿      ∿                     └─────┴─────┴───┴────┴──
                              Bas         Milieu   Haut
 Onde complexe                Composantes fréquentielles individuelles
```

Chaque bin de fréquence est représenté par un nombre complexe. En pratique, on utilise simplement une paire de nombres flottants (parties réelle et imaginaire), ou, pour les tenseurs, une paire de tenseurs de nombres flottants.
Puisqu'on part d'échantillons audio à valeurs réelles, on utilise la **_Real FFT_ (_RFFT_)**, qui exploite la symétrie pour ne produire que la moitié non redondante du spectre. L'inverse (**IRFFT**) retransforme en échantillons.

### Le fenêtrage (_Windowing_)

On ne peut pas appliquer la _FFT_ à toute la chanson d'un coup, car les chansons ne sont pas périodiques, et les traiter ainsi entraîne une perte d'information. On travaille donc sur de petites fenêtres qui se chevauchent. Les fenêtres voisines se superposent pour que chaque partie de la chanson soit couverte en douceur, évitant les clics et artefacts aux frontières.

Le fenêtrage est **pré-appliqué** avant le début de votre _pipeline_. Vous recevez des données déjà fenêtrées et retournez des fenêtres traitées. La reconstruction du signal est hors sujet aujourd'hui.

### L'effet de décalage de phase

Pour chaque bin de fréquence `k` dans le spectre, on applique une rotation proportionnelle à l'indice du bin :

```
spectrum[k]  *=  e^(i·α·k)  =  cos(α·k) + i·sin(α·k)
```

Cela donne aux basses fréquences une légère nudge et aux hautes fréquences une rotation plus importante, créant une coloration caractéristique du signal. Un scalaire unique `α` contrôle l'intensité de l'effet.

### Le pipeline

Trois _kernels_ sont exécutés : _RFFT_, décalage de phase et _IRFFT_. Le _benchmark_ mesure l'exécution de l'ensemble du _pipeline_. Lors du profilage, vous pourrez aussi voir un _PRNG_ (_pseudo-random number generator_) tournant pour créer des données, mais celui-ci n'est pas mesuré dans les _benchmarks_ évalués.

```
Entrée pré-fenêtrée   [num_windows, num_channels, window_length]
        │
        ▼
      RFFT           [num_windows, num_channels, freq_bins]   (complexe : deux tenseurs de flottants)
        │
        ▼
  Décalage de phase  spectrum[k] *= e^(i·α·k)
        │
        ▼
      IRFFT          [num_windows, num_channels, window_length]
        │
        ▼
  Fenêtres traitées
```

Par exemple, avec `window_length := 2048`, la _RFFT_ produit `freq_bins := window_length / 2 + 1 = 1025` bins de fréquence.

## Tenseurs, formes et disposition mémoire

Un **tenseur** est un tableau multidimensionnel. On peut le voir comme une généralisation d'une matrice à un nombre quelconque de dimensions. Dans cette base de code, les données audio sont stockées sous forme de tenseurs. Par exemple, l'entrée du _pipeline_ a la forme `[num_windows, num_channels, window_length]` : la première dimension indexe la fenêtre, la seconde le canal, et la troisième l'échantillon dans la fenêtre.

La **forme** (_shape_) indique combien d'éléments existent le long de chaque dimension. Les **strides** (_strides_) indiquent comment traduire un indice multidimensionnel en décalage mémoire plat. Pour un tenseur _row-major_ de forme `[4, 3]`, les _strides_ sont `[3, 1]` : avancer d'un indice sur la dimension 0 saute 3 éléments en mémoire, et avancer sur la dimension 1 en saute 1.

```
Forme [4, 3], strides [3, 1] :

Disposition logique :  Mémoire (plate) :
┌───┬───┬───┐
│ 0 │ 1 │ 2 │          0  1  2  3  4  5  6  7  8  9 10 11
├───┼───┼───┤          ▲        ▲        ▲        ▲
│ 3 │ 4 │ 5 │          ligne 0  ligne 1  ligne 2  ligne 3
├───┼───┼───┤
│ 6 │ 7 │ 8 │
├───┼───┼───┤
│ 9 │10 │11 │
└───┴───┴───┘
```

**Tous les tenseurs de cette base de code sont _row-major_**, ce qui signifie que la dernière dimension est toujours contiguë en mémoire (_stride_ 1). Cela importe pour l'optimisation : lorsque les _threads_ accèdent à des éléments consécutifs sur la dernière dimension, les chargements mémoire peuvent être coalescés sur _GPU_ et favorables au cache sur _CPU_.

Vous n'aurez pas besoin de calculer manuellement les décalages mémoire, car `layout.rs` gère les _strides_ à votre place. Si vous êtes curieux des détails, c'est l'endroit où regarder.

## Démarrage

Commencez par forker ce dépôt (utilisez le bouton `Fork` dans le coin supérieur droit de la page GitHub).
Durant la compétition, on vous demandera de fournir le lien vers votre fork.
Assurez-vous de travailler sur votre fork personnelle, en particulier pour votre dernier commit.

### Lancer l'application

```bash
cargo run --release
```

Utilisez toujours `--release` pour des mesures fiables ; sinon, une version de débogage est compilée, souvent bien plus lente et non représentative.

**Sélection du backend :**

```bash
# CPU
cargo run --release --features cubecl/cpu

# CUDA
cargo run --release --features cubecl/cuda

# Défaut : WGPU (aucun flag nécessaire)
```

Vous serez évalué sur _CPU_ et _CUDA_, donc faites vos _benchmarks_ avec ces configurations.

### Profilage

Activez le profileur CubeCL en mettant `stdout` à `true` dans `cubecl.toml` pour identifier les _kernels_ goulots d'étranglement.
Vous devrez peut-être aussi définir la variable d'environnement `CUBECL_DEBUG_OPTION` à `profile` :

```bash
export CUBECL_DEBUG_OPTION=profile
```

Plus de détails de configuration sont disponibles [ici](https://burn.dev/books/cubecl/advanced-usage/config.html#configuration-file-structure) bien que de nombreux paramètres soient superflus pour ce projet (pas de _streams_ ni d'_autotuning_).

### Tester la correction

Lancez la suite de tests fréquemment :

```bash
cargo test
```

**Sélection du backend :**

```bash
# CPU
cargo test --features cubecl/cpu

# CUDA
cargo test --features cubecl/cuda

# Défaut : WGPU
cargo test
```

**Tests critiques :**

- `large_fft_roundtrip_no_phase_shift` : vérifie la précision de l'aller-retour _FFT_/_IFFT_
- `small_fft_round_trip_with_phase_shift` : valide le _pipeline_ d'effet complet

Ces deux tests sont les seuls qui comptent vraiment pour l'évaluation. Les tests dans `cubefx-engine` sont là pour vous aider à déboguer en cours de route. Consultez les entrées des tests et _benchmarks_ pour comprendre les hypothèses raisonnables sur les données (ex. taille de fenêtre, nombre de canaux).

### Débogage

**Mode test :**

Vous pouvez contrôler la façon dont les tests gèrent les erreurs numériques et de compilation via la variable d'environnement `CUBE_TEST_MODE` :

- `Correct` (défaut) : les erreurs numériques font échouer le test, en affichant uniquement la première erreur.
- `PrintFail` : affiche tous les éléments du tenseur, en indiquant lesquels sont erronés. Accepte un suffixe de filtre optionnel pour ne voir qu'une partie des données.

```bash
export CUBE_TEST_MODE=Correct                  # défaut
export CUBE_TEST_MODE=PrintFail:.,10-20        # filtre : toutes les premières dims, indices 10–20 sur la seconde
```

Les filtres sont des sélecteurs de dimension séparés par des virgules : `.` pour tous les indices, `M` pour un indice unique, `M-N` pour une plage. Le nombre d'entrées doit correspondre au rang du tenseur.
Pour plus de détails, consultez [ici](https://github.com/tracel-ai/cubek/blob/7b9a1f87d9e0cb984cfcb83fb0f04240513038e7/crates/cubek-test-utils/src/test_mode/base.rs).

**Code généré :**

Pour _CUDA_ ou _WGPU_, vous pouvez afficher le code généré de chaque _kernel_ en définissant la variable d'environnement `CUBECL_DEBUG_LOG` à `stdout`. Plus de détails [ici](https://burn.dev/books/cubecl/advanced-usage/config.html#environment-variable-overrides).
Cela peut ne rien afficher si le test réussit ; faites échouer le test ou ajoutez le flag `--nocapture`.

```bash
export CUBECL_DEBUG_LOG=stdout
cargo test --features cubecl/cuda -- --nocapture
```

Mettez la variable à `0` pour la désactiver.

```bash
export CUBECL_DEBUG_LOG=0
```

## Structure du code

Le projet est divisé en deux _crates_ : **`cubefx-eval`** est le binaire principal, et **`cubefx-engine`** est la bibliothèque où réside tout votre travail.

### cubefx-eval (Ne pas modifier)

Gère le _benchmarking_ et les tests de correction. Vos modifications ici ne seront pas utilisées lors de l'évaluation, car nous utiliserons notre propre version.
Notez que le type de données (`f32`) est sélectionné dans cette _crate_, donc choisir un type plus petit n'est pas une option.

### cubefx-engine (Votre espace de travail)

Toute la logique de traitement audio réside ici. Vous pouvez tout modifier, à condition que :

1. L'API utilisée par `cubefx-eval` reste compatible
2. Les deux tests de correction critiques passent

Le _backend_ est sélectionné à la compilation via `TestRuntime` de CubeCL, en utilisant `--features cubecl/cuda` ou `--features cubecl/cpu`.

**Fichiers :**

- **`base.rs`** : Ne pas modifier. Contient le point d'entrée appelé par `cubefx-eval`.
- **`cube/`**
  - **`phase_shift.rs`** : _Kernel_ CubeCL et code de lancement pour le décalage de phase par bin
  - **`layout.rs`** : Convertit les tenseurs en vues sur un seul élément de _batch_ à la fois, gérant les décalages de _batch_ et les _strides_
  - **`tests/`** : Tests _RFFT_ et _IRFFT_, vérifiés contre une implémentation de référence en Rust pur
  - **`fft/`**
    - **`rfft.rs`** : _Kernel_ CubeCL et code de lancement pour la _Real FFT_
    - **`irfft.rs`** : _Kernel_ CubeCL et code de lancement pour l'_IRFFT_
    - **`fft_inner.rs`** : Calcul interne partagé pour les deux directions _FFT_. Pas d'E/S mémoire globale ni de _dispatch_ ici, seulement l'arithmétique de base. Plus complexe à comprendre et probablement plus difficile à optimiser ; à aborder avec précaution.

La plupart de vos modifications se feront dans `phase_shift.rs`, `rfft.rs` et `irfft.rs`. Vous trouverez peut-être aussi des opportunités dans `layout.rs` et `fft_inner.rs`, bien que ce soit probablement plus difficile.

## Opportunités d'optimisation

Normalement, le _kernel_ de décalage de phase devrait s'exécuter beaucoup plus rapidement que les deux autres, même dans leur forme sous-optimale actuelle.
Si ce n'est pas le cas, le _kernel_ de décalage de phase fait probablement quelque chose de stupide.
Une fois cela corrigé, vous pouvez passer à des modifications de _kernels_ moins triviales.

### Configuration de lancement

Les _kernels_ par défaut sont extrêmement naïfs : un seul _worker_ traite toute l'entrée séquentiellement.

Les _kernels_ CubeCL sont lancés avec un _cube count_ et un _cube dim_. Pour l'instant, tous les _cube dims_ et _cube counts_ sont codés en dur à 1.

- **_cube count_** : Le nombre de tâches indépendantes pouvant s'exécuter sur différents processeurs de flux.  
  Sur _CUDA_, cela correspond aux _blocks_. Sur _CPU_, cela crée une boucle sur toutes les tâches.  
  Dans le _kernel_, vous pouvez accéder à l'identifiant du _cube_ courant via `CUBE_POS` (ou `CUBE_POS_X`, `CUBE_POS_Y`, `CUBE_POS_Z` si on utilise plusieurs dimensions).

- **_cube dim_** : Le nombre de _workers_ par _cube_ (appelés _units_ dans CubeCL).  
  Sur _CUDA_, cela correspond aux _threads_. Sur _CPU_, cela correspond aussi aux _threads_ (cœurs).

  Dans un _cube_, les _units_ sont regroupés en _planes_ (appelés _warps_ sur _CUDA_).  
  Sur _CPU_, la taille de _plane_ est 1 (une _unit_ = un _plane_).

  Les _units_ dans un _plane_ s'exécutent en _lockstep_ : ils accèdent à la mémoire en même temps et suivent le même chemin de code (sauf en cas de divergence).

  Vous pouvez interroger la taille de _plane_ à l'exécution avec :
  `client.properties().hardware.plane_size_max`

  Il est recommandé de définir `cube_dim` en 2D :  
  `(plane_size, number_of_planes)`

  Dans le _kernel_ :
  - `CUBE_DIM_X` : taille du _plane_
  - `CUBE_DIM_Y` : nombre de _planes_
  - `UNIT_POS_X` : identifiant de l'_unit_ dans le _plane_
  - `UNIT_POS_Y` : identifiant du _plane_ dans le _cube_

### Motifs d'accès mémoire

La façon dont vos _threads_ accèdent à la mémoire a un impact majeur sur les performances :

- **_GPU_ :** Les _threads_ dans le même _plane_ (_warp_) devraient accéder à des adresses mémoire consécutives. C'est ce qu'on appelle la _coalescence mémoire_ (_memory coalescing_).  
  Par exemple, si l'_unit_ 0 lit l'adresse 0, l'_unit_ 1 lit l'adresse 1, … jusqu'à l'_unit_ 31, le _GPU_ peut tout charger en une seule transaction.  
  Un accès strié ou dispersé réduit l'efficacité de la bande passante.

- **_CPU_ :** Chaque _thread_ devrait travailler sur un bloc continu de mémoire qui tient dans le cache _CPU_.  
  Un accès strié ou dispersé force les _threads_ à charger des données depuis différentes lignes de cache, augmentant les _cache misses_ et ralentissant l'exécution.  
  Les _CPU_ préfèrent un accès séquentiel par _thread_ pour maximiser l'utilisation des lignes de cache et le _prefetching_.

Ces exigences peuvent parfois être contradictoires, il est donc recommandé d'interroger les propriétés matérielles (ex. taille de _plane_) et de concevoir votre _kernel_ pour gérer efficacement les deux _backends_.

### Vectorisation

CubeCL supporte une abstraction « _line_ » qui permet à chaque _thread_ de traiter plusieurs éléments par transaction mémoire. Aucun _kernel_ ne l'utilise actuellement. L'activer signifie moins d'accès à la mémoire globale pour la même quantité de travail, ce qui se combine bien avec de bons motifs d'accès.
Dans les propriétés matérielles, il y a un `load_width` spécifiant combien de bits peuvent être chargés en même temps par une _unit_.
Le facteur de vectorisation maximal pour votre _backend_ est `load_width` divisé par le nombre de bits de chaque élément (combien de bits y a-t-il dans un `f32` ?).
Un bon facteur de vectorisation peut drastiquement accélérer les lectures et écritures en mémoire globale, surtout sur _GPU_. Sur _CPU_, c'est plus utile pour accélérer le calcul, mais le calcul interne de la _FFT_ peut être très difficile à vectoriser.

Pour pouvoir vectoriser un tenseur, la dimension dans laquelle les éléments sont contigus en mémoire (la dernière dimension dans notre cas, car tout est en ordre _row-major_) doit être divisible par le facteur de vectorisation.
Bien que ce soit le cas pour les fenêtres d'échantillons de signal qui sont supposées être une puissance de 2 (typiquement `window_length=2048 éléments`), ce ne sera pas le cas pour les spectres, à cause de la formule `freq_bins = window_length / 2 + 1 = 1025`. Peut-être que _padder_ chaque fenêtre de bins de fréquence (pour avoir une forme de, disons, 1032) avec des zéros pourrait aider.

### Fusion de kernels (_Kernel Fusion_)

La fusion de _kernels_ combine plusieurs opérations en un seul _kernel_, permettant aux résultats intermédiaires de rester dans les registres ou la mémoire partagée plutôt que d'être écrits et relus depuis la mémoire globale.

Actuellement, la _FFT_, le décalage de phase et l'_IRFFT_ sont des lancements de _kernels_ séparés.

En principe, ces trois _kernels_ pourraient être fusionnés en un seul (ou au moins deux), puisqu'ils ont des configurations de lancement similaires, et les données de la _FFT_ sont déjà en mémoire partagée avant d'être écrites en mémoire globale.

### Autres

- **_Unrolling_** : CubeCL supporte le déroulage de boucles via `#[unroll]` sur les boucles `for`, mais uniquement lorsque la plage de la boucle est une valeur **comptime**. Les valeurs _comptime_ sont un concept CubeCL : des constantes intégrées dans le _kernel_ à la compilation plutôt que passées en arguments à l'exécution. Si la borne d'une boucle est une variable à l'exécution, elle ne peut pas être déroulée. Quand applicable, le déroulage élimine le surcoût des branchements et peut exposer davantage de parallélisme au niveau instruction au compilateur.

- **Remplacer les boucles `while` par des boucles `for`** : les boucles `while` peuvent créer des branchements imprévisibles dans un _plane_, entraînant divergence et performances réduites. Utiliser des boucles `for` avec des bornes connues rend l'exécution plus prévisible.

- **Équilibrer les charges de travail** : Idéalement, sur _GPU_, le _cube count_ devrait se répartir équitablement entre les processeurs de flux (_SMs_ — _streaming multiprocessors_). Le nombre de _SMs_ est disponible dans les propriétés matérielles.

- **Respecter le matériel** : Utiliser trop de _planes_ à la fois peut dépasser ce que le système peut supporter efficacement. Il en va de même pour les registres, la mémoire partagée et le cache. Ces ressources sont limitées, et les pousser trop loin peut silencieusement basculer vers des comportements moins efficaces. En haute performance, tout est un compromis.

## Évaluation et classement

### Critères de disqualification

Les équipes produisant des résultats incorrects sur _CPU_ ou _CUDA_ seront disqualifiées et exclues des statistiques de classement.

### Système de notation

Toutes les équipes seront évaluées sur la même machine pour chaque _backend_. Votre score est calculé comme suit :

$$
\text{score} = \frac{\text{mean}_{\text{CPU}} - \text{duration}_{\text{CPU}}}{\text{std}_{\text{CPU}}} + \frac{\text{mean}_{\text{CUDA}} - \text{duration}_{\text{CUDA}}}{\text{std}_{\text{CUDA}}}
$$

Où `mean` et `std` sont calculés sur toutes les équipes qualifiées pour chaque _backend_, et `duration` est le temps de _benchmark_ de votre équipe (moyenne de 10 exécutions après préchauffage). **Plus c'est élevé, mieux c'est.**

### Départage

En raison de la nature non déterministe des _benchmarks_, des scores extrêmement proches peuvent être considérés comme ex æquo.

Si cela se produit, les soumissions des équipes à égalité seront de nouveau _benchmarkées_ sur une machine différente en utilisant des _backends_ supplémentaires (_AMD_ et/ou _Metal_). Ces résultats seront utilisés pour déterminer le classement final.

### Soumission

Make sure to push on your personal fork.
Seul votre **dernier _commit_** sera évalué. Assurez-vous qu'il :

1. Passe les deux tests de correction critiques
2. Inclut votre travail d'optimisation
3. N'introduit pas de nouvelles dépendances au-delà de ce qui est fourni

Si vous découvrez un problème avec votre dernier _commit_ après la soumission, cela vaut la peine de nous contacter, mais aucune garantie ne peut être donnée.

## Conseils

1. **Profilez d'abord.** Utilisez le profileur CubeCL (mettez `stdout` à `true` dans `cubecl.toml`) pour identifier les goulots d'étranglement avant de changer quoi que ce soit. La base de code est assez petite pour raisonner sur les temps d'exécution, mais les mesures sont plus fiables.

2. **Les Entrées/Sorties dominent le calcul, surtout sur _GPU_.** Lire et écrire en mémoire globale est bien plus coûteux que l'arithmétique elle-même. Sur _GPU_ en particulier, les plus grands gains viennent généralement de la réduction du volume de données déplacées, pas d'extraire plus de _flops_ du calcul.

3. **Vérifiez la correction.** Lancez `cargo test` après chaque modification. Des résultats incorrects vous disqualifieront ! Plus lent mais correct vaut toujours mieux.

4. **Commitez souvent.** Chaque fois que vous avez de meilleures performances et des tests qui passent, sauvegardez cet état. Il est facile de casser quelque chose ; les points de sauvegarde permettent de récupérer.

5. **Commencez petit.** Faites des changements incrémentaux et testez fréquemment. Il est plus facile de déboguer un changement qu'une réécriture massive.

6. **Équilibrez les _backends_.** Ne sur-optimisez pas pour l'un au détriment de l'autre. Votre score dépend des deux, _CPU_ et _CUDA_.

7. **Acceptez la casse temporaire.** Supprimer du code pour mesurer combien il coûte est une façon valide de construire une intuition. Vous perdrez l'aspect correct pendant un moment mais gagnerez en compréhension de l'impact sur les performances de certaines lignes de code. Ne soumettez juste pas dans cet état.

8. **Vérifiez si un changement vaut la peine d'être poursuivi.** Suite au point 7, avant d'optimiser du code, vérifiez que supprimer entièrement ce code accélère effectivement quelque chose.
