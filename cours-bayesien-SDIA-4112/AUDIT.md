# Audit du document source

**Fichier analysé** : `C:\Users\HP\Downloads\EC _ EPD-SDIA-4112 – Statistique Bayésienne__Parco.pdf`
**Taille** : 5 636 892 octets — **176 pages** — 612 × 792 pt
**Producteur** : Chromium / Skia PDF m127 (métadonnée `creationDate: 2026-09-20T02:15:06Z`)
**Nature réelle** : export « imprimer en PDF » d'une **session de conversation avec un LLM**,
contenant du texte de chat, du **code source LaTeX Beamer en texte brut**, des journaux de
compilation et des listes d'exercices. Ce n'est pas un support de cours imprimable.

## 1. Inventaire mesuré

| Pages | Contenu réel | `\begin{frame}` |
|---|---|---|
| 1–2 | Fiche descriptive de l'EC : objectifs, prérequis, contenu, charge 45 h / 3 crédits, bibliographie | 0 |
| 3–12 | Plan global proposé (5 chapitres × 50 slides) + description « slide par slide » du chapitre 1 | 0 |
| 13–40 | **Chapitre 1 en LaTeX Beamer** — version 1 (avec Font Awesome) | 51 |
| 41–68 | **Chapitre 1 — version 2** (révisée, sans Font Awesome) : **doublon** | 51 |
| 69–76 | Journaux de compilation `pdflatex`, messages d'erreur, prose de chat | 0 |
| 77–96 | **Chapitre 2 en LaTeX Beamer** — version 1 | 51 |
| 97–119 | **Chapitre 2 — version 2** (corrigée) : **doublon** | 51 |
| 120–148 | Prose : plans de chapitres, listes d'exercices (20 + 10 + 20), réponses du chat | 0 |
| 149–170 | **Chapitre 3 en LaTeX Beamer** | 51 |
| 171–176 | Ressources web (liens PyMC, ArviZ, tutoriels), fin de conversation | 0 |

Mesures automatiques (`src/extract.py`, `src/audit.py`, `src/clip.py`, `src/outline.py`) :

- **270 089 caractères** de texte extractible, **0 page sans couche texte** ;
- **255 environnements `frame`** au total, mais dont **102 sont des doublons** :
  seuls **153 frames distincts** existent réellement (chapitres 1, 2 et 3) ;
- **22 cibles `\includegraphics` distinctes**, pour **47 occurrences** au total :
  `beta_animation.png`, `posterior_plot.png`, `regression_bayes.png`,
  `pvalue_vs_posterior.gif`, `qr_code.png`, `timeline_bayes.png`, `trace_plot.png`…
  — et **aucun de ces fichiers n'est fourni** (une seule image est incorporée dans
  tout le PDF : page 1) ;
- pire : sur ces 47 occurrences, **23 sont commentées** (`%`) et **24 sont actives**.
  La version « active » — qui correspond à l'un des deux doublons du chapitre 1 —
  **ne peut pas compiler**, faute de fichiers image. C'est cohérent avec les
  journaux d'erreurs `pdflatex` présents aux pages 69–76 ;
- **0 compilation réussie** : le document contient au contraire **sept blocs de
  journaux d'erreurs** `pdflatex`.

## 2. Troncature des lignes de code (mesurée géométriquement)

L'export a été produit avec des blocs de code en `white-space: pre` et
`overflow: hidden` : **les lignes trop longues sont coupées net**, sans retour
à la ligne, et le texte perdu **n'existe pas** dans le fichier PDF (ce ne sont pas
des lignes visuellement rognées, les glyphes sont absents du flux de contenu).

- Détection géométrique : `x1 = 547,3 pt` est le bord droit du conteneur de code.
  **187 lignes** s'arrêtent exactement sur ce bord, réparties sur **76 pages**.
- Exemples de pertes constatées : `... de manière probabiliste et interprétab`,
  `... une situation incertaine dans so`, `... Qu'est-ce que c'est que la`, `... quell`.
- Conclusion : **le source LaTeX du PDF n'est pas copiable-collable** en l'état.
  Toute reprise « à l'identique » produirait un fichier non compilable.

## 3. Erreurs de fond et de forme relevées dans le matériel d'origine

| # | Problème | Vérification |
|---|---|---|
| 1 | Slide « spam avec 2 occurrences » annonce **90 %** | Le calcul exact donne $0{,}25 \times 12^2 = 36$, soit $36/37 = \mathbf{97{,}3\ \%}$. Le facteur de vraisemblance **se multiplie**, il ne s'additionne pas. Corrigé (chapitre 1, slide 23) et signalé en encadré. |
| 2 | `\checkmark already defined` (journal p. 71) | Le paquet `amssymb` définit déjà `\checkmark`. Supprimé, remplacé par un `\newcommand` propre. |
| 3 | `Invalid UTF-8 byte sequence` — 14 occurrences (journal p. 69–70) | Accents français dans les blocs `listings` sous `pdflatex`. Résolu par XeLaTeX + `lstset{literate=…}` explicite. |
| 4 | `Illegal parameter number in definition of \iterate` — 20 occurrences | Cause réelle : `lstlisting` placé dans une frame **non déclarée `[fragile]`**. Reproduit et corrigé ; règle documentée dans `src/check_env.py`, qui refuse désormais toute frame concernée. |
| 5 | `Fontconfig error: Cannot load default config file` | Présent à chaque compilation de l'original. |
| 6 | Chapitres 1 et 2 présents **en double**, avec deux mises en forme divergentes | Fusion en une version unique et propre. |
| 7 | Chapitres 4 et 5 **inexistants** (plan seulement, en prose) | Écrits intégralement. |
| 8 | Aucun notebook, aucun jeu de données, aucun TP exécutable, malgré 8 h de TP annoncés | Cinq notebooks + un mini-projet créés et **exécutés**. |
| 9 | Toutes les valeurs numériques du document sont **posées à la main**, sans source de calcul | *(vérification impossible : les scripts n'existaient pas)*. Voir le rapport suivant. |

## 4. Vérification numérique : ce que le matériel d'origine affirmait

Chaque valeur du support d'origine qui a pu être recalculée a été vérifiée par exécution.
Trois familles de résultats :

**a) Valeurs confirmées** — `Beta(8,4)` moyenne 0,667 ; intervalle $[0{,}390\,;\,0{,}891]$ ;
$P(\theta>0{,}5\given y)=0{,}887$ ; diagnostic COVID à 5 % de prévalence : 71,4 %.

**b) Valeurs fausses** — la probabilité de spam après 2 occurrences (90 % annoncés, 97,3 % réels) ;
l'intervalle de crédibilité du slide 15 annoncé `[0,35 ; 0,87]` avec un « plus étroit grâce au prior »
qui n'a pas de fondement (aucune simulation à l'appui).

**c) Valeurs invérifiables** — toutes les sorties PyMC/ArviZ annoncées
(« mode ≈ 0,65 », « IC 95 % = [0,39 ; 0,85] », `R̂ < 1,01`) : **les scripts n'existaient pas**.
Elles ont été remplacées par des sorties **réellement produites sur cette machine**
(voir `figures/key_numbers_ch*.json`, un fichier par chapitre, écrit par les scripts `src/fig_*.py`).

## 5. Décision retenue

Réécriture intégrale à neuf sur la base de la **structure pédagogique** (qui est solide :
5 chapitres, 50 slides, quiz, activités, code) et non du code LaTeX d'origine.
Toute valeur numérique du nouveau support provient d'un script exécuté et reproductible.
