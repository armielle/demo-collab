"""Video : la propagation dans un CNN, de l'image a la sortie dense.

8 scenes independantes. Toutes les valeurs affichees proviennent de
prep_donnees.py (calculees, graine 4112) — aucune valeur inventee.
"""
import json, os
import numpy as np
from manim import *

ICI = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(ICI, "assets")
D = json.load(open(os.path.join(A, "donnees.json"), encoding="utf-8"))

BG = "#1C1C1C"
PRIMARY = "#58C4DD"
SECOND = "#83C167"
ACCENT = "#FFFF00"
ROUGE = "#FC6255"
GRIS = "#888888"
TEXTE = "#EAEAEA"
ORANGE = "#FF9F43"
VIOLET = "#A29BFE"
MONO = "Consolas"

pet = np.load(os.path.join(A, "grille12.npy"))
sort = np.load(os.path.join(A, "sortie10.npy"))
pool5 = np.load(os.path.join(A, "pool5.npy"))
vec16 = np.load(os.path.join(A, "vec16.npy"))
h8 = np.load(os.path.join(A, "h8.npy"))


# ------------------------------------------------------------------ utilitaires
def T(txt, taille=30, coul=TEXTE, gras=False):
    return Text(txt, font_size=taille, font=MONO, color=coul,
                weight=BOLD if gras else NORMAL)


def carte_grille(m, cellule=0.34, valeurs=True, cmap="blue", opacite=1.0):
    """Grille de carres colories d'apres une matrice normalisee."""
    m = np.asarray(m, float)
    mn, mx = m.min(), m.max()
    norm = (m - mn) / (mx - mn + 1e-9)
    cmin, cmax = {"blue": ("#0B1E3A", PRIMARY), "red": ("#2A0F12", ROUGE),
                  "green": ("#0E2A18", SECOND), "grey": ("#222222", "#BBBBBB")}[cmap]
    g = VGroup()
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = norm[i, j]
            c = Manim.color_gradient([cmin, cmax], 2)[0] if v < .5 else \
                Manim.color_gradient([cmin, cmax], 2)[1]
            sq = Square(side_length=cellule, stroke_width=.6, stroke_color=GRIS,
                        fill_color=c, fill_opacity=.12 + .78 * v * opacite)
            g.add(sq)
    g.arrange_in_grid(rows=m.shape[0], cols=m.shape[1], buff=0)
    if valeurs:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                g[i * m.shape[1] + j].add(
                    Text(f"{m[i, j]:.1f}" if abs(m[i, j]) < 10 else f"{m[i, j]:.0f}",
                         font_size=10, font=MONO,
                         color=TEXTE if norm[i, j] > .45 else "#AAB4C0"
                         ).move_to(g[i * m.shape[1] + j]))
    return g


def image_mob(fichier, hauteur):
    m = ImageMobject(os.path.join(A, fichier))
    m.height = hauteur
    return m


def entete(scene, txt):
    e = T(txt, 30, PRIMARY, True).to_edge(UP, buff=0.45)
    scene.play(Write(e), run_time=1.0)
    return e


class S1_Ouverture(Scene):
    def construct(self):
        self.camera.background_color = BG
        img = image_mob("entree.png", 4.2)
        titre = T("De l'image à la décision", 46, PRIMARY, True).to_edge(UP, buff=0.7)
        sous = T("la propagation avant dans un réseau convolutif", 26, TEXTE)\
            .next_to(titre, DOWN, buff=0.4)
        self.add_subcaption("De l'image à la décision", duration=2)
        self.play(FadeIn(titre, shift=DOWN * .3), run_time=1.2)
        self.play(FadeIn(sous), run_time=.8)
        self.add_subcaption("Voici notre image de travail", duration=2)
        self.play(GrowFromCenter(img), run_time=1.5)
        self.wait(1.2)
        accroche = T("Comment 9 nombres suffisent-ils à décider ?", 28, ACCENT)\
            .to_edge(DOWN, buff=0.6)
        self.add_subcaption("Comment 9 nombres suffisent-ils a decider ?", duration=3)
        self.play(Write(accroche), run_time=1.6)
        self.wait(2.0)
        self.play(FadeOut(VGroup(titre, sous, accroche, img)), run_time=.6)


class S2_ImageTenseur(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "1. Une image, c'est des nombres")
        g = carte_grille(pet, 0.34).move_to(ORIGIN)
        self.add_subcaption("Chaque pixel est un nombre", duration=2)
        self.play(Create(g), run_time=1.6)
        self.wait(.8)
        # un pixel mis en evidence
        px = SurroundingRectangle(g[5 * 12 + 6], color=ACCENT, buff=.03)
        lbl = T("ce pixel vaut une intensité", 22, ACCENT).to_edge(DOWN, buff=0.5)
        self.add_subcaption("Ce pixel vaut une intensite", duration=2)
        self.play(Create(px), Write(lbl), run_time=1.2)
        self.wait(1.4)
        note = T("et rien d'autre : pas de structure, pas de sens", 22, GRIS)\
            .next_to(lbl, UP, buff=.25)
        self.play(Write(note), run_time=1.0)
        self.wait(1.4)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=.5)


class S3_Convolution(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "2. Le noyau de convolution")
        noyau = np.array(D["noyau_sobel"])
        g = carte_grille(pet, 0.30).scale(.9).to_edge(LEFT, buff=.7).shift(DOWN * .5)
        # opacite en couches : la fenetre 3x3 active, le reste discret
        for i in range(144):
            g[i].set_opacity(.25)
        k = carte_grille(noyau, 0.42, cmap="red").next_to(g, RIGHT, buff=1.2)
        k_title = T("noyau 3×3", 20, ROUGE).next_to(k, UP, buff=.2)
        self.play(Create(g), run_time=1.2)
        self.play(FadeIn(k), Write(k_title), run_time=1.0)
        self.wait(.8)

        fen = SurroundingRectangle(VGroup(g[0], g[2]), color=ACCENT, buff=.02)
        fen.set(width=3 * .27, height=3 * .27)
        self.play(Create(fen), run_time=.8)

        # position (0,0) : calcul detaille
        zone = pet[:3, :3] * noyau
        prod = T(" + ".join(f"{v:.1f}" for v in zone.ravel()), 20, TEXTE)
        somme = T(f"= {D['premier_calcul']['somme']:.1f}", 30, ACCENT, True)
        bloc = VGroup(prod, somme).arrange(DOWN, buff=.3).next_to(k, RIGHT, buff=1.0)
        fleche = Arrow(k.get_right(), bloc.get_left(), color=GRIS, buff=.1)
        self.add_subcaption("On multiplie zone par noyau, puis on additionne", duration=3)
        self.play(Write(prod), Create(fleche), run_time=1.4)
        self.play(Write(somme), run_time=.8)
        self.wait(1.6)

        # sortie : grille 10x10 qui se remplit
        s = carte_grille(sort, 0.22, valeurs=False, cmap="green").next_to(bloc, RIGHT, buff=1.0)
        s_title = T("carte d'activation", 20, SECOND).next_to(s, UP, buff=.2)
        self.play(Create(s), Write(s_title), run_time=1.2)
        cell0 = Square(side_length=.22, fill_color=SECOND, fill_opacity=.9,
                       stroke_width=0).move_to(s[0])
        self.play(FadeIn(cell0), run_time=.6)
        self.wait(1.0)

        # balayage : quelques positions
        positions = [(0, 1), (0, 2), (1, 0), (2, 1), (3, 3), (4, 2), (2, 4), (1, 3)]
        self.add_subcaption("Le noyau balaie toute l'image", duration=3)
        for (i, j) in positions:
            cible = VGroup(g[i * 12 + j], g[i * 12 + j + 2]).copy()
            self.play(fen.animate.move_to(cible.get_center()),
                      run_time=.35, rate_func=linear)
            v = sort[i, j]
            op = .12 + .78 * abs(v) / max(abs(sort).max(), 1e-9)
            self.play(s[i * 10 + j].animate.set_fill(opacity=op),
                      run_time=.25, rate_func=linear)
        self.wait(.6)

        # la vraie carte, calculee
        vraie = image_mob("carte_contours.png", 3.6).next_to(s, RIGHT, buff=1.1)
        lbl = T("carte complète (calculée)", 20, SECOND).next_to(vraie, UP, buff=.2)
        self.add_subcaption("La carte complete, calculee sur l'image reelle", duration=3)
        self.play(FadeIn(vraie, shift=LEFT * .3), Write(lbl), run_time=1.4)
        self.wait(1.8)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=.6)


class S4_ReLU(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "3. ReLU : couper ce qui est négatif")
        g = carte_grille(sort, 0.30, cmap="grey").move_to(ORIGIN).shift(LEFT * .8)
        # negatifs en rouge
        for i in range(100):
            v = sort.ravel()[i]
            if v < 0:
                g[i][0].set_fill(ROUGE, opacity=.55)
        self.play(Create(g), run_time=1.4)
        lbl = T("négatifs", 22, ROUGE).next_to(g, DOWN, buff=.4)
        self.play(Write(lbl), run_time=.8)
        self.wait(.8)

        formule = T("ReLU(x) = max(0, x)", 34, PRIMARY, True).to_edge(RIGHT, buff=.8)
        self.add_subcaption("Les valeurs negatives deviennent zero", duration=3)
        self.play(Write(formule), run_time=1.4)
        self.wait(1.0)

        # animation : les negatifs tombent a zero
        for i in range(100):
            v = sort.ravel()[i]
            if v < 0:
                g[i][0].set_fill(GRIS, opacity=.12)
        n_txt = T(f"{D['relu_negatifs']} cellules sur {D['relu_total']} remises à zéro",
                  22, ORANGE).next_to(formule, DOWN, buff=.5)
        self.play(Write(n_txt), run_time=1.2)
        self.wait(1.8)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=.5)


class S5_Pooling(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "4. Le pooling : réduire, sans apprendre")
        g = carte_grille(pool5 * 0 + sort[:10, :10], 0.30, valeurs=False,
                         cmap="grey").to_edge(LEFT, buff=.9).shift(DOWN * .4)
        # valeurs de la carte 10x10
        for i in range(100):
            v = sort[:10, :10].ravel()[i]
            g[i].add(Text(f"{v:.1f}", font_size=10, font=MONO, color=TEXTE)
                     .move_to(g[i]))
        self.play(Create(g), run_time=1.4)
        fen = SurroundingRectangle(VGroup(g[0], g[11]), color=ACCENT, buff=.02)
        fen.set(width=2 * .30, height=2 * .30).move_to(g[0])
        self.play(Create(fen), run_time=.7)

        sortie = carte_grille(pool5, 0.40, cmap="green").next_to(g, RIGHT, buff=2.2)
        s_title = T("5×5", 24, SECOND).next_to(sortie, UP, buff=.2)
        self.play(Create(sortie), Write(s_title), run_time=1.2)

        self.add_subcaption("Dans chaque bloc 2x2, on garde le maximum", duration=3)
        positions = [(0, 0), (0, 2), (1, 1), (2, 0), (3, 2), (4, 1)]
        for (i, j) in positions:
            centre = g[i * 2 * 10 + j * 2]
            self.play(fen.animate.move_to(centre.get_center()),
                      run_time=.4, rate_func=linear)
            v = pool5[i, j]
            op = .12 + .78 * v / max(pool5.max(), 1e-9)
            self.play(sortie[i * 5 + j].animate.set_fill(opacity=op),
                      run_time=.3, rate_func=linear)
        self.wait(.8)

        cap = T("moitié moins de pixels, zéro paramètre appris", 24, ACCENT)\
            .to_edge(DOWN, buff=0.55)
        self.play(Write(cap), run_time=1.4)
        self.wait(2.0)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=.5)


class S6_Hierarchie(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "5. Empiler : du détail à l'abstraction")
        m1 = image_mob("carte_contours.png", 3.4).shift(LEFT * 4.3)
        m2 = image_mob("carte_textures.png", 2.7).next_to(m1, RIGHT, buff=.7)
        m3 = image_mob("carte_parties.png", 2.1).next_to(m2, RIGHT, buff=.7)
        t1 = T("couche 1\ncontours", 20, PRIMARY).next_to(m1, DOWN, buff=.3)
        t2 = T("couche 2\ntextures", 20, ORANGE).next_to(m2, DOWN, buff=.3)
        t3 = T("couche 3\nparties d'objets", 20, SECOND).next_to(m3, DOWN, buff=.3)
        for m in (m1, m2, m3):
            self.add_subcaption("Des cartes de plus en plus abstraites", duration=2)
        self.play(FadeIn(m1, shift=UP * .3), Write(t1), run_time=1.2)
        self.wait(.6)
        self.play(FadeIn(m2, shift=UP * .3), Write(t2), run_time=1.0)
        self.wait(.6)
        self.play(FadeIn(m3, shift=UP * .3), Write(t3), run_time=1.0)
        fleches = VGroup(*[Arrow(m1.get_right(), m2.get_left(), color=GRIS, buff=.08),
                           Arrow(m2.get_right(), m3.get_left(), color=GRIS, buff=.08)])
        self.play(Create(fleches), run_time=.9)
        note = T("chaque couche voit plus large : le champ récepteur grandit",
                 22, TEXTE).to_edge(DOWN, buff=.55)
        self.add_subcaption("Le champ recepteur grandit couche apres couche", duration=3)
        self.play(Write(note), run_time=1.4)
        self.wait(2.0)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=.5)


class S7_FlattenDense(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "6. Aplatir, puis décider")
        # 4x4 -> vecteur de 16
        g = carte_grille(vec16.reshape(4, 4), 0.42, cmap="grey").to_edge(LEFT, buff=1.0)
        t_g = T("carte finale 4×4", 20, GRIS).next_to(g, UP, buff=.2)
        self.play(Create(g), Write(t_g), run_time=1.4)
        self.wait(.6)

        vec = VGroup()
        for v in vec16:
            cell = Square(side_length=.30, stroke_width=.6, stroke_color=PRIMARY,
                          fill_color=PRIMARY, fill_opacity=.15 + .7 * abs(v) / max(abs(vec16).max(), 1e-9))
            vec.add(cell)
        vec.arrange(DOWN, buff=.08).next_to(g, RIGHT, buff=1.6)
        t_v = T("16 valeurs", 20, PRIMARY).next_to(vec, UP, buff=.2)
        fleches = VGroup(*[Arrow(g[i].get_right(), vec[i].get_left(),
                                 color=GRIS, stroke_width=2, buff=.06)
                           for i in range(16)])
        self.add_subcaption("On aplatit la carte en un vecteur de 16 valeurs", duration=3)
        self.play(Create(vec), Write(t_v), run_time=1.2)
        self.play(*[GrowArrow(a) for a in fleches], run_time=1.2)
        self.wait(1.0)

        # couche dense 16 -> 8
        cachee = VGroup()
        for v in h8:
            d = Dot(radius=.14, color=SECOND,
                    fill_opacity=.3 + .7 * abs(v) / max(abs(h8).max(), 1e-9))
            cachee.add(d)
        cachee.arrange(DOWN, buff=.32).next_to(vec, RIGHT, buff=1.9)
        t_h = T("8 neurones", 20, SECOND).next_to(cachee, UP, buff=.2)
        e1 = VGroup(*[Line(vec[i].get_right(), cachee[j].get_left(),
                           color=GRIS, stroke_width=1)
                      for i in range(16) for j in range(8)])
        self.play(Create(cachee), Write(t_h), run_time=1.0)
        self.play(Create(e1), run_time=1.0)
        # impulsion : les activations se propagent
        pulses = VGroup(*[Dot(radius=.07, color=ACCENT)
                          .move_to(vec[i].get_right()) for i in range(16)])
        self.add_subcaption("Chaque connexion porte un poids appris", duration=3)
        self.play(*[pulses[i].animate.move_to(cachee[i % 8].get_left())
                    for i in range(16)], run_time=1.2)
        self.play(FadeOut(pulses), run_time=.3)
        self.wait(.8)

        # sortie 3 neurones
        sortie = VGroup(*[Dot(radius=.20, color=ORANGE, fill_opacity=.35)
                          for _ in range(3)]).arrange(DOWN, buff=.7)\
            .next_to(cachee, RIGHT, buff=2.2)
        t_s = T("3 sorties", 20, ORANGE).next_to(sortie, UP, buff=.2)
        e2 = VGroup(*[Line(cachee[j].get_right(), sortie[k].get_left(),
                           color=GRIS, stroke_width=1)
                      for j in range(8) for k in range(3)])
        self.play(Create(sortie), Write(t_s), run_time=1.0)
        self.play(Create(e2), run_time=.9)
        p2 = VGroup(*[Dot(radius=.07, color=ACCENT)
                      .move_to(cachee[j].get_right()) for j in range(8)])
        self.play(*[p2[j].animate.move_to(sortie[j % 3].get_left()) for j in range(8)],
                  run_time=1.1)
        self.play(FadeOut(p2), run_time=.3)
        note = T("chaque flèche porte un poids appris", 22, TEXTE).to_edge(DOWN, buff=.55)
        self.play(Write(note), run_time=1.2)
        self.wait(1.8)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=.5)


class S8_Softmax(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "7. La décision")
        soft = np.array(D["softmax"])
        noms = ["classe A", "classe B", "classe C"]
        axes = Axes(x_range=[-0.5, 2.5, 1], y_range=[0, 1, .25],
                    x_length=6.5, y_length=3.6,
                    axis_config={"stroke_color": GRIS, "stroke_width": 1.5},
                    tips=False).shift(DOWN * .6)
        lab_y = T("probabilité", 20, GRIS).rotate(90 * DEGREES)\
            .next_to(axes.y_axis, LEFT, buff=.3)
        self.play(Create(axes), Write(lab_y), run_time=1.2)

        barres = VGroup()
        vals = T("  ".join(f"{n} : {v:.1%}" for n, v in zip(noms, soft)),
                 22, TEXTE).to_edge(UP, buff=1.35)
        for i, (n, v) in enumerate(zip(noms, soft)):
            b = Rectangle(width=.9, height=3.2 * v,
                          fill_color=PRIMARY if i != int(D["argmax"]) else ACCENT,
                          fill_opacity=.85, stroke_width=0)
            b.move_to(axes.c2p(i, 0), aligned_edge=UP)
            barres.add(b)
        self.add_subcaption("Le softmax transforme les scores en probabilites", duration=3)
        self.play(GrowFromEdge(barres, DOWN), FadeIn(vals), run_time=1.6)
        self.wait(1.4)

        gagnant = SurroundingRectangle(barres[int(D["argmax"])], color=ACCENT, buff=.1)
        arg = T("la décision : " + noms[int(D["argmax"])], 28, ACCENT, True)\
            .to_edge(DOWN, buff=.6)
        self.add_subcaption("La classe la plus probable gagne", duration=3)
        self.play(Create(gagnant), Write(arg), run_time=1.4)
        self.wait(1.6)

        # recapitulatif du pipeline
        self.play(FadeOut(VGroup(e, axes, lab_y, barres, vals, gagnant, arg)),
                  run_time=.6)
        pipeline = VGroup(*[
            T(w, 22, c, True) for w, c in
            [("image", GRIS), ("conv", PRIMARY), ("ReLU", SECOND),
             ("pool", ORANGE), ("conv", PRIMARY), ("ReLU", SECOND),
             ("pool", ORANGE), ("flatten", VIOLET), ("dense", VIOLET),
             ("softmax", ACCENT)]])
        pipeline.arrange(RIGHT, buff=.38).to_edge(UP, buff=1.0)
        self.play(*[FadeIn(w, shift=UP * .2) for w in pipeline],
                  run_time=1.4, lag_ratio=.12)
        self.wait(1.6)
        fin = T("Le même noyau, réutilisé partout : c'est toute l'idée du CNN.",
                26, PRIMARY, True).to_edge(DOWN, buff=.8)
        self.add_subcaption("Le meme noyau, reutilise partout", duration=3)
        self.play(Write(fin), run_time=1.8)
        self.wait(2.4)
        self.play(FadeOut(VGroup(*self.mobjects)), run_time=.6)
