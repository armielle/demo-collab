"""Video : la propagation dans un CNN, de l'image a la sortie dense.

8 scenes independantes. Toutes les valeurs affichees proviennent de
prep_donnees.py (calculees, graine 4112) — aucune valeur inventee.
"""
import json, os
import numpy as np
from manim import *
from manim.utils.color.core import rgb_to_color

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


def melange(c1, c2, t):
    a = np.array(ManimColor(c1).to_rgb())
    b = np.array(ManimColor(c2).to_rgb())
    return rgb_to_color(tuple(a * (1 - t) + b * t))


CMAPS = {"blue": ("#0B1E3A", PRIMARY), "red": ("#2A0F12", ROUGE),
         "green": ("#0E2A18", SECOND), "grey": ("#1A1A1A", "#C8CDD2")}


def carte_grille(m, cellule=0.34, valeurs=True, cmap="blue", opacite=1.0):
    """Grille de carres colories d'apres une matrice, normalisee."""
    m = np.asarray(m, float)
    mn, mx = m.min(), m.max()
    norm = (m - mn) / (mx - mn + 1e-9)
    cmin, cmax = CMAPS[cmap]
    g = VGroup()
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = norm[i, j]
            sq = Square(side_length=cellule, stroke_width=.6, stroke_color=GRIS,
                        fill_color=melange(cmin, cmax, v),
                        fill_opacity=(.10 + .82 * v) * opacite)
            if valeurs:
                vtxt = f"{m[i, j]:.1f}" if abs(m[i, j]) < 10 else f"{m[i, j]:.0f}"
                sq.add(Text(vtxt, font_size=10, font=MONO,
                            color=TEXTE if v > .45 else "#9AA4AE").move_to(sq))
            g.add(sq)
    g.arrange_in_grid(rows=m.shape[0], cols=m.shape[1], buff=0)
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
        img = image_mob("entree.png", 3.7).shift(DOWN * 1.05)
        titre = T("De l'image à la décision", 46, PRIMARY, True).to_edge(UP, buff=0.7)
        sous = T("la propagation avant dans un réseau convolutif", 26, TEXTE)
        sous.next_to(titre, DOWN, buff=0.4)
        credit = T("Par Dr. Noulapeu Ngaffo Armielle", 24, SECOND)
        credit.next_to(sous, DOWN, buff=0.25)
        ecole = T("ENSPD — Université de Douala", 19, GRIS)
        ecole.next_to(credit, DOWN, buff=0.12)
        self.add_subcaption("De l'image a la decision", duration=2)
        self.play(FadeIn(titre, shift=DOWN * .3), run_time=1.2)
        self.play(FadeIn(sous), run_time=.8)
        self.play(Write(credit), run_time=1.2)
        self.play(FadeIn(ecole), run_time=.7)
        self.wait(1.2)
        self.add_subcaption("Voici notre image de travail", duration=2)
        self.play(FadeIn(img, scale=1.08), run_time=1.5)
        self.wait(1.6)
        accroche = T("Comment 9 nombres suffisent-ils à décider ?", 28, ACCENT)
        accroche.to_edge(DOWN, buff=0.55)
        self.add_subcaption("Comment 9 nombres suffisent-ils a decider ?", duration=3)
        self.play(Write(accroche), run_time=1.6)
        self.wait(3.2)
        self.play(FadeOut(Group(titre, sous, credit, ecole, accroche, img)),
                  run_time=.6)



class S2_ImageTenseur(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "1. Une image, c'est des nombres")
        g = carte_grille(pet, 0.34).move_to(ORIGIN)
        self.add_subcaption("Chaque pixel est un nombre", duration=2)
        self.play(Create(g), run_time=1.6)
        self.wait(.8)
        px = SurroundingRectangle(g[5 * 12 + 6], color=ACCENT, buff=.03)
        lbl = T("ce pixel vaut une intensité", 22, ACCENT).to_edge(DOWN, buff=0.5)
        self.add_subcaption("Ce pixel vaut une intensite", duration=2)
        self.play(Create(px), Write(lbl), run_time=1.2)
        self.wait(1.4)
        note = T("et rien d'autre : pas de structure, pas de sens", 22, GRIS)\
            .next_to(lbl, UP, buff=.25)
        self.play(Write(note), run_time=1.0)
        self.wait(2.6)
        self.play(FadeOut(Group(*self.mobjects)), run_time=.5)



class S2b_MLPvsCNN(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "Pourquoi pas un réseau dense ?")
        gauche = T("48 milliards", 42, ROUGE, True).shift(LEFT * 3.4 + UP * .6)
        g_lab = T("paramètres d'une couche dense\nsur une photo de 24 mégapixels", 20, GRIS)
        g_lab.next_to(gauche, DOWN, buff=.4)
        droite = T("1 792", 42, SECOND, True).shift(RIGHT * 3.4 + UP * .6)
        d_lab = T("paramètres d'une couche\nconvolutive 3×3, 64 filtres", 20, GRIS)
        d_lab.next_to(droite, DOWN, buff=.4)
        self.play(Write(gauche), FadeIn(g_lab), run_time=1.4)
        self.add_subcaption("Une couche dense sur une photo : des dizaines de milliards "
                            "de parametres", duration=3)
        self.wait(1.6)
        self.play(Write(droite), FadeIn(d_lab), run_time=1.4)
        self.wait(1.0)
        fleche = CurvedArrow(gauche.get_bottom(), droite.get_bottom(),
                             color=GRIS, angle=-.55)
        ratio = T("≈ 27 000 fois moins de paramètres", 30, ACCENT, True)
        ratio.next_to(fleche, DOWN, buff=.4)
        self.add_subcaption("Le partage de poids divise le compte par des dizaines "
                            "de milliers", duration=3)
        self.play(Create(fleche), Write(ratio), run_time=1.6)
        self.wait(2.2)
        fin = T("C'est pour cela que les CNN existent.", 26, PRIMARY, True)
        fin.to_edge(DOWN, buff=.6)
        self.play(Write(fin), run_time=1.4)
        self.wait(2.4)
        self.play(FadeOut(Group(e, gauche, g_lab, droite, d_lab, fleche, ratio, fin)),
                  run_time=.6)


class S6b_ChampRecepteur(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "Ce que voit chaque couche")
        axes = Axes(x_range=[0, 10, 2], y_range=[0, 24, 4],
                    x_length=8.2, y_length=3.8,
                    axis_config={"stroke_color": GRIS, "stroke_width": 1.5},
                    tips=False).shift(DOWN * .6)
        xlab = T("nombre de couches (noyau 3×3)", 20, GRIS)
        xlab.next_to(axes.x_axis, DOWN, buff=.25)
        ylab = T("champ récepteur", 20, GRIS).rotate(90 * DEGREES)
        ylab.next_to(axes.y_axis, LEFT, buff=.3)
        self.play(Create(axes), Write(xlab), Write(ylab), run_time=1.4)
        self.wait(.6)

        etapes = [(1, 3, "3×3"), (3, 7, "7×7"), (5, 11, "11×11"), (10, 21, "21×21")]
        self.add_subcaption("Chaque couche elargit le champ recepteur de deux pixels "
                            "de chaque cote", duration=4)
        for i, (nx, ny, txt) in enumerate(etapes):
            pt = Dot(axes.c2p(nx, ny), radius=.09, color=PRIMARY)
            lb = T(f"{txt}", 22, ACCENT, True).next_to(pt, UR, buff=.15)
            self.play(GrowFromCenter(pt), Write(lb), run_time=.9)
            self.wait(.7)
        courbe = axes.plot(lambda x: 1 + 2 * x, x_range=[0.01, 10])
        self.play(Create(courbe), run_time=1.4)
        self.wait(1.0)
        fin = T("Trois noyaux 3×3 voient un champ de 7×7 —\nen apprenant davantage "
                "en chemin.", 24, TEXTE).to_edge(DOWN, buff=.55)
        self.add_subcaption("Empiler plutot qu'agrandir le noyau", duration=3)
        self.play(Write(fin), run_time=1.6)
        self.wait(2.4)
        self.play(FadeOut(Group(*self.mobjects)), run_time=.6)



class S3_Convolution(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "2. Le noyau de convolution")
        noyau = np.array(D["noyau_sobel"])
        g = carte_grille(pet, 0.30).scale(.8).to_edge(LEFT, buff=.6)
        g.shift(DOWN * .5)
        for i in range(144):
            g[i].set_opacity(.22)
        self.play(Create(g), run_time=1.2)

        k = carte_grille(noyau, 0.42, cmap="red").next_to(g, RIGHT, buff=1.1)
        k.shift(DOWN * .5)
        k_title = T("noyau 3x3", 20, ROUGE).next_to(k, UP, buff=.2)
        self.play(FadeIn(k), Write(k_title), run_time=1.0)
        self.wait(.6)

        cell = g[0].width
        fen = Square(side_length=3 * cell, color=ACCENT, stroke_width=2.6,
                     fill_opacity=0)
        fen.move_to(g[13].get_center())          # centre du bloc (0,0)
        self.play(Create(fen), run_time=.8)

        # calcul detaille : 3 lignes de 3 produits, puis la somme
        zone = pet[:3, :3] * noyau
        vals = [f"{v:+.1f}" for v in zone.ravel()]
        lignes = VGroup(*[
            T("  ".join(vals[r * 3:(r + 1) * 3]), 19, TEXTE) for r in range(3)])
        lignes.arrange(DOWN, buff=.12).align_to(k, LEFT).shift(DOWN * 2.0)
        total = D["premier_calcul"]["somme"]
        somme = T(f"= {total:.1f}", 30, ACCENT, True).next_to(lignes, DOWN, buff=.2)
        self.add_subcaption("On multiplie la zone par le noyau, puis on additionne",
                            duration=3)
        self.play(Write(lignes), run_time=1.3)
        self.play(Write(somme), run_time=.7)
        self.wait(1.4)

        s = carte_grille(sort, 0.24, valeurs=False, cmap="green")
        s.next_to(k, RIGHT, buff=1.5)
        s.shift(UP * .4)
        s_title = T("carte d'activation", 20, SECOND).next_to(s, UP, buff=.2)
        self.play(Create(s), Write(s_title), run_time=1.2)
        v0 = abs(sort[0, 0]) / max(abs(sort).max(), 1e-9)
        s[0].set_fill(melange(*CMAPS["green"], v0), opacity=.10 + .82 * v0)
        self.wait(1.0)

        positions = [(0, 1), (0, 2), (1, 0), (2, 1), (3, 3), (4, 2), (2, 4), (1, 3)]
        self.add_subcaption("Le noyau balaie toute l'image", duration=3)
        for (i, j) in positions:
            self.play(fen.animate.move_to(g[(i + 1) * 12 + (j + 1)].get_center()),
                      run_time=.42, rate_func=linear)
            v = abs(sort[i, j]) / max(abs(sort).max(), 1e-9)
            self.play(s[i * 10 + j].animate.set_fill(
                melange(*CMAPS["green"], v), opacity=.10 + .82 * v),
                run_time=.22, rate_func=linear)
        self.wait(.6)

        # la carte reelle, seule a l'ecran
        self.play(FadeOut(Group(e, g, k, k_title, fen, lignes, somme, s, s_title)),
                  run_time=.6)
        vraie = image_mob("carte_contours.png", 4.4)
        lbl = T("carte d'activation complete (calculee)", 24, SECOND)
        lbl.next_to(vraie, UP, buff=.35)
        self.add_subcaption("La carte complete, calculee sur l'image reelle",
                            duration=3)
        partage = T("les m\u00eames 9 poids, r\u00e9utilis\u00e9s des dizaines de milliers de fois", 22, ACCENT)
        partage.next_to(vraie, DOWN, buff=.35)
        self.play(FadeIn(vraie, shift=UP * .3), Write(lbl), run_time=1.4)
        self.play(Write(partage), run_time=1.3)
        self.wait(3.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=.6)



class S4_ReLU(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "3. ReLU : couper ce qui est négatif")
        g = carte_grille(sort, 0.30, cmap="grey").move_to(ORIGIN).shift(LEFT * 1.0)
        for i in range(100):
            if sort.ravel()[i] < 0:
                g[i][0].set_fill(ROUGE, opacity=.55)
        self.play(Create(g), run_time=1.4)
        lbl = T("négatifs", 22, ROUGE).next_to(g, DOWN, buff=.4)
        self.play(Write(lbl), run_time=.8)
        self.wait(.8)

        formule = T("ReLU(x) = max(0, x)", 34, PRIMARY, True)\
            .to_edge(RIGHT, buff=.8).shift(UP * .6)
        self.add_subcaption("Les valeurs negatives deviennent zero", duration=3)
        self.play(Write(formule), run_time=1.4)
        self.wait(1.0)

        for i in range(100):
            if sort.ravel()[i] < 0:
                g[i][0].set_fill(GRIS, opacity=.10)
        n_txt = T(f"{D['relu_negatifs']} cellules sur {D['relu_total']} remises à zéro",
                  22, ORANGE).next_to(formule, DOWN, buff=.5)
        self.play(Write(n_txt), run_time=1.2)
        self.wait(1.6)
        nl = T("sans elle, empiler des couches ne changerait rien :", 22, TEXTE)
        nl2 = T("une suite de produits reste un produit.", 22, TEXTE)
        VGroup(nl, nl2).arrange(DOWN, buff=.15).next_to(n_txt, DOWN, buff=.5)
        self.play(Write(nl), Write(nl2), run_time=1.6)
        self.wait(2.6)
        self.play(FadeOut(Group(*self.mobjects)), run_time=.5)


class S5_Pooling(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "4. Le pooling : réduire, sans apprendre")
        g = carte_grille(sort[:10, :10], 0.30, cmap="grey")\
            .to_edge(LEFT, buff=.9).shift(DOWN * .4)
        self.play(Create(g), run_time=1.4)
        cell = g[0].width
        fen = Square(side_length=2 * cell, color=ACCENT, stroke_width=2.6,
                     fill_opacity=0)
        fen.move_to(g[11].get_center())          # centre du bloc (0,0)
        self.play(Create(fen), run_time=.7)

        sortie = carte_grille(pool5, 0.40, cmap="green").next_to(g, RIGHT, buff=2.4)
        sortie.shift(DOWN * .4)
        s_title = T("5×5", 24, SECOND).next_to(sortie, UP, buff=.2)
        self.play(Create(sortie), Write(s_title), run_time=1.2)

        self.add_subcaption("Dans chaque bloc 2x2, on garde le maximum", duration=3)
        positions = [(0, 0), (0, 2), (1, 1), (2, 0), (3, 2), (4, 1)]
        for (i, j) in positions:
            self.play(fen.animate.move_to(
                g[(2 * i + 1) * 10 + (2 * j + 1)].get_center()),
                run_time=.38, rate_func=linear)
            v = pool5[i, j] / max(pool5.max(), 1e-9)
            self.play(sortie[i * 5 + j].animate.set_fill(
                melange(*CMAPS["green"], v), opacity=.10 + .82 * v),
                run_time=.28, rate_func=linear)
        self.wait(.8)

        cap = T("moitié moins de pixels, zéro paramètre appris", 24, ACCENT)\
            .to_edge(DOWN, buff=0.55)
        self.play(Write(cap), run_time=1.4)
        self.wait(1.6)
        cap2 = T("les architectures modernes lui préfèrent souvent un stride de 2",
                 21, GRIS).next_to(cap, UP, buff=.3)
        self.play(Write(cap2), run_time=1.3)
        self.wait(2.6)
        self.play(FadeOut(Group(*self.mobjects)), run_time=.5)


class S6_Hierarchie(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "5. Empiler : du détail à l'abstraction")
        m1 = image_mob("carte_contours.png", 3.4).shift(LEFT * 4.2 + DOWN * .5)
        m2 = image_mob("carte_textures.png", 2.7).next_to(m1, RIGHT, buff=.6)
        m2.shift(DOWN * .5)
        m3 = image_mob("carte_parties.png", 2.1).next_to(m2, RIGHT, buff=.6)
        m3.shift(DOWN * .5)
        t1 = T("couche 1 : contours", 20, PRIMARY).next_to(m1, DOWN, buff=.3)
        t2 = T("couche 2 : textures", 20, ORANGE).next_to(m2, DOWN, buff=.3)
        t3 = T("couche 3 : parties", 20, SECOND).next_to(m3, DOWN, buff=.3)
        self.add_subcaption("Des cartes de plus en plus abstraites", duration=3)
        self.play(FadeIn(m1, shift=UP * .3), Write(t1), run_time=1.2)
        self.wait(.6)
        self.play(FadeIn(m2, shift=UP * .3), Write(t2), run_time=1.0)
        self.wait(.6)
        self.play(FadeIn(m3, shift=UP * .3), Write(t3), run_time=1.0)
        fleches = VGroup(Arrow(m1.get_right(), m2.get_left(), color=GRIS, buff=.08),
                         Arrow(m2.get_right(), m3.get_left(), color=GRIS, buff=.08))
        self.play(Create(fleches), run_time=.9)
        note = T("chaque couche voit plus large : le champ récepteur grandit",
                 22, TEXTE).to_edge(DOWN, buff=.55)
        self.add_subcaption("Le champ recepteur grandit couche apres couche", duration=3)
        self.play(Write(note), run_time=1.4)
        self.wait(3.2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=.5)


class S7_FlattenDense(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "6. Aplatir, puis decider")
        g = carte_grille(vec16.reshape(4, 4), 0.34, cmap="grey")
        g.to_edge(LEFT, buff=1.1)
        g.shift(DOWN * .3)
        t_g = T("carte finale 4x4", 20, GRIS).next_to(g, UP, buff=.25)
        self.play(Create(g), Write(t_g), run_time=1.4)
        self.wait(.6)

        vec = VGroup()
        for v in vec16:
            vec.add(Square(side_length=.17, stroke_width=.6, stroke_color=PRIMARY,
                           fill_color=PRIMARY,
                           fill_opacity=.15 + .7 * abs(v) / max(abs(vec16).max(), 1e-9)))
        vec.arrange(DOWN, buff=.05).next_to(g, RIGHT, buff=1.2)
        vec.set_y(g.get_y())
        t_v = T("16 valeurs", 20, PRIMARY).next_to(vec, UP, buff=.25)
        fleches = VGroup(*[Line(g[i].get_right(), vec[i].get_left(),
                                color=GRIS, stroke_width=1.5) for i in range(16)])
        self.add_subcaption("On aplatit la carte en un vecteur de 16 valeurs",
                            duration=3)
        self.play(Create(vec), Write(t_v), run_time=1.2)
        self.play(Create(fleches), run_time=1.2)
        self.wait(.6)
        resume = T("16 nombres résument toute l'image", 22, ACCENT)
        resume.next_to(t_v, UP, buff=.35)
        self.play(Write(resume), run_time=1.2)
        self.wait(1.4)

        cachee = VGroup()
        for v in h8:
            cachee.add(Dot(radius=.09, color=SECOND,
                           fill_opacity=.3 + .7 * abs(v) / max(abs(h8).max(), 1e-9)))
        cachee.arrange(DOWN, buff=.18).next_to(vec, RIGHT, buff=1.5)
        cachee.set_y(g.get_y())
        t_h = T("8 neurones", 20, SECOND).next_to(cachee, UP, buff=.25)
        e1 = VGroup(*[Line(vec[i].get_right(), cachee[j].get_left(),
                           color=GRIS, stroke_width=1)
                      for i in range(16) for j in range(8)])
        self.play(Create(cachee), Write(t_h), run_time=1.0)
        self.play(Create(e1), run_time=1.0)
        pulses = VGroup(*[Dot(radius=.06, color=ACCENT)
                          .move_to(vec[i].get_right()) for i in range(16)])
        self.add_subcaption("Chaque connexion porte un poids appris", duration=3)
        self.play(*[pulses[i].animate.move_to(cachee[i % 8].get_left())
                    for i in range(16)], run_time=1.2)
        self.play(FadeOut(pulses), run_time=.3)
        self.wait(.8)

        sortie = VGroup(*[Dot(radius=.15, color=ORANGE, fill_opacity=.35)
                          for _ in range(3)]).arrange(DOWN, buff=.45)
        sortie.next_to(cachee, RIGHT, buff=1.6)
        sortie.set_y(g.get_y())
        t_s = T("3 sorties", 20, ORANGE).next_to(sortie, UP, buff=.25)
        e2 = VGroup(*[Line(cachee[j].get_right(), sortie[k].get_left(),
                           color=GRIS, stroke_width=1)
                      for j in range(8) for k in range(3)])
        self.play(Create(sortie), Write(t_s), run_time=1.0)
        self.play(Create(e2), run_time=.9)
        p2 = VGroup(*[Dot(radius=.06, color=ACCENT)
                      .move_to(cachee[j].get_right()) for j in range(8)])
        self.play(*[p2[j].animate.move_to(sortie[j % 3].get_left()) for j in range(8)],
                  run_time=1.1)
        self.play(FadeOut(p2), run_time=.3)
        note = T("chaque fleche porte un poids appris", 22, TEXTE)
        note.to_edge(DOWN, buff=.55)
        self.play(Write(note), run_time=1.2)
        self.wait(1.8)
        self.play(FadeOut(Group(*self.mobjects)), run_time=.5)



class S8_Softmax(Scene):
    def construct(self):
        self.camera.background_color = BG
        e = entete(self, "7. La décision")
        soft = np.array(D["softmax"])
        noms = ["classe A", "classe B", "classe C"]
        axes = Axes(x_range=[-0.5, 2.5, 1], y_range=[0, 1, .25],
                    x_length=6.5, y_length=3.6,
                    axis_config={"stroke_color": GRIS, "stroke_width": 1.5},
                    tips=False).shift(DOWN * .7)
        lab_y = T("probabilité", 20, GRIS).rotate(90 * DEGREES)\
            .next_to(axes.y_axis, LEFT, buff=.3)
        self.play(Create(axes), Write(lab_y), run_time=1.2)

        barres = VGroup()
        vals = T("   ".join(f"{n} : {v:.1%}" for n, v in zip(noms, soft)),
                 22, TEXTE).to_edge(UP, buff=1.35)
        for i, (n, v) in enumerate(zip(noms, soft)):
            b = Rectangle(width=.9, height=max(3.2 * v, .02),
                          fill_color=PRIMARY if i != int(D["argmax"]) else ACCENT,
                          fill_opacity=.85, stroke_width=0)
            b.move_to(axes.c2p(i, 0), aligned_edge=DOWN)
            barres.add(b)
        self.add_subcaption("Le softmax transforme les scores en probabilites",
                            duration=3)
        self.play(GrowFromEdge(barres, DOWN), FadeIn(vals), run_time=1.6)
        self.wait(1.4)

        lab_x = VGroup(*[T(n, 20, TEXTE).next_to(axes.c2p(i, 0), DOWN, buff=.25)
                         for i, n in enumerate(noms)])
        self.play(FadeIn(lab_x), run_time=.7)
        self.wait(.6)

        gagnant = SurroundingRectangle(barres[int(D["argmax"])], color=ACCENT, buff=.1)
        arg = T("la décision : " + noms[int(D["argmax"])], 28, ACCENT, True)\
            .to_edge(DOWN, buff=.6)
        self.add_subcaption("La classe la plus probable gagne", duration=3)
        self.play(Create(gagnant), Write(arg), run_time=1.4)
        self.wait(1.6)

        self.play(FadeOut(Group(e, axes, lab_y, barres, vals, lab_x, gagnant, arg)),
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
        self.wait(1.8)
        fin2 = T("Vous savez maintenant lire une architecture de bout en bout.",
                 22, TEXTE).next_to(fin, DOWN, buff=.35)
        self.play(Write(fin2), run_time=1.6)
        self.wait(2.6)
        self.play(FadeOut(Group(*self.mobjects)), run_time=.6)
