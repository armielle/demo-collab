"""Extrait le texte des frames de TP (gitN_d.tex) pour rediger les enonces."""
import re
import pathlib
import sys

PARTS = pathlib.Path(r"C:\Users\HP\Documents\cours-aiops-GIT-5534\slides\parts")

KEEP = re.compile(r"\\(textbf|emph|texttt|underline|py|file|item|begin|end|toprule|midrule|bottomrule|addlinespace|hline)\b")


def clean(s: str) -> str:
    s = re.sub(r"\\py\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\file\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\texttt\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^]]*\])?", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?", "", s)
    s = s.replace("{", "").replace("}", "").replace("$", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def frames(txt):
    out = []
    for m in re.finditer(r"\\begin\{frame\}", txt):
        # fin de la frame
        end = txt.find("\\end{frame}", m.start())
        if end < 0:
            continue
        bloc = txt[m.start():end]
        title = re.search(r"\\begin\{frame\}(\[[^]]*\])?\{", bloc)
        t = ""
        if title:
            i = title.end()
            depth = 1
            j = i
            while j < len(bloc) and depth:
                if bloc[j] == "{":
                    depth += 1
                elif bloc[j] == "}":
                    depth -= 1
                j += 1
            t = bloc[i:j - 1]
        out.append((clean(t), clean(bloc[j:] if title else bloc)))
    return out


def main():
    n = sys.argv[1]
    txt = (PARTS / f"git{n}_d.tex").read_text(encoding="utf-8")
    for title, body in frames(txt):
        print("=" * 70)
        print("TITRE :", title)
        print(body[:1400])


if __name__ == "__main__":
    main()
