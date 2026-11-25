import fitz, os, json, re, sys

PDF = r"C:\Users\HP\Downloads\EC _ EPD-SDIA-4112 – Statistique Bayésienne__Parco.pdf"
OUT = r"C:\Users\HP\Documents\cours-bayesien-SDIA-4112"
PAGES = os.path.join(OUT, "pages")
os.makedirs(PAGES, exist_ok=True)

doc = fitz.open(PDF)
print("pages:", doc.page_count)
print("metadata:", doc.metadata)

info = []
txt_all = []
for i, page in enumerate(doc):
    t = page.get_text("text").strip()
    imgs = page.get_images(full=True)
    info.append({"page": i + 1, "chars": len(t), "images": len(imgs),
                 "w": round(page.rect.width), "h": round(page.rect.height)})
    txt_all.append(f"\n===== PAGE {i+1} =====\n{t}")

with open(os.path.join(OUT, "src", "text_all.txt"), "w", encoding="utf-8") as f:
    f.write("".join(txt_all))
with open(os.path.join(OUT, "src", "pageinfo.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, indent=1)

withtext = [p["page"] for p in info if p["chars"] > 40]
notext = [p["page"] for p in info if p["chars"] <= 40]
print("pages with text layer:", len(withtext))
print("pages without text:", len(notext))
print("first 40 with text:", withtext[:40])
print("first 40 without:", notext[:40])
tot = sum(p["chars"] for p in info)
print("total chars extracted:", tot)
