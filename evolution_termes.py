from __future__ import annotations

import csv
import html
import math
import os
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import spacy
except ImportError:  # message explicite si spaCy n'est pas installé
    spacy = None

# ==========================================================
# CONFIGURATION A MODIFIER
# ==========================================================

# Placez ce script dans le même dossier que le dossier corpus_Charcot,
# ou remplacez par un chemin complet, par exemple :
# CORPUS_PATH = Path('/Users/vous/Desktop/corpus_Charcot')
CORPUS_PATH = Path("corpus_Autres")
OUTPUT_DIR = Path("sortie_frequences_autres")

# Fichier contenant les expressions régulières à visualiser.
# Format attendu : une expression régulière par ligne.
# Les lignes vides et les lignes commençant par # sont ignorées.
REGEX_FILE = Path("liste_concepts_regex.txt")

# Noms lisibles des concepts, dans le même ordre que les regex du fichier.
# Ces noms servent à produire les noms de colonnes dans Flourish.
# Exemple : "sclérose latérale amyotrophique" devient
# "sclerose_laterale_amyotrophique" dans le CSV.
CONCEPT_LABELS = [
    "épilepsie",
    "anévrisme miliaire",
    "aphasie",
    "arthropathie tabétique",
    "astasie-abasie",
    "ataxie locomotrice progressive",
    "athétose",
    "atrophie musculaire progressive",
    "bulbe rachidien",
    "chorée",
    "clonus",
    "coupe verticale",
    "fugue",
    "glossy skin",
    "hypnose",
    "hystérie",
    "localisation cérébrale",
    "maladie de Parkinson",
    "maladie de Gilles de la Tourette",
    "méthode anatomo-clinique",
    "migraine ophtalmique",
    "pachyméningite cervicale hypertrophique",
    "paralysie agitante",
    "sclérose en plaques disséminées",
    "sclérose latérale amyotrophique",
    "syndrome de Parkinson",
    "syndrome de Gilles de la Tourette",
    "systématisation de la moelle",
    "tabès dorsalis",
    "tabès dorsalis spasmodique",
    "thrombose de l'artère cérébrale postérieure",
    "tic convulsif",
    "tremblement",
]

# Les expressions seront chargées depuis REGEX_FILE dans main().
EXPRESSIONS = []

# Fréquences normalisées en ppm : occurrences par million de tokens.
# Cela correspond au principe décrit dans OBVIE : "parties par million".
NORMALIZATION_BASE = 1_000_000

# Pour un axe logarithmique, les valeurs 0 ne peuvent pas être représentées.
# Le script produit donc aussi un CSV log10(ppm), où les zéros sont laissés vides.
# Pour Flourish, la solution recommandée reste d'importer le CSV en ppm
# et de régler l'axe Y sur Log dans l'interface.
WRITE_LOG10_PPM_CSV = True
LOG_ZERO_AS_BLANK = True

# Pour le CSV principal en ppm destiné à un axe logarithmique dans Flourish :
# les valeurs 0 sont laissées vides. Cela évite que Flourish trace des
# segments vers le bas de l'axe log, puisque 0 n'est pas représentable
# sur une échelle logarithmique. Les zéros restent conservés dans
# frequences_detaillees.csv.
PPM_ZERO_AS_BLANK_FOR_LOG_AXIS = True

# True = neutralise les accents : "hystérie" matchera aussi "hysterie".
# Recommandé pour des corpus OCRisés.
STRIP_ACCENTS = True

# Nombre de processus parallèles.
# None = choix automatique. Mettez 1 si vous voulez une exécution simple, sans parallélisation.
N_WORKERS = 1

# True = utilise le tokenizer français de spaCy pour calculer le dénominateur
# des fréquences en ppm, c'est-à-dire total_tokens.
# La détection des concepts reste faite par les regex de REGEX_FILE.
# Installation si nécessaire : pip install spacy
USE_SPACY_TOKENIZER = True

# spaCy bloque par défaut les textes de plus d'environ 1 000 000 caractères.
# Ici on utilise seulement le tokenizer, sans parser ni NER : on peut donc
# augmenter cette limite sans changer la méthode de comptage.
SPACY_MAX_LENGTH = 10_000_000

# ==========================================================
# CODE
# ==========================================================

YEAR_RE = re.compile(r"(17|18|19|20)\d{2}")
DATE_WHEN_RE = re.compile(r"<date\b[^>]*\bwhen=[\"']([^\"']+)[\"']", re.IGNORECASE)
# Date du texte original : dans ce corpus, elle se trouve dans
# <teiHeader><profileDesc><creation><date when="YYYY" />.
# On évite de prendre par erreur une date de numérisation, d'encodage ou de modification.
CREATION_DATE_WHEN_RE = re.compile(
    r"<profileDesc\b[^>]*>.*?<creation\b[^>]*>.*?<date\b[^>]*\bwhen=[\"']([^\"']+)[\"']",
    re.IGNORECASE | re.DOTALL,
)
BODY_RE = re.compile(r"<body\b[^>]*>(.*?)</body>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[a-z]+")

# Table rapide pour les accents français les plus fréquents.
# Pour des besoins plus larges, la fonction normalize_text peut aussi passer par unicodedata.
ACCENT_TRANSLATION = str.maketrans({
    "à": "a", "â": "a", "ä": "a", "á": "a", "ã": "a", "å": "a", "ā": "a",
    "ç": "c",
    "é": "e", "è": "e", "ê": "e", "ë": "e", "ē": "e", "ė": "e", "ę": "e",
    "î": "i", "ï": "i", "í": "i", "ì": "i", "ī": "i",
    "ô": "o", "ö": "o", "ó": "o", "ò": "o", "õ": "o", "ø": "o", "ō": "o",
    "û": "u", "ü": "u", "ù": "u", "ú": "u", "ū": "u",
    "ÿ": "y", "ñ": "n",
    "œ": "oe", "æ": "ae",
})


def normalize_text(text: str, strip_accents: bool = STRIP_ACCENTS) -> str:
    """Minuscules, ligatures et accents optionnels."""
    text = text.lower().replace("œ", "oe").replace("æ", "ae")
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    if strip_accents:
        # Rapide pour le français courant.
        text = text.translate(ACCENT_TRANSLATION)
        # Le corpus Charcot utilise surtout des caractères accentués précomposés.
        # La table ci-dessus suffit pour les accents français courants.
    return text


def normalize_for_regex_search(text: str) -> str:
    """Normalise le texte avant l'application des regex."""
    return normalize_text(text)


def tokenize(text: str) -> list[str]:
    """
    Découpe en tokens alphabétiques normalisés.
    Exemple : "l'hystérie" devient ["l", "hysterie"].
    
    Cette fonction reste utilisée pour produire des noms de colonnes simples
    dans les CSV, pas pour compter le dénominateur des ppm quand
    USE_SPACY_TOKENIZER=True.
    """
    return TOKEN_RE.findall(normalize_text(text))


_NLP_FR = None


def get_spacy_fr():
    """Charge seulement le tokenizer français spaCy, sans modèle du langage."""
    if spacy is None:
        raise RuntimeError(
            "spaCy n'est pas installé. Installez-le avec : pip install spacy\n"
            "Le script utilise spacy.blank('fr'), donc aucun modèle fr_core_news_sm "
            "n'est nécessaire pour cette étape."
        )

    global _NLP_FR
    if _NLP_FR is None:
        _NLP_FR = spacy.blank("fr")
        _NLP_FR.max_length = SPACY_MAX_LENGTH
    return _NLP_FR


def count_tokens_for_normalization(text: str) -> int:
    """
    Calcule le nombre de tokens servant au dénominateur des ppm.

    Si USE_SPACY_TOKENIZER=True, utilise le tokenizer français de spaCy.
    Sinon, garde l'ancienne méthode par expression régulière.

    La détection des concepts reste indépendante : elle est toujours faite
    par les regex de REGEX_FILE.
    """
    normalized = normalize_text(text)

    if not USE_SPACY_TOKENIZER:
        return len(TOKEN_RE.findall(normalized))

    nlp = get_spacy_fr()

    # Sécurité supplémentaire : si un fichier dépasse quand même la limite
    # configurée, on augmente la limite juste pour ce texte. C'est acceptable
    # ici parce qu'on utilise seulement le tokenizer de spacy.blank("fr"),
    # sans parser ni NER.
    if len(normalized) >= nlp.max_length:
        nlp.max_length = len(normalized) + 1

    doc = nlp.make_doc(normalized)

    # On ne compte que les tokens alphabétiques pour rester proche
    # de l'ancienne définition, qui ignorait chiffres et ponctuation.
    return sum(1 for token in doc if token.is_alpha)

# slug = nom de colonne simplifié
def make_slug(expression: str) -> str:
    """Nom de colonne simple pour Flourish."""
    return "_".join(tokenize(expression))


def decode_xml(data: bytes) -> str:
    """Décode le XML. Le corpus fourni est en UTF-8."""
    return data.decode("utf-8", errors="replace")


def extract_year(xml_text: str, filename: str) -> str | None:
    """Extrait l'année du texte original.

    Priorité :
    1. <teiHeader><profileDesc><creation><date when="YYYY" />
       C'est la date du texte dans le corpus Charcot.
    2. Secours très limité : année présente dans le nom du fichier.

    On ne cherche plus une année quelconque dans les 5000 premiers caractères,
    car cela peut récupérer par erreur une date de numérisation, d'encodage
    ou de modification du fichier.
    """
    match = CREATION_DATE_WHEN_RE.search(xml_text)
    if match:
        year = YEAR_RE.search(match.group(1))
        if year:
            return year.group(0)

    # Secours limité : seulement si le nom de fichier contient explicitement une année.
    year = YEAR_RE.search(filename)
    if year:
        return year.group(0)

    return None


def extract_body_text(xml_text: str) -> str:
    """
    Extrait le contenu du <body>, supprime les balises, puis décode les entités XML/HTML.
    Cela évite de compter les métadonnées du teiHeader.
    """
    match = BODY_RE.search(xml_text)
    body = match.group(1) if match else xml_text
    body = TAG_RE.sub(" ", body)
    body = html.unescape(body)
    return body


def iter_xml_sources(corpus_path: Path):
    """Parcourt les fichiers XML/TEI depuis un ZIP ou un dossier."""
    valid_suffixes = (".xml", ".tei")

    def keep(name: str) -> bool:
        path = Path(name)
        lower = name.lower()
        if "__macosx" in lower:
            return False
        if any(part.startswith(".") for part in path.parts):
            return False
        if path.name.startswith("._"):
            return False
        return lower.endswith(valid_suffixes)

    if corpus_path.is_file() and corpus_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(corpus_path) as zf:
            for name in zf.namelist():
                if keep(name):
                    yield name, zf.read(name)
    elif corpus_path.is_dir():
        for path in sorted(corpus_path.rglob("*")):
            rel = str(path.relative_to(corpus_path))
            if path.is_file() and keep(rel):
                yield str(path), path.read_bytes()
    else:
        raise FileNotFoundError(f"Chemin introuvable ou non supporté : {corpus_path}")



def normalize_regex_pattern(pattern: str) -> str:
    """
    Normalise une regex lue depuis le fichier REGEX_FILE pour qu'elle soit
    compatible avec le texte de recherche, lui aussi mis en minuscules et
    désaccentué quand STRIP_ACCENTS=True.

    Les espaces littéraux non échappés sont convertis en \\s+ pour permettre
    des retours à la ligne ou des espaces multiples entre les mots.
    """
    pattern = pattern.strip()

    # Dans les regex fournies, cette plage sert à dire "lettre accentuée".
    # Comme le texte est désaccentué avant recherche, \w suffit ici.
    pattern = pattern.replace("À-ÖØ-öø-ÿ", "")
    pattern = pattern.replace("à-öø-ÿ", "")

    pattern = normalize_text(pattern)

    # Les apostrophes typographiques ont déjà été harmonisées par normalize_text.
    # On rend les espaces littéraux plus robustes dans les expressions composées.
    pattern = re.sub(r"(?<!\\) +", r"\\s+", pattern)
    return pattern


def readable_name_from_regex(pattern: str) -> str:
    """Crée un nom lisible approximatif à partir d'une regex."""
    simplified = normalize_regex_pattern(pattern)
    simplified = simplified.replace(r"\s+", " ").replace(r"\s", " ")

    def replace_class(match: re.Match) -> str:
        content = match.group(1)
        if r"\w" in content:
            return "mot"
        for char in content:
            if char.isalpha():
                return char
        return " "

    simplified = re.sub(r"\[([^\]]+)\]", replace_class, simplified)
    simplified = simplified.replace("(?:", "(")
    simplified = simplified.replace("?", " ")
    simplified = simplified.replace("+", " ")
    simplified = simplified.replace("*", " ")
    simplified = simplified.replace("{", " ").replace("}", " ")
    simplified = simplified.replace("|", " ")
    simplified = simplified.replace("(", " ").replace(")", " ")
    simplified = simplified.replace("-", " ")
    tokens = [tok for tok in tokenize(simplified) if tok not in {"s", "w"}]
    return " ".join(tokens[:8]) if tokens else pattern.strip()



def prefix_candidates_from_pattern(pattern: str, label: str) -> list[str]:
    """Trouve un préfixe littéral pour accélérer les recherches regex."""
    normalized = normalize_regex_pattern(pattern)
    literal = []
    escaped = False
    for char in normalized:
        if escaped:
            break
        if char == "\\":
            escaped = True
            break
        if char in "[().?+*{|^$":
            break
        literal.append(char)

    prefix = "".join(literal).strip()
    if len(prefix) >= 2:
        return [prefix]

    label_tokens = tokenize(label)
    if label_tokens:
        return [label_tokens[0]]

    return []

def load_expressions_from_file(regex_file: Path) -> list[dict]:
    """Charge une regex par ligne et prépare les entrées attendues par le reste du script."""
    if not regex_file.exists():
        raise FileNotFoundError(
            f"Fichier de regex introuvable : {regex_file}. "
            "Placez liste_concepts_regex.txt dans le même dossier que ce script, "
            "ou modifiez REGEX_FILE."
        )

    expressions = []
    with regex_file.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            raw_pattern = line.strip()
            if not raw_pattern or raw_pattern.startswith("#"):
                continue

            # Si un nom lisible est défini dans CONCEPT_LABELS, on l'utilise.
            # Sinon, on garde le comportement automatique.
            concept_index = len(expressions)
            if concept_index < len(CONCEPT_LABELS):
                label = CONCEPT_LABELS[concept_index]
            else:
                label = readable_name_from_regex(raw_pattern)

            # Nom de colonne dans le CSV Flourish.
            # Ex. "hystérie" -> "hysterie".
            # On ne préfixe plus avec concept_XX, pour avoir une légende lisible.
            slug = make_slug(label) or f"concept_{line_number:02d}"

            expressions.append({
                "label": label,
                "slug": slug,
                "pattern": normalize_regex_pattern(raw_pattern),
                "raw_pattern": raw_pattern,
                # Préfixe littéral utilisé pour accélérer la recherche.
                "prefixes": prefix_candidates_from_pattern(raw_pattern, label),
            })

    if not expressions:
        raise ValueError(f"Aucune regex trouvée dans : {regex_file}")

    return expressions

def prepare_expressions(expressions: list[dict]) -> list[dict]:
    expr_info = []
    used_slugs = Counter()

    for expr in expressions:
        label = expr["label"]
        pattern = expr["pattern"]
        slug = expr.get("slug") or make_slug(label)
        if not slug:
            raise ValueError(f"Expression vide après normalisation : {label!r}")

        used_slugs[slug] += 1
        if used_slugs[slug] > 1:
            slug = f"{slug}_{used_slugs[slug]}"

        # On ajoute des frontières de mot pour éviter les correspondances à l'intérieur d'un mot.
        bounded_pattern = rf"(?<!\w)(?:{pattern})(?!\w)"
        prefixes = expr.get("prefixes")
        if prefixes is None:
            # Premier mot du libellé, normalisé comme le texte de recherche.
            # Cela sert à trouver rapidement des positions candidates avant d'appliquer la regex.
            label_tokens = tokenize(label)
            prefixes = [label_tokens[0]] if label_tokens else []

        expr_info.append({
            "label": label,
            "slug": slug,
            "pattern": pattern,
            "raw_pattern": expr.get("raw_pattern", pattern),
            "bounded_pattern": bounded_pattern,
            "prefixes": tuple(prefixes),
        })

    return expr_info


def iter_prefix_positions(text: str, prefix: str):
    """Renvoie les positions candidates d'un préfixe littéral dans le texte."""
    start = 0
    while True:
        index = text.find(prefix, start)
        if index == -1:
            break
        yield index
        start = index + 1


def count_expressions_in_text(search_text: str, expr_info: list[dict]) -> Counter:
    """Compte les expressions avec les regex demandées.

    Pour éviter que certaines regex complexes soient lentes sur de gros fichiers,
    on cherche d'abord un préfixe littéral, puis on applique la regex seulement
    aux positions candidates.
    """
    counts = Counter()
    for info in expr_info:
        candidate_positions = set()
        for prefix in info["prefixes"]:
            candidate_positions.update(iter_prefix_positions(search_text, prefix))

        regex = re.compile(info["bounded_pattern"], flags=re.IGNORECASE)

        if not info["prefixes"]:
            counts[info["slug"]] = sum(1 for _ in regex.finditer(search_text))
            continue

        count = 0
        for position in sorted(candidate_positions):
            if regex.match(search_text, position):
                count += 1
        counts[info["slug"]] = count
    return counts


def process_one_file(args):
    """Traite un fichier ; fonction séparée pour permettre la parallélisation."""
    filename, data, expr_info = args
    xml_text = decode_xml(data)
    year = extract_year(xml_text, filename)
    if year is None:
        return {"skipped": (filename, "date introuvable")}

    body_text = extract_body_text(xml_text)

    # On normalise une seule fois le texte du fichier pour la recherche par regex.
    # La détection des concepts reste donc inchangée.
    normalized_text = normalize_text(body_text)
    search_text = normalized_text
    doc_counts = count_expressions_in_text(search_text, expr_info)

    # Le dénominateur des ppm est calculé séparément : soit avec spaCy,
    # soit avec l'ancienne expression régulière si USE_SPACY_TOKENIZER=False.
    total_tokens = count_tokens_for_normalization(body_text)

    return {
        "filename": filename,
        "year": year,
        "total_tokens": total_tokens,
        "counts": dict(doc_counts),
    }


def choose_workers() -> int:
    if N_WORKERS is not None:
        return max(1, int(N_WORKERS))
    cpu = os.cpu_count() or 2
    return max(1, min(cpu - 1, 4))


def normalized_frequency(raw_count: int, total_tokens: int) -> float:
    """Fréquence normalisée en ppm : occurrences par million de tokens."""
    return (raw_count / total_tokens * NORMALIZATION_BASE) if total_tokens else 0.0


def ppm_for_flourish_log_axis(value: float):
    """Valeur en ppm à écrire dans le CSV principal pour Flourish.

    Sur un axe logarithmique, 0 n'est pas représentable. Si
    PPM_ZERO_AS_BLANK_FOR_LOG_AXIS=True, on écrit donc une cellule vide
    au lieu de 0, ce qui crée une absence de point dans Flourish plutôt
    qu'une chute artificielle vers le bas du graphique.
    """
    if value <= 0 and PPM_ZERO_AS_BLANK_FOR_LOG_AXIS:
        return ""
    return round(value, 6)


def log10_or_blank(value: float):
    """Valeur log10 pour le CSV optionnel ; zéro = cellule vide par défaut."""
    if value <= 0:
        return "" if LOG_ZERO_AS_BLANK else 0
    return round(math.log10(value), 6)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expressions = load_expressions_from_file(REGEX_FILE)
    expr_info = prepare_expressions(expressions)

    sources = list(iter_xml_sources(CORPUS_PATH))
    tasks = [(filename, data, expr_info) for filename, data in sources]
    workers = choose_workers()

    if workers == 1:
        results = [process_one_file(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process_one_file, tasks))

    by_year = defaultdict(lambda: {
        "documents": 0,
        "total_tokens": 0,
        "counts": Counter(),
    })

    skipped = []
    processed_files = 0

    for result in results:
        if "skipped" in result:
            skipped.append(result["skipped"])
            continue

        year = result["year"]
        by_year[year]["documents"] += 1
        by_year[year]["total_tokens"] += result["total_tokens"]
        by_year[year]["counts"].update(result["counts"])
        processed_files += 1

    years = sorted(by_year.keys(), key=int)

    # 1) CSV large, prêt pour Flourish : date + une colonne par expression en ppm.
    # ppm = occurrences par million de tokens.
    # Dans Flourish, importez ce fichier puis réglez l'axe Y sur Log.
    flourish_path = OUTPUT_DIR / "frequences_ppm_flourish.csv"
    with flourish_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date"] + [info["slug"] for info in expr_info])
        for year in years:
            total = by_year[year]["total_tokens"]
            row = [year]
            for info in expr_info:
                raw = by_year[year]["counts"][info["slug"]]
                freq = normalized_frequency(raw, total)
                row.append(ppm_for_flourish_log_axis(freq))
            writer.writerow(row)

    # Ancien nom conservé pour compatibilité avec les versions précédentes du script.
    legacy_flourish_path = OUTPUT_DIR / "frequences_normalisees_flourish.csv"
    with legacy_flourish_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date"] + [info["slug"] for info in expr_info])
        for year in years:
            total = by_year[year]["total_tokens"]
            row = [year]
            for info in expr_info:
                raw = by_year[year]["counts"][info["slug"]]
                freq = normalized_frequency(raw, total)
                row.append(ppm_for_flourish_log_axis(freq))
            writer.writerow(row)

    # 1bis) CSV optionnel déjà transformé en log10(ppm).
    # À utiliser seulement si vous ne trouvez pas l'option Log dans Flourish.
    # Attention : l'axe Y représente alors log10(ppm), pas directement des ppm.
    log_flourish_path = OUTPUT_DIR / "frequences_log10_ppm_flourish.csv"
    if WRITE_LOG10_PPM_CSV:
        with log_flourish_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date"] + [info["slug"] for info in expr_info])
            for year in years:
                total = by_year[year]["total_tokens"]
                row = [year]
                for info in expr_info:
                    raw = by_year[year]["counts"][info["slug"]]
                    freq = normalized_frequency(raw, total)
                    row.append(log10_or_blank(freq))
                writer.writerow(row)

    # 2) CSV détaillé : pour vérifier les comptes et documenter la méthode.
    detail_path = OUTPUT_DIR / "frequences_detaillees.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "date", "documents", "total_tokens", "expression", "colonne_flourish",
            "regex", "occurrences", "freq_ppm",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for year in years:
            total = by_year[year]["total_tokens"]
            for info in expr_info:
                raw = by_year[year]["counts"][info["slug"]]
                freq = normalized_frequency(raw, total)
                writer.writerow({
                    "date": year,
                    "documents": by_year[year]["documents"],
                    "total_tokens": total,
                    "expression": info["label"],
                    "colonne_flourish": info["slug"],
                    "regex": info.get("raw_pattern", info["pattern"]),
                    "occurrences": raw,
                    "freq_ppm": round(freq, 6),
                })

    # 3) Contrôle des dates utilisées : un fichier par ligne.
    # Ce fichier sert à vérifier que le regroupement chronologique est correct.
    dates_path = OUTPUT_DIR / "controle_dates.csv"
    with dates_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "date", "total_tokens"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(
            (r for r in results if "skipped" not in r),
            key=lambda r: (int(r["year"]), r["filename"])
        ):
            writer.writerow({
                "filename": result["filename"],
                "date": result["year"],
                "total_tokens": result["total_tokens"],
            })

    # 4) Rapport de contrôle.
    report_path = OUTPUT_DIR / "rapport_controle.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Corpus : {CORPUS_PATH}\n")
        f.write(f"Fichier de regex : {REGEX_FILE}\n")
        f.write(f"Fichiers XML/TEI trouvés : {len(sources)}\n")
        f.write(f"Fichiers traités : {processed_files}\n")
        f.write(f"Nombre de processus : {workers}\n")
        f.write(f"Nombre d'années : {len(years)}\n")
        f.write(f"Années : {', '.join(years)}\n")
        f.write(f"Base de normalisation : {NORMALIZATION_BASE} tokens\n")
        f.write(f"Tokenisation spaCy pour total_tokens : {USE_SPACY_TOKENIZER}\n")
        f.write(f"Accents neutralisés : {STRIP_ACCENTS}\n")
        f.write("\nExpressions régulières :\n")
        for info in expr_info:
            total_occ = sum(by_year[year]["counts"][info["slug"]] for year in years)
            f.write(f"- {info['label']} -> {info['slug']} -> {info.get('raw_pattern', info['pattern'])} -> total occurrences: {total_occ}\n")
        if skipped:
            f.write("\nFichiers ignorés :\n")
            for filename, reason in skipped:
                f.write(f"- {filename}: {reason}\n")
        else:
            f.write("\nAucun fichier ignoré.\n")

    print(f"OK : {flourish_path}")
    print(f"OK : {legacy_flourish_path}")
    if WRITE_LOG10_PPM_CSV:
        print(f"OK : {log_flourish_path}")
    print(f"OK : {detail_path}")
    print(f"OK : {dates_path}")
    print(f"OK : {report_path}")


if __name__ == "__main__":
    main()
