# Script Python pour Smode (outil "Python Script") : écrit le Prompt et le Negative Prompt d'un
# Stream Diffusion Modifier en les adaptant au modèle chargé (SD 1.5 / SD 2.1 / SDXL, anime,
# réaliste, Pony, Illustrious, modèles distillés Turbo / Lightning / Hyper / LCM).
#
# Utilisation :
#   1. Ajouter un outil Python Script (Tools > Python Script), coller ce script.
#   2. Glisser le Stream Diffusion Modifier sur le paramètre "modifier" du script.
#   3. Écrire l'idée dans "idea", en anglais ou en français (voir "translation").
#      Vide = le Prompt actuel du modifier est repris et nettoyé.
#   4. Launch Mode "At Parameter Change" (ou Execute à la main).
#
# Français : l'encodeur CLIP ne comprend que l'anglais. "translation" traduit l'idée et
# "extraNegative" avant de construire le prompt :
#   Off  = jamais ; Auto = si le texte semble français (accents, mots courants) ; On = toujours
#   (pour un texte court sans accent comme "chat noir").
# La traduction passe par Google Translate (en ligne, le texte est envoyé à Google). Chaque
# morceau traduit est gardé dans %LOCALAPPDATA%\SmodePromptAssist\translations.json : un texte
# déjà traduit ne repart pas sur le réseau et reste disponible hors ligne. Les tags avec "_" ou
# commençant par un chiffre (score_9, 1girl) ne sont pas traduits.
#
# Ce que fait le script :
#   - détecte le modèle (champ du modifier, ou "modelOverride") et les LoRA de vitesse ;
#   - ajoute les tags dont la famille a besoin (score_9... pour Pony, masterpiece... pour
#     l'anime, formulation photo pour les modèles réalistes), plus un style optionnel ;
#   - retire la syntaxe A1111 que ce package ne lit pas : poids "(mot:1.2)", "[ ]", "<lora:...>",
#     "BREAK" (ici le texte passe tel quel dans CLIP, les parenthèses deviennent des tokens) ;
#   - garde le prompt sous la limite de CLIP (75 tokens : la suite est coupée sans prévenir) en
#     retirant d'abord les tags de qualité puis de style, jamais l'idée ;
#   - écrit un Negative Prompt adapté et dit s'il sert à quelque chose : il n'est lu qu'en CFG
#     "initialize" ou "full" avec une Guidance Scale > 1 ("self", le défaut, l'ignore).
#
# Mots déclencheurs des LoRA ("loraTriggers") : placés en tête du prompt, jamais traduits ni
# coupés. Cherchés pour chaque LoRA du modifier (les LoRA de vitesse Hyper / LCM / Lightning... et
# l'offset noise n'en ont pas) :
#   - dépôt HF ("auteur/nom" ou "auteur/nom::fichier") : "instance_prompt" de la fiche du modèle,
#     sinon une ligne "Trigger word: ..." du README ;
#   - fichier local (.safetensors) : métadonnée "modelspec.trigger_phrase", sinon les fichiers
#     d'infos posés à côté ("nom.civitai.info" de Civitai Helper, "nom.json" d'A1111). Sans rien de
#     tout ça, les tags les plus fréquents de l'entraînement sont affichés comme suggestion.
# Résultats gardés dans %LOCALAPPDATA%\SmodePromptAssist\lora_triggers.json. "triggerWords" ajoute
# des mots à la main (en plus, ou à la place avec loraTriggers décoché).
#
# Si un champ n'est pas trouvé, cocher "dumpTree" : la liste des variables du modifier est
# affichée dans la console Smode.

modifier: Oil.createObject("WeakPointer(StreamDiffusionTextureModifier)")
idea: Oil.String("")
translation: Oil.CustomEnumeration(("Off", "Auto", "On"), 1)
style: Oil.CustomEnumeration(("Auto", "Photo", "Anime", "Illustration", "Painting", "3D Render", "None"), 0)
qualityTags: Oil.Boolean(True)
loraTriggers: Oil.Boolean(True)
triggerWords: Oil.String("")
writeNegative: Oil.Boolean(True)
extraNegative: Oil.String("")
modelOverride: Oil.String("")
dumpTree: Oil.Boolean(False)

import json
import math
import os
import re
import struct
import urllib.parse
import urllib.request

TOKEN_BUDGET = 75  # 77 positions CLIP moins les tokens de début et de fin
# Valeurs de l'énumération Smode "Stream Diffusion Config Type" (= ConfigType de ipc/protocol.py).
CFG_TYPES = {"1": "none", "2": "full", "3": "self", "4": "initialize"}

STYLES = ("Auto", "Photo", "Anime", "Illustration", "Painting", "3D Render", "None")
TRANSLATION_MODES = ("Off", "Auto", "On")
TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=fr&tl=en&dt=t&q="
CACHE_PATH = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                          "SmodePromptAssist", "translations.json")
CACHE_MAX = 5000
TRIGGER_CACHE_PATH = os.path.join(os.path.dirname(CACHE_PATH), "lora_triggers.json")
HF_API_URL = "https://huggingface.co/api/models/"
HF_README_URL = "https://huggingface.co/{}/raw/main/README.md"
LORA_EXT = (".safetensors", ".ckpt", ".pt", ".bin")
LORA_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+(::\S.*)?$")
# LoRA sans mot déclencheur : vitesse (agissent sur l'échantillonnage) et offset noise.
NO_TRIGGER_KW = ("hyper-sd", "hyper_sd", "lcm", "lightning", "turbo", "tcd", "dmd", "flash", "offset")

# Architecture : mêmes mots-clés SDXL que le package (engines/streamdiffusion/engine.py). Pour SD 2
# seulement des noms explicites : "2.1" seul classerait kohaku-v2.1 (un SD 1.5) en SD 2.1.
SDXL_KW = ("sdxl", "xl", "sd-xl", "sd_xl")
SD2_KW = ("sd-turbo", "sd_turbo", "stable-diffusion-2", "sd2", "sd-2", "sd_2")
DISTILLED_KW = ("turbo", "lightning", "hyper", "lcm", "tcd", "dmd", "flash", "1step", "2step", "4step", "8step")

FLAVOR_KW = (
    ("pony", ("pony",)),
    ("illustrious", ("illustrious", "noob")),
    ("anime", ("anime", "anything", "counterfeit", "kohaku", "meina", "animagine", "abyssorange",
               "waifu", "hassaku", "cetus", "mistoon", "toon", "manga", "nai")),
    ("realistic", ("realistic", "realvis", "juggernaut", "epicrealism", "photon", "cyberrealistic",
                   "absolutereality", "reality", "deliberate", "photo", "zavy", "rundiffusion")),
)

# Préfixe (en tête, avant l'idée), qualité (en fin, coupée en premier), négatif.
NEG_TAGS_ANIME = ("worst quality, low quality, lowres, bad anatomy, bad hands, missing fingers, "
                  "extra digits, jpeg artifacts, blurry, signature, watermark, text")
NEG_REALISTIC = ("cartoon, anime, painting, cgi, 3d render, blurry, lowres, worst quality, low quality, "
                 "jpeg artifacts, deformed, bad anatomy, extra fingers, watermark, text")
NEG_GENERIC = ("blurry, lowres, worst quality, low quality, jpeg artifacts, deformed, bad anatomy, "
               "watermark, text, signature")

RECIPES = {
    # (architecture, famille): (syntaxe, préfixe, qualité, négatif)
    ("sdxl", "pony"): ("tags", "score_9, score_8_up, score_7_up", "",
                       "score_4, score_5, score_6, " + NEG_TAGS_ANIME),
    ("sdxl", "illustrious"): ("tags", "masterpiece, best quality, very aesthetic, absurdres", "", NEG_TAGS_ANIME),
    ("sdxl", "anime"): ("tags", "masterpiece, best quality, very aesthetic, absurdres", "", NEG_TAGS_ANIME),
    ("sdxl", "realistic"): ("natural", "", "photograph, highly detailed, sharp focus, natural lighting, film grain",
                            NEG_REALISTIC),
    ("sdxl", "generic"): ("natural", "", "highly detailed, sharp focus, cinematic lighting", NEG_GENERIC),
    ("sd15", "anime"): ("tags", "masterpiece, best quality", "", NEG_TAGS_ANIME),
    ("sd15", "realistic"): ("natural", "", "RAW photo, highly detailed, sharp focus, soft lighting, film grain",
                            NEG_REALISTIC),
    ("sd15", "generic"): ("natural", "", "highly detailed, sharp focus, best quality", NEG_GENERIC),
    ("sd21", "generic"): ("natural", "", "highly detailed, sharp focus", NEG_GENERIC),
}

STYLE_PHRASES = {
    # style: (formulation naturelle, formulation en tags)
    "Photo": ("photograph, 35mm, natural light, shallow depth of field", "photorealistic, realistic"),
    "Anime": ("anime style illustration, cel shading", "anime coloring"),
    "Illustration": ("digital illustration, clean lineart, vibrant colors", "illustration, lineart"),
    "Painting": ("oil painting, visible brush strokes, rich colors", "traditional media, painting"),
    "3D Render": ("3d render, global illumination, octane render", "3d, blender (medium)"),
}

FRENCH_WORDS = {"une", "un", "le", "la", "les", "des", "dans", "avec", "sur", "sous", "et", "du", "de",
                "au", "aux", "est", "qui", "pour", "femme", "homme", "fille", "garçon", "ville", "nuit",
                "rouge", "noir", "noire", "blanc", "blanche", "bleu", "vert", "chat", "chien", "yeux",
                "cheveux", "robe", "ciel", "lune", "soleil", "mer", "visage", "sourire", "fond", "fleurs"}


def _norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


# ---------------------------------------------------------------- arbre Oil

def _var_names(elem, i):
    names = []
    for getter in (lambda: elem.getVariableName(i), lambda: elem.getVariable(i).getFriendlyName()):
        try:
            n = getter()
            if n:
                names.append(str(n))
        except Exception:
            pass
    return names


def _class_name(obj):
    try:
        return str(obj.getOilClassName())
    except Exception:
        return type(obj).__name__


def _children(obj):
    """(noms, variable) des variables d'un objet Oil, puis des éléments s'il s'agit d'un
    OwnedVector (Lora Scales : les éléments ne sont pas des variables, lus par len / [i])."""
    try:
        n = obj.getNumVariables()
    except Exception:
        n = 0
    for i in range(n):
        try:
            var = obj.getVariable(i)
        except Exception:
            continue
        yield _var_names(obj, i), var
    if "ownedvector" in _class_name(obj).lower():
        try:
            count = len(obj)
        except Exception:
            count = 0
        for i in range(count):
            try:
                yield [f"[{i}]"], obj[i]
            except Exception:
                continue


def walk(elem, max_depth=6, limit=4000):
    """(chemin, noms, variable) de chaque variable, en largeur, sans suivre les pointeurs."""
    queue = [(elem, (), 0)]
    seen = 0
    while queue:
        obj, path, depth = queue.pop(0)
        for i, (names, var) in enumerate(_children(obj)):
            seen += 1
            if seen > limit:
                return
            label = names[-1] if names else f"#{i}"
            yield path + (label,), names, var
            if depth + 1 < max_depth and "pointer" not in _class_name(var).lower():
                queue.append((var, path + (label,), depth + 1))


def find_var(elem, keys, exclude=()):
    """Première variable nommée comme l'une des clés, de préférence un champ texte."""
    keys = {_norm(k) for k in keys}
    exclude = {_norm(k) for k in exclude}
    first = (None, None)
    for path, names, var in walk(elem):
        normed = {_norm(x) for x in names}
        if normed & keys and not normed & exclude:
            if any(k in _class_name(var).lower() for k in ("string", "text")):
                return path, var
            if first[1] is None:
                first = (path, var)
    return first


def value_str(var):
    try:
        v = var.get()
    except Exception:
        return ""
    return "" if v is None else str(v)


def dump(elem):
    for path, names, var in walk(elem):
        v = value_str(var)
        v = (v[:60] + "...") if len(v) > 60 else v
        print(f"  {' > '.join(path)}  [{_class_name(var)}]  {v}")


# ---------------------------------------------------------------- prompt

def detect(model, loras):
    """Architecture et famille d'après le modèle (les tags de famille servent au modèle de base,
    une LoRA "manga" sur SDXL base n'en fait pas un modèle anime), distillation d'après les deux."""
    m = model.lower()
    arch = "sdxl" if any(k in m for k in SDXL_KW) else ("sd21" if any(k in m for k in SD2_KW) else "sd15")
    flavor = "generic"
    for name, kws in FLAVOR_KW:
        if any(k in m for k in kws):
            flavor = name
            break
    distilled = any(k in (m + " " + loras.lower()) for k in DISTILLED_KW)
    return arch, flavor, distilled


def recipe(arch, flavor):
    """Recette de (architecture, famille), sinon celle de la famille en SD 1.5, sinon générique."""
    if flavor in ("pony", "illustrious") and arch != "sdxl":
        flavor = "anime"
    for key in ((arch, flavor), ("sd15", flavor), (arch, "generic")):
        if key in RECIPES:
            return RECIPES[key]
    return RECIPES[("sd15", "generic")]


def clean(text):
    t = str(text)
    t = re.sub(r"<[^>]*>", " ", t)                       # <lora:...>, <embedding>
    t = re.sub(r"\bBREAK\b", ",", t)
    t = re.sub(r"\(([^():]+):\s*[\d.]+\)", r"\1", t)     # (mot:1.2) -> mot
    t = re.sub(r"[()\[\]{}]", " ", t)
    parts = [re.sub(r"\s+", " ", p).strip(" .") for p in re.split(r"[,\n]", t)]
    return [p for p in parts if p]


def split_tags(text):
    return [p.strip() for p in text.split(",") if p.strip()]


def est_tokens(text):
    """Approximation du BPE de CLIP : mot court = 1 token, mot long ~6 lettres par token,
    ponctuation = 1 token."""
    n = 0
    for w in re.findall(r"[A-Za-zÀ-ÿ]+|\d|[^\sA-Za-zÀ-ÿ\d]", text):
        n += max(1, math.ceil(len(w) / 6)) if w.isalpha() and len(w) > 7 else 1
    return n


def dedupe(parts):
    out, seen = [], set()
    for p in parts:
        k = _norm(p)
        if k and k not in seen:
            seen.add(k)
            out.append(p)
    return out


def build(idea_parts, arch, flavor, style_name, quality, triggers=()):
    syntax, prefix, qual, negative = recipe(arch, flavor)
    head = list(triggers) + split_tags(prefix) + idea_parts
    tail = []
    if style_name in STYLE_PHRASES:
        tail += split_tags(STYLE_PHRASES[style_name][1 if syntax == "tags" else 0])
    if quality:
        tail += split_tags(qual)
    head, tail = dedupe(head), [t for t in dedupe(tail) if _norm(t) not in {_norm(h) for h in head}]
    dropped = []
    while tail and est_tokens(", ".join(head + tail)) > TOKEN_BUDGET:
        dropped.insert(0, tail.pop())
    return ", ".join(head + tail), negative, dropped


def looks_french(text):
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", text.lower())
    return bool(re.search(r"[àâçéèêëîïôûùüœ]", text.lower())) or sum(w in FRENCH_WORDS for w in words) >= 2


# ---------------------------------------------------------------- traduction

def _load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        items = list(cache.items())[-CACHE_MAX:]
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(dict(items), f, ensure_ascii=False, indent=0)
    except Exception as e:
        print(f"[PromptAssist] Cache de traduction non enregistré : {e}")


def _google(lines):
    """Une requête pour tous les morceaux, un par ligne ; renvoie les lignes traduites."""
    url = TRANSLATE_URL + urllib.parse.quote("\n".join(lines))
    with urllib.request.urlopen(url, timeout=4) as r:
        data = json.loads(r.read().decode("utf-8"))
    text = "".join(seg[0] for seg in data[0] if seg and seg[0])
    return [t.strip() for t in text.split("\n")]


def keep_as_is(part):
    """Tags techniques à ne pas traduire : score_9, 1girl, 8k..."""
    return "_" in part or bool(re.match(r"\d", part))


def translate_parts(parts):
    """Traduit chaque morceau (cache d'abord). Renvoie (morceaux, nombre traduit en ligne, erreur)."""
    cache = _load_cache()
    todo = dedupe([p for p in parts if not keep_as_is(p) and p not in cache])
    err = None
    if todo:
        try:
            res = _google(todo)
            if len(res) != len(todo):
                raise ValueError(f"{len(res)} lignes reçues pour {len(todo)} envoyées")
            for src, dst in zip(todo, res):
                cache[src] = dst.strip(" .,") or src
            _save_cache(cache)
        except Exception as e:
            err = str(e)
    out = [p if keep_as_is(p) else cache.get(p, p) for p in parts]
    return out, (0 if err else len(todo)), err


def maybe_translate(text, mode, what):
    """clean() puis traduction selon le mode ; message si du français part tel quel dans CLIP."""
    parts = clean(text)
    french = looks_french(text)
    if not parts or mode == "Off" or (mode == "Auto" and not french):
        if french:
            print(f"[PromptAssist] Texte en français non traduit ({what}, translation = {mode}) : "
                  f"CLIP ne comprend que l'anglais.")
        return parts
    out, fetched, err = translate_parts(parts)
    if err:
        print(f"[PromptAssist] Traduction impossible ({what} : {err}) : texte gardé tel quel "
              f"(sauf les morceaux déjà en cache).")
    else:
        print(f"[PromptAssist] Traduction ({what}, " + (f"{fetched} morceau(x) en ligne" if fetched else "cache")
              + f") : {', '.join(out)}")
    return out


def enum_name(value, names):
    s = str(value)
    return names[int(s)] if s.isdigit() and int(s) < len(names) else s


# ---------------------------------------------------------------- mots déclencheurs des LoRA

def lora_ids(elem):
    """Identifiants des LoRA du modifier : valeurs texte sous une variable nommée "lora" qui
    ressemblent à un dépôt HF ou à un fichier."""
    ids = []
    for path, names, var in walk(elem):
        if not any("lora" in x.lower() for x in path + tuple(names)):
            continue
        v = value_str(var).strip().strip('"')
        if v and (LORA_ID_RE.match(v) or v.lower().endswith(LORA_EXT)) and v not in ids:
            ids.append(v)
    return ids


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _words(value):
    """Texte ou liste -> liste de mots déclencheurs (séparés par des virgules)."""
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value if v)
    return dedupe(split_tags(str(value or "")))


def safetensors_metadata(path):
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        if size > 50_000_000:
            return {}
        header = json.loads(f.read(size).decode("utf-8"))
    return header.get("__metadata__") or {}


def top_training_tags(meta, n=5):
    """Tags les plus fréquents des légendes d'entraînement (kohya "ss_tag_frequency")."""
    try:
        freq = json.loads(meta.get("ss_tag_frequency") or "{}")
    except ValueError:
        return []
    total = {}
    for tags in freq.values():
        for tag, count in (tags or {}).items():
            total[tag.strip()] = total.get(tag.strip(), 0) + int(count)
    return [t for t, _ in sorted(total.items(), key=lambda kv: -kv[1])[:n] if t]


def local_triggers(path):
    """(mots, source, suggestion) d'un fichier LoRA local."""
    meta = {}
    try:
        meta = safetensors_metadata(path) if path.lower().endswith(".safetensors") else {}
    except Exception as e:
        print(f"[PromptAssist] Métadonnées illisibles ({os.path.basename(path)} : {e}).")
    words = _words(meta.get("modelspec.trigger_phrase"))
    if words:
        return words, "métadonnées du fichier", []
    base = os.path.splitext(path)[0]
    words = _words(_load_json(base + ".civitai.info").get("trainedWords"))
    if words:
        return words, os.path.basename(base) + ".civitai.info", []
    words = _words(_load_json(base + ".json").get("activation text"))
    if words:
        return words, os.path.basename(base) + ".json", []
    return [], "fichier sans mot déclencheur", top_training_tags(meta)


def _get(url):
    with urllib.request.urlopen(url, timeout=4) as r:
        return r.read().decode("utf-8", "replace")


def hf_triggers(lora_id):
    """(mots, source) d'un dépôt HF : instance_prompt de la fiche, sinon "Trigger word:" du README."""
    repo = lora_id.split("::", 1)[0]
    card = json.loads(_get(HF_API_URL + repo)).get("cardData") or {}
    words = _words(card.get("instance_prompt"))
    if words:
        return words, "fiche HF (instance_prompt)"
    try:
        readme = _get(HF_README_URL.format(repo))
    except Exception:
        readme = ""
    m = re.search(r"trigger(?:\s*words?|\s*phrases?)?\s*(?:is|are)?\s*[:：]\s*[`*\"']*([^\n`*\"']{1,60})",
                  readme, re.I)
    if m:
        return _words(m.group(1)), "README HF (Trigger word)"
    return [], "fiche HF sans mot déclencheur"


def find_triggers(ids):
    """Mots déclencheurs de chaque LoRA, cache d'abord. Renvoie la liste de tous les mots."""
    cache = _load_json(TRIGGER_CACHE_PATH)
    changed = False
    out = []
    for lora_id in ids:
        name = os.path.basename(lora_id) if lora_id.lower().endswith(LORA_EXT) else lora_id
        if any(k in lora_id.lower() for k in NO_TRIGGER_KW):
            print(f"[PromptAssist] LoRA {name} : pas de mot déclencheur (LoRA de vitesse ou offset).")
            continue
        local = os.path.isfile(lora_id)
        key = f"{os.path.abspath(lora_id)}|{int(os.path.getmtime(lora_id))}" if local else lora_id
        hint = []
        if key in cache:
            words, source = cache[key]["words"], cache[key]["source"] + ", cache"
            hint = cache[key].get("hint", [])
        else:
            try:
                if local:
                    words, source, hint = local_triggers(lora_id)
                elif lora_id.lower().endswith(LORA_EXT):
                    print(f"[PromptAssist] LoRA {name} : fichier introuvable ({lora_id}).")
                    continue
                else:
                    words, source = hf_triggers(lora_id)
            except Exception as e:
                print(f"[PromptAssist] LoRA {name} : recherche impossible ({e}), réessayée au prochain lancement.")
                continue
            cache[key] = {"words": words, "source": source, "hint": hint}
            changed = True
        if hint:
            print(f"[PromptAssist] LoRA {name} : tags les plus fréquents à l'entraînement "
                  f"(suggestion, à mettre dans triggerWords) : {', '.join(hint)}")
        if words:
            print(f"[PromptAssist] LoRA {name} : {', '.join(words)} ({source})")
        else:
            print(f"[PromptAssist] LoRA {name} : aucun mot déclencheur trouvé ({source}), "
                  "la LoRA agit sans (sinon le mettre dans triggerWords).")
        out += words
    if changed:
        try:
            os.makedirs(os.path.dirname(TRIGGER_CACHE_PATH), exist_ok=True)
            with open(TRIGGER_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=0)
        except Exception as e:
            print(f"[PromptAssist] Cache des mots déclencheurs non enregistré : {e}")
    return dedupe(out)


# ---------------------------------------------------------------- exécution

def run():
    target = script.modifier.get()
    if target is None:
        print("[PromptAssist] Glisser un Stream Diffusion Modifier sur le paramètre 'modifier'.")
        return
    if script.dumpTree.get():
        print("[PromptAssist] Variables du modifier :")
        dump(target)

    p_path, p_var = find_var(target, ("Prompt",), exclude=("Negative Prompt",))
    n_path, n_var = find_var(target, ("Negative Prompt",))
    if p_var is None:
        print("[PromptAssist] Champ 'Prompt' introuvable : cocher dumpTree et regarder la liste.")
        return

    model = str(script.modelOverride.get()).strip()
    if not model:
        _, m_var = find_var(target, ("Model Path", "Model", "Model Name", "Model Id", "Model Repository"))
        model = value_str(m_var).strip() if m_var is not None else ""
    ids = lora_ids(target)
    loras = " ".join(ids)
    if not model:
        print("[PromptAssist] Modèle introuvable dans le modifier : le renseigner dans 'modelOverride' "
              "(ex. KBlueLeaf/kohaku-v2.1). Profil SD 1.5 générique utilisé.")

    raw = str(script.idea.get()).strip() or value_str(p_var)
    mode = enum_name(script.translation.get(), TRANSLATION_MODES)
    idea_parts = maybe_translate(raw, mode, "idée")

    triggers = []
    if script.loraTriggers.get():
        if ids:
            triggers = find_triggers(ids)
        else:
            # Liste vide = pas de LoRA, rien à dire ; des éléments illisibles = format à adapter.
            seen = [f"  {' > '.join(path)}  [{_class_name(var)}]  {value_str(var)[:80]}"
                    for path, _, var in walk(target)
                    if any("lora" in x.lower() for x in path) and len(path) > 1 and any(
                        x.startswith("[") or value_str(var) for x in path[-1:])]
            if seen:
                print("[PromptAssist] LoRA présentes mais nom illisible, variables lues :")
                print("\n".join(seen))
    triggers = dedupe(triggers + _words(script.triggerWords.get()))
    # Les mots déclencheurs déjà présents dans l'idée (prompt repris du modifier) passent en tête.
    trigger_keys = {_norm(t) for t in triggers}
    idea_parts = [p for p in idea_parts if _norm(p) not in trigger_keys]

    arch, flavor, distilled = detect(model, loras)
    style_name = enum_name(script.style.get(), STYLES)
    prompt, negative, dropped = build(idea_parts, arch, flavor, style_name,
                                      bool(script.qualityTags.get()), triggers)

    p_var.set(prompt)
    print(f"[PromptAssist] Modèle '{model or '?'}' -> {arch.upper()} {flavor}"
          + (" (distillé)" if distilled else ""))
    print(f"[PromptAssist] Prompt (~{est_tokens(prompt)}/{TOKEN_BUDGET} tokens) : {prompt}")
    if dropped:
        print(f"[PromptAssist] Retiré pour tenir dans CLIP : {', '.join(dropped)}")
    if est_tokens(prompt) > TOKEN_BUDGET:
        print("[PromptAssist] L'idée seule dépasse la limite de CLIP : la fin sera coupée, "
              "mettre l'essentiel au début.")

    if script.writeNegative.get() and n_var is not None:
        extra = maybe_translate(str(script.extraNegative.get()), mode, "négatif ajouté")
        neg = ", ".join(dedupe(split_tags(negative) + extra))
        n_var.set(neg)
        print(f"[PromptAssist] Negative Prompt : {neg}")
    elif n_var is None:
        print("[PromptAssist] Champ 'Negative Prompt' introuvable (dumpTree pour la liste).")

    # Le négatif n'est lu qu'en CFG initialize/full avec Guidance Scale > 1.
    _, c_var = find_var(target, ("Stream Diffusion Config Type", "Cfg Type", "CFG Type", "Config Type", "Cfg"))
    _, g_var = find_var(target, ("Guidance Scale", "Guidance"))
    cfg = value_str(c_var).lower() if c_var is not None else ""
    cfg = CFG_TYPES.get(cfg, cfg)
    try:
        guidance = float(value_str(g_var)) if g_var is not None else None
    except ValueError:
        guidance = None
    named = any(k in cfg for k in ("none", "full", "self", "initialize"))
    if named and not (("initialize" in cfg or "full" in cfg) and (guidance or 0) > 1.0):
        print(f"[PromptAssist] Le Negative Prompt est ignoré avec CFG '{cfg}' et Guidance "
              f"{guidance if guidance is not None else '?'} : il faut CFG initialize ou full et Guidance > 1.")
    elif not named:
        print(f"[PromptAssist] Rappel : le Negative Prompt n'est lu qu'en CFG initialize ou full avec "
              f"Guidance > 1 (CFG lu : '{cfg or '?'}', Guidance : {guidance if guidance is not None else '?'}).")
    if distilled and guidance is not None and guidance > 1.5:
        print(f"[PromptAssist] Modèle / LoRA distillé avec Guidance {guidance} : au-delà de ~1.5 l'image "
              "brûle (contraste, saturation).")


try:
    run()
except Exception as e:
    print(f"[PromptAssist] Échec : {e}")
