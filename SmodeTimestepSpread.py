# Script Python pour Smode (outil "Python Script") : calcule et applique automatiquement
# les "Timestep Indices" d'un Stream Diffusion Modifier en fonction du nombre de pas.
#
# Utilisation :
#   1. Ajouter un outil Python Script (Tools > Python Script), coller ce script.
#   2. Glisser le Stream Diffusion Modifier sur le paramètre "modifier" du script.
#   3. Régler "numSteps" (nombre de pas) et "strength" (1.0 = repart du bruit pur,
#      0.5 = garde la moitié de l'image d'entrée...).
#   4. Launch Mode "At Parameter Change" (ou Execute à la main).
#   Chemin écrit : Stream Diffusion Instance > Inference Step Config > Timestep Indices.
#
# Étalement : le "trailing spacing" de diffusers, celui avec lequel les LoRA Hyper-SD
# (1/2/4/8 pas) et LCM ont été entraînées. Sur la grille TCD à 50 pas (t = 999 - 20*i)
# le pas k vaut i_k = s + (50 - s) * k / N, avec s l'index de départ donné par strength.
#   N=1 -> [1]            (t=979)
#   N=2 -> [1, 25]        (t=979, 499)    = Hyper-SDXL-2steps
#   N=4 -> [1, 13, 25, 38]                = Hyper-SDXL-4steps
#   N=5 -> [1, 10, 20, 30, 40]
#   N=8 -> [1, 6, 13, 19, 25, 31, 38, 44] = Hyper-SDXL-8steps
# Les index sont bornés à [1, 49] (limites de l'interface Smode) et strictement croissants.

modifier: Oil.createObject("WeakPointer(StreamDiffusionTextureModifier)")
numSteps: Oil.PositiveInteger(2)
strength: Oil.Percentage(1.0)
minIndex: Oil.PositiveInteger(1)
maxIndex: Oil.PositiveInteger(49)
verbose: Oil.Boolean(False)

GRID = 50  # num_inference_steps du scheduler TCD/LCM dans le package


def compute_indices(n, strength, lo, hi):
    n = max(1, int(n))
    strength = min(1.0, max(0.0, float(strength)))
    start = (1.0 - strength) * (hi + 1)          # index de départ (t le plus élevé)
    out = []
    for k in range(n):
        i = int(start + (GRID - start) * k / n + 0.5)   # arrondi demi-supérieur
        i = min(hi, max(lo, i))
        if out and i <= out[-1]:                 # garde des index strictement croissants
            i = out[-1] + 1
        if i > hi:
            break
        out.append(i)
    return out


def _norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def child(elem, friendly):
    """Variable enfant d'un objet Oil par son nom affiché ("Timestep Indices")."""
    key = _norm(friendly)
    try:
        v = getattr(elem, friendly[0].lower() + friendly.title().replace(" ", "")[1:])
        if v is not None:
            return v
    except Exception:
        pass
    n = elem.getNumVariables()
    for i in range(n):
        names = []
        for getter in (lambda: elem.getVariableName(i), lambda: elem.getVariable(i).getFriendlyName()):
            try:
                names.append(getter())
            except Exception:
                pass
        if any(_norm(x) == key for x in names):
            return elem.getVariable(i)
    raise RuntimeError(f"variable '{friendly}' introuvable dans {elem.getOilClassName()}")


def set_int(obj, value):
    """Écrit un entier dans un TimeStepIndexInteger (valeur directe ou sous-variable Value)."""
    try:
        obj.set(int(value))
        return
    except Exception:
        pass
    child(obj, "Value").set(int(value))


def write_vector(vec, values):
    """Redimensionne l'OwnedVector(TimeStepIndexInteger) et écrit chaque index."""
    n = len(vec)
    while n > len(values):                        # retire les éléments en trop (par la fin)
        removed = False
        for op in (lambda: vec.remove(n - 1), lambda: vec.erase(n - 1), lambda: vec.__delitem__(n - 1)):
            try:
                op(); removed = True; break
            except Exception:
                pass
        if not removed:
            raise RuntimeError("impossible de retirer un élément du vecteur (remove/erase/del)")
        n = len(vec)
    while n < len(values):                        # ajoute les éléments manquants
        vec.append(Oil.createObject("TimeStepIndexInteger"))
        n = len(vec)
    for i, v in enumerate(values):
        set_int(vec[i], v)


target = script.modifier.get()
if target is None:
    print("[TimestepSpread] Glisser un Stream Diffusion Modifier sur le paramètre 'modifier'.")
else:
    lo = max(1, int(script.minIndex.get()))
    hi = min(49, max(lo, int(script.maxIndex.get())))
    indices = compute_indices(script.numSteps.get(), script.strength.get(), lo, hi)
    vec = None
    try:
        instance = child(target, "Stream Diffusion Instance")
        vec = child(child(instance, "Inference Step Config"), "Timestep Indices")
        write_vector(vec, indices)
        t_values = [999 - 20 * i for i in indices]
        print(f"[TimestepSpread] {len(indices)} pas -> indices {indices} (t = {t_values})")
        if script.verbose.get():
            Oil.docMe(vec)
            if len(vec) > 0:
                Oil.docMe(vec[0])
    except Exception as e:
        print(f"[TimestepSpread] Échec : {e}")
        if vec is not None:
            Oil.docMe(vec)
            if len(vec) > 0:
                Oil.docMe(vec[0])
