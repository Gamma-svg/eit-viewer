"""
generar_imagenes_eit.py
=======================
Recorre la carpeta de pruebas EIT y genera una imagen PNG por cada combinación:
  phantom / experimento / posición / dispositivo / patrón / algoritmo / frecuencia

Uso:
    python generar_imagenes_eit.py \
        --datos   /ruta/a/EIT_Pruebas_Hechas \
        --salida  /ruta/a/web_eit/imgs \
        --metodos svd jac greit bp \
        --comp    abs

Las imágenes se guardan como:
  imgs/<phantom>/<experimento>/<pos>/<dispositivo>/<patron>/<algoritmo>/<freq_hz>.png

También genera un fichero manifest.json con toda la estructura para la web.
"""

import os
import re
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
from pyeit.eit import svd, bp, jac, greit
from pyeit.mesh.shape import thorax
from pyeit.eit.fem import EITForward

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG POR DEFECTO
# ─────────────────────────────────────────────────────────────
DEFAULT_COMP    = "abs"          # "real" | "imag" | "abs"
DEFAULT_METODOS = ["svd", "jac", "greit", "bp"]
FASE_EN_GRADOS  = True
FIG_SIZE        = (3.2, 3.2)    # px de cada imagen individual
DPI             = 90

# ─────────────────────────────────────────────────────────────
# LECTURA .dat
# ─────────────────────────────────────────────────────────────

def leer_dat(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip().rstrip("\r")
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("freq") or line.lower().startswith("f+:"):
                continue
            parts = re.split(r"\t", line)
            parts = [p.strip().rstrip(",") for p in parts if p.strip()]
            if len(parts) < 7:
                continue
            try:
                rows.append({
                    "frequency": float(parts[0]),
                    "Ipos": int(float(parts[1])),
                    "Ineg": int(float(parts[2])),
                    "Spos": int(float(parts[3])),
                    "Sneg": int(float(parts[4])),
                    "real": float(parts[5]) if parts[5].lower() != "nan" else np.nan,
                    "imag": float(parts[6]) if parts[6].lower() != "nan" else np.nan,
                    "mag":  float(parts[7]) if len(parts) > 7 else np.nan,
                    "phas": float(parts[8]) if len(parts) > 8 else np.nan,
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def leer_params_header(path):
    dist_exc = step_meas = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                break
            if "dist_exc" in line:
                try:
                    dist_exc = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "step_meas" in line:
                try:
                    step_meas = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
    return dist_exc, step_meas


def detectar_patron_estadistico(df, electrodos):
    n_el   = len(electrodos)
    el_idx = {e: i for i, e in enumerate(sorted(electrodos))}
    pares_exc = set(tuple(sorted(p)) for p in df[["Ipos", "Ineg"]].values)
    dist_exc_vals = [
        min(abs(el_idx[Ip] - el_idx[In]), n_el - abs(el_idx[Ip] - el_idx[In]))
        for Ip, In in pares_exc if Ip in el_idx and In in el_idx
    ]
    pares_meas = set(tuple(sorted(p)) for p in df[["Spos", "Sneg"]].values)
    dist_meas_vals = [
        min(abs(el_idx[Sp] - el_idx[Sn]), n_el - abs(el_idx[Sp] - el_idx[Sn]))
        for Sp, Sn in pares_meas if Sp in el_idx and Sn in el_idx
    ]
    if not dist_exc_vals or not dist_meas_vals:
        return 1, 1
    opciones = {1: 1, n_el // 2: n_el // 2}
    best_exc  = min(opciones, key=lambda k: abs(k - np.mean(dist_exc_vals)))
    best_meas = min(opciones, key=lambda k: abs(k - np.mean(dist_meas_vals)))
    return best_exc, best_meas


def obtener_patron(path, df, electrodos):
    dist_exc, step_meas = leer_params_header(path)
    if dist_exc is None or step_meas is None:
        dist_exc, step_meas = detectar_patron_estadistico(df, electrodos)
    return dist_exc, step_meas


def df_a_vector(df, componente, orden_pares):
    idx = {}
    for _, row in df.iterrows():
        key = (int(row["Ipos"]), int(row["Ineg"]), int(row["Spos"]), int(row["Sneg"]))
        r, i = row["real"], row["imag"]
        m, p = row.get("mag", np.nan), row.get("phas", np.nan)
        if (np.isnan(r) or np.isnan(i)) and not (np.isnan(m) or np.isnan(p)):
            p_rad = np.radians(p) if FASE_EN_GRADOS else p
            r = m * np.cos(p_rad)
            i = m * np.sin(p_rad)
        if componente == "real":   idx[key] = r
        elif componente == "imag": idx[key] = i
        elif componente == "abs":  idx[key] = np.sqrt(r**2 + i**2) if not (np.isnan(r) or np.isnan(i)) else m

    result = []
    for (Ip, In, Sp, Sn) in orden_pares:
        if   (Ip, In, Sp, Sn) in idx: result.append( idx[(Ip, In, Sp, Sn)])
        elif (Ip, In, Sn, Sp) in idx: result.append(-idx[(Ip, In, Sn, Sp)])
        elif (In, Ip, Sp, Sn) in idx: result.append(-idx[(In, Ip, Sp, Sn)])
        elif (In, Ip, Sn, Sp) in idx: result.append( idx[(In, Ip, Sn, Sp)])
        else:                          result.append(np.nan)
    v = np.array(result, dtype=float)
    if np.any(np.isnan(v)):
        v[np.isnan(v)] = np.nanmedian(v)
    return v


# ─────────────────────────────────────────────────────────────
# RECONSTRUCCIÓN → PNG
# ─────────────────────────────────────────────────────────────

def reconstruir_freq(metodo, proto, mesh_obj, v0_freq, v1_freq):
    metodo = metodo.lower()
    if metodo == "svd":
        eit = svd.SVD(mesh_obj, proto)
        eit.setup(n=50, method="svd", perm=1, jac_normalized=True)
    elif metodo == "bp":
        eit = bp.BP(mesh_obj, proto)
        eit.setup(weight="none")
    elif metodo == "jac":
        eit = jac.JAC(mesh_obj, proto)
        eit.setup(p=0.5, lamb=0.01, method="kotre")
    elif metodo == "greit":
        eit = greit.GREIT(mesh_obj, proto)
        eit.setup(method="dist", w=None, p=0.2, lamb=1e-2, n=64, s=20.0, ratio=0.1)
    else:
        raise ValueError(f"Método desconocido: {metodo}")

    normalize = metodo in ["svd", "jac", "bp"]
    ds = eit.solve(v1_freq, v0_freq, normalize=normalize)
    return eit, ds


def guardar_imagen(eit, ds, mesh_obj, metodo, out_path):
    pts, tri = mesh_obj.node, mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    metodo = metodo.lower()
    try:
        if metodo in ["svd", "jac"]:
            from pyeit.visual.plot import sim2pts
            ds_n = sim2pts(pts, tri, np.real(ds))
            ax.tripcolor(x, y, tri, ds_n, shading="flat", cmap="RdBu_r")
        elif metodo == "bp":
            ax.tripcolor(x, y, tri, ds, shading="flat", cmap="RdBu_r")
        elif metodo == "greit":
            xg, yg, ds_grid = eit.mask_value(ds, mask_value=np.nan)
            ax.pcolormesh(xg, yg, ds_grid, shading="auto", cmap="RdBu_r")
    except Exception as e:
        # fallback
        ax.tripcolor(x, y, tri, np.real(ds), shading="flat", cmap="RdBu_r")

    # Electrodos
    for j, e in enumerate(mesh_obj.el_pos):
        ax.annotate(str(j + 1), xy=(x[e], y[e]),
                    color="white", fontsize=7, ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="#222", ec="white",
                              linewidth=0.6, alpha=0.85))

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# NOMBRADO LEGIBLE
# ─────────────────────────────────────────────────────────────

def nombre_patron(dist_exc, step_meas, n_el):
    def _n(d):
        if d == 1:         return "Adj"
        if d == n_el // 2: return "Op"
        return f"Skip{d-1}"
    return f"{_n(dist_exc)}-{_n(step_meas)}"


def fmt_freq(hz):
    if   hz >= 1e6: return f"{hz/1e6:.2f}MHz"
    elif hz >= 1e3: return f"{hz/1e3:.1f}kHz"
    else:           return f"{hz:.0f}Hz"


def fmt_freq_label(hz):
    if   hz >= 1e6: return f"{hz/1e6:.2f} MHz"
    elif hz >= 1e3: return f"{hz/1e3:.1f} kHz"
    else:           return f"{hz:.0f} Hz"


# ─────────────────────────────────────────────────────────────
# DESCUBRIMIENTO DE PARES (baseline, data)
# ─────────────────────────────────────────────────────────────

PATRON_MAP = {
    "adj-adj":    ("Adjacent", "Adjacent Adjacent"),
    "op-adj":     ("Opposite", "Opposite Adjacent"),
    "skip2-adj":  ("Skip2",    "Skip 2"),
    "skip-2-adj": ("Skip2",    "Skip 2"),
}

def patron_desde_filename(fname):
    fname_lower = fname.lower()
    if "adj-adj" in fname_lower or "adjacent adjacent" in fname_lower:
        return "adj-adj"
    if "op-adj" in fname_lower or "opposite adjacent" in fname_lower:
        return "op-adj"
    if "skip2-adj" in fname_lower or "skip 2" in fname_lower:
        return "skip2-adj"
    return None


def encontrar_baseline(dat_path, base_dir):
    """
    Dada la ruta de un .dat de datos, encuentra el .dat de baseline
    del mismo dispositivo y patrón de excitación.
    """
    parts = dat_path.parts
    # Identificar phantom (PLA, Pollo, Vegetales)
    phantom_idx = None
    for i, p in enumerate(parts):
        if p in ("PLA", "Pollo", "Vegetales"):
            phantom_idx = i
            break
    if phantom_idx is None:
        return None

    phantom  = parts[phantom_idx]
    device   = parts[phantom_idx + 3]   # ScioSpec_Xel, KIT_16el, mACQ_8el
    patron_f = patron_desde_filename(dat_path.name)

    # Buscar en baseline del mismo phantom
    baseline_root = base_dir / phantom / "Baseline"
    if not baseline_root.exists():
        # PLA tiene Baseline_11 y Baseline_55
        for sub in base_dir.glob(f"{phantom}/Baseline*"):
            baseline_root = sub
            break

    candidates = list(baseline_root.rglob(f"*{device}*/*.dat")) + \
                 list(baseline_root.rglob(f"*/{device}/*.dat"))

    for c in candidates:
        c_patron = patron_desde_filename(c.name)
        if c_patron == patron_f:
            return c

    # fallback: cualquier .dat de ese dispositivo en baseline
    if candidates:
        return candidates[0]
    return None


# ─────────────────────────────────────────────────────────────
# PROCESADO DE UN PAR (baseline, data) → imágenes
# ─────────────────────────────────────────────────────────────

def procesar_par(path_base, path_data, salida_dir, metodos, componente):
    """
    Genera imágenes para todos los métodos y frecuencias de un par (baseline, data).
    Devuelve lista de dicts con metadatos para el manifest.
    """
    resultados = []

    try:
        df_base = leer_dat(str(path_base))
        df_data = leer_dat(str(path_data))
    except Exception as e:
        print(f"  ✗ Error leyendo: {e}")
        return []

    if df_base.empty or df_data.empty:
        print(f"  ✗ DataFrame vacío: {path_data.name}")
        return []

    electrodos = sorted(set(df_base["Ipos"].unique()) | set(df_base["Ineg"].unique()))
    n_el = len(electrodos)
    if n_el < 4:
        print(f"  ✗ Muy pocos electrodos: {n_el}")
        return []

    dist_exc, step_meas = obtener_patron(str(path_base), df_base, electrodos)
    patron_str = nombre_patron(dist_exc, step_meas, n_el)

    try:
        proto    = protocol.create(n_el, dist_exc=dist_exc, step_meas=step_meas, parser_meas="std")
        mesh_obj = mesh.create(n_el, h0=0.05)
    except Exception as e:
        print(f"  ✗ Error construyendo protocolo/malla: {e}")
        return []

    el = np.array(electrodos)
    orden_pares = []
    for ei in range(proto.ex_mat.shape[0]):
        Ipos = int(el[proto.ex_mat[ei, 0]])
        Ineg = int(el[proto.ex_mat[ei, 1]])
        for mi in range(proto.meas_mat.shape[1]):
            Spos = int(el[proto.meas_mat[ei, mi, 0]])
            Sneg = int(el[proto.meas_mat[ei, mi, 1]])
            orden_pares.append((Ipos, Ineg, Spos, Sneg))

    frecuencias = sorted(df_base["frequency"].unique())
    if not frecuencias:
        return []

    # Vectores por frecuencia
    v0_list, v1_list = [], []
    for freq in frecuencias:
        sub_b = df_base[df_base["frequency"].round(-1) == round(freq, -1)]
        sub_d = df_data[df_data["frequency"].round(-1) == round(freq, -1)]
        if sub_b.empty:
            sub_b = df_base[df_base["frequency"] == df_base["frequency"].unique()[
                np.argmin(np.abs(df_base["frequency"].unique() - freq))]]
        if sub_d.empty:
            sub_d = df_data
        v0_list.append(df_a_vector(sub_b, componente, orden_pares))
        v1_list.append(df_a_vector(sub_d, componente, orden_pares))

    for metodo in metodos:
        for i, freq in enumerate(frecuencias):
            freq_str = fmt_freq(freq)
            img_path = salida_dir / f"{patron_str}" / f"{metodo}" / f"{freq_str}.png"

            if img_path.exists():
                pass  # ya existe, no regenerar
            else:
                try:
                    eit_obj, ds = reconstruir_freq(metodo, proto, mesh_obj,
                                                   v0_list[i], v1_list[i])
                    guardar_imagen(eit_obj, ds, mesh_obj, metodo, img_path)
                except Exception as e:
                    print(f"    ✗ {metodo} @ {fmt_freq_label(freq)}: {e}")
                    continue

            resultados.append({
                "patron":    patron_str,
                "metodo":    metodo.upper(),
                "freq_hz":   freq,
                "freq_label": fmt_freq_label(freq),
                "img": "imgs/" + "/".join(img_path.parts[-7:]),
            })

    return resultados


# ─────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL MANIFEST
# ─────────────────────────────────────────────────────────────

def construir_manifest(datos_dir, salida_dir, metodos, componente):
    datos_dir = Path(datos_dir)
    salida_dir = Path(salida_dir)

    manifest = {}   # manifest[phantom][experimento][pos][dispositivo] = [resultados]

    # Encontrar todos los .dat que NO son baseline
    all_dats = sorted(datos_dir.rglob("*.dat"))
    data_dats = [p for p in all_dats if "baseline" not in str(p).lower()]

    total = len(data_dats)
    print(f"\n{'='*60}")
    print(f"  Encontrados {total} archivos .dat de datos")
    print(f"  Algoritmos : {', '.join(m.upper() for m in metodos)}")
    print(f"  Componente : {componente}")
    print(f"{'='*60}\n")

    for idx, dat_path in enumerate(data_dats):
        parts = dat_path.parts
        # Localizar phantom_idx
        phantom_idx = None
        for i, p in enumerate(parts):
            if p in ("PLA", "Pollo", "Vegetales"):
                phantom_idx = i
                break
        if phantom_idx is None:
            continue

        phantom    = parts[phantom_idx]
        experimento = parts[phantom_idx + 1]
        pos         = parts[phantom_idx + 2]
        device      = parts[phantom_idx + 3]

        # Buscar baseline
        path_base = encontrar_baseline(dat_path, datos_dir)
        if path_base is None:
            print(f"  [{idx+1}/{total}] ✗ Sin baseline: {dat_path.name}")
            continue

        patron_f = patron_desde_filename(dat_path.name)
        if patron_f is None:
            print(f"  [{idx+1}/{total}] ✗ Patrón no reconocido: {dat_path.name}")
            continue

        print(f"  [{idx+1}/{total}] {phantom}/{experimento}/{pos}/{device}/{patron_f}")

        # Directorio de salida para este experimento
        exp_salida = salida_dir / phantom / experimento / pos / device
        exp_salida.mkdir(parents=True, exist_ok=True)

        resultados = procesar_par(path_base, dat_path, exp_salida,
                                  metodos, componente)

        if not resultados:
            continue

        # Añadir foto del phantom si existe
        foto_orig = dat_path.parent.parent / "recorte.png"
        foto_debug = dat_path.parent.parent / "debug.jpg"
        foto = None
        if foto_orig.exists():
            dst = salida_dir / phantom / experimento / pos / "recorte.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                import shutil
                shutil.copy2(str(foto_orig), str(dst))
            foto = "imgs/" + "/".join(dst.parts[-4:])
        elif foto_debug.exists():
            dst = salida_dir / phantom / experimento / pos / "debug.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                import shutil
                shutil.copy2(str(foto_debug), str(dst))
            foto = "imgs/" + "/".join(dst.parts[-4:])

        # Registrar en manifest
        m = manifest
        for key in [phantom, experimento, pos, device]:
            m = m.setdefault(key, {})
        m.setdefault("resultados", []).extend(resultados)
        if foto:
            m["foto"] = foto

    return manifest


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genera imágenes EIT para la web de defensa")
    parser.add_argument("--datos",   required=True, help="Ruta a la carpeta EIT_Pruebas_Hechas")
    parser.add_argument("--salida",  required=True, help="Carpeta de salida (dentro del proyecto web)")
    parser.add_argument("--metodos", nargs="+", default=DEFAULT_METODOS,
                        choices=["svd", "bp", "jac", "greit"],
                        help="Algoritmos a usar")
    parser.add_argument("--comp",    default=DEFAULT_COMP,
                        choices=["real", "imag", "abs"],
                        help="Componente de la señal")
    args = parser.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    manifest = construir_manifest(args.datos, salida, args.metodos, args.comp)

    manifest_path = salida.parent / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Contar imágenes generadas
    n_imgs = sum(1 for _ in salida.rglob("*.png")) + \
             sum(1 for _ in salida.rglob("*.jpg"))
    print(f"\n{'='*60}")
    print(f"  ✓ Completado")
    print(f"  Imágenes generadas: {n_imgs}")
    print(f"  Manifest guardado : {manifest_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
