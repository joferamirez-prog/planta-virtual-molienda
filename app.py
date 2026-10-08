"""
Planta Virtual de Molienda y Tamizaje  ·  versión avanzada
==========================================================
Circuito: recepción -> alimentador de placas -> chancadora -> banda -> tolva
-> alimentador + imán -> molino de bolas (circuito cerrado) -> elevador
-> zaranda -> silo de producto, con colector de polvo.

Producto: biblioteca de materiales, material personalizado o mezcla.
Modelo didáctico simplificado (Bond + Rosin-Rammler + balance de masa).
Las propiedades de los materiales son valores típicos de referencia:
ajústalos con tus propios ensayos.
"""
import html
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------------------------------------------
# 1. BIBLIOTECA DE MATERIALES
# ----------------------------------------------------------------------------
# wi: índice de trabajo de Bond (kWh/t) · rho: densidad real (t/m³)
# bulk: densidad aparente (t/m³) · mohs: dureza · ai: índice de abrasión de Bond
# moist: humedad de alimentación (%) · n: dispersión Rosin-Rammler
MATERIALS = {
    "Caliza": dict(wi=11.6, rho=2.70, bulk=1.50, mohs=3.0, ai=0.020, moist=2.0, n=0.95,
                   color="#e3dcc8", uso="Cemento, cal, agregados, carga mineral", hazard=""),
    "Dolomita": dict(wi=11.3, rho=2.85, bulk=1.55, mohs=3.7, ai=0.030, moist=2.0, n=0.95,
                     color="#cdd0c6", uso="Siderurgia, vidrio, fertilizantes", hazard=""),
    "Clínker de cemento": dict(wi=13.5, rho=3.15, bulk=1.35, mohs=5.5, ai=0.070, moist=0.5, n=0.85,
                               color="#7a6c5d", uso="Molienda de cemento", hazard=""),
    "Yeso": dict(wi=6.7, rho=2.30, bulk=1.20, mohs=2.0, ai=0.010, moist=4.0, n=1.00,
                 color="#f1efe6", uso="Cemento, paneles, agricultura", hazard=""),
    "Carbón mineral": dict(wi=11.4, rho=1.40, bulk=0.85, mohs=2.5, ai=0.010, moist=8.0, n=1.00,
                           color="#2b2b30", uso="Combustible pulverizado",
                           hazard="Polvo de carbón explosivo: requiere inertización, venteo de explosión y control de temperatura."),
    "Arena de sílice (cuarzo)": dict(wi=12.8, rho=2.65, bulk=1.60, mohs=7.0, ai=0.750, moist=1.0, n=1.10,
                                     color="#e8c98f", uso="Vidrio, fundición, filtración",
                                     hazard="Sílice cristalina respirable: control estricto de polvo y protección respiratoria."),
    "Feldespato": dict(wi=11.7, rho=2.60, bulk=1.40, mohs=6.0, ai=0.200, moist=1.0, n=1.00,
                       color="#e6b6a0", uso="Cerámica y vidrio", hazard=""),
    "Roca fosfórica": dict(wi=9.9, rho=2.70, bulk=1.50, mohs=5.0, ai=0.040, moist=3.0, n=0.95,
                           color="#a9a480", uso="Fertilizantes", hazard=""),
    "Bauxita": dict(wi=9.0, rho=2.50, bulk=1.30, mohs=3.0, ai=0.050, moist=12.0, n=1.00,
                    color="#a65f3f", uso="Alúmina y aluminio", hazard=""),
    "Mena de cobre": dict(wi=13.1, rho=2.90, bulk=1.70, mohs=5.5, ai=0.150, moist=3.0, n=0.90,
                          color="#a8683a", uso="Concentración de cobre", hazard=""),
    "Mena de oro": dict(wi=14.8, rho=2.75, bulk=1.65, mohs=6.5, ai=0.400, moist=3.0, n=0.85,
                        color="#d4a72c", uso="Lixiviación / flotación", hazard=""),
    "Hematita (mena de hierro)": dict(wi=12.7, rho=3.60, bulk=2.10, mohs=5.5, ai=0.200, moist=3.0, n=0.90,
                                      color="#8a3b2c", uso="Siderurgia, pellets", hazard=""),
    "Escoria de alto horno": dict(wi=15.8, rho=2.90, bulk=1.25, mohs=6.0, ai=0.350, moist=10.0, n=0.85,
                                  color="#7d8a8f", uso="Cemento con adición de escoria", hazard=""),
}
CUSTOM_LABEL = "★ Personalizado"

OP_DEFAULTS = dict(
    running=True,
    feed_tph=120.0,
    f80_mm=12.0,
    wi_adj=0,
    D=3.0,
    L=4.5,
    nc=74,
    J=30,
    aperture_mm=0.60,
    eff0=88,
    area=25.0,
    target_mm=0.50,
    hopper_level=70,
    silo_level=40,
    price_kwh=0.10,
    price_steel=1.40,
)
MAT_DEFAULTS = dict(
    mat_mode="Biblioteca",
    mat_lib="Caliza",
    cust_name="Mi material",
    cust_wi=13.0, cust_rho=2.70, cust_bulk=1.50, cust_mohs=5.0,
    cust_ai=0.150, cust_moist=3.0, cust_n=0.95, cust_color="#c9b38a",
    mix_sel=["Caliza", "Arena de sílice (cuarzo)"],
    view="Vista completa",
)

PRESETS = {
    "Condición base": {},
    "Alimentación alta (160 t/h)": dict(feed_tph=160.0),
    "Mineral más duro (Wi +30 %)": dict(wi_adj=30),
    "Malla fina (0.40 mm)": dict(aperture_mm=0.40, target_mm=0.30),
    "Zaranda sucia (eficiencia 60 %)": dict(eff0=60),
    "Molino con poca carga de bolas": dict(J=20),
}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(c):
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, round(v)))) for v in c)


def blend_materials(comps):
    """comps: [(nombre, fracción_másica, props)] -> propiedades de la mezcla."""
    tot = sum(w for _, w, _ in comps) or 1.0
    comps = [(n, w / tot, m) for n, w, m in comps]

    def wsum(k):
        return sum(w * m[k] for _, w, m in comps)

    rho = 1.0 / sum(w / m["rho"] for _, w, m in comps)      # mezcla en volumen
    bulk = 1.0 / sum(w / m["bulk"] for _, w, m in comps)
    rgb = [sum(w * hex_to_rgb(m["color"])[i] for _, w, m in comps) for i in range(3)]
    hz = [m["hazard"] for _, _, m in comps if m.get("hazard")]
    if len(comps) == 1:
        name = comps[0][0]
    else:
        name = " + ".join(f"{round(w * 100)} % {n}" for n, w, _ in comps)
    return dict(name=name, wi=wsum("wi"), rho=rho, bulk=bulk, mohs=wsum("mohs"), ai=wsum("ai"),
                moist=wsum("moist"), n=wsum("n"), color=rgb_to_hex(rgb),
                hazard=" ".join(dict.fromkeys(hz)), uso="Alimentación", components=comps)


def custom_material(ss):
    return dict(wi=float(ss["cust_wi"]), rho=float(ss["cust_rho"]), bulk=float(ss["cust_bulk"]),
                mohs=float(ss["cust_mohs"]), ai=float(ss["cust_ai"]), moist=float(ss["cust_moist"]),
                n=float(ss["cust_n"]), color=ss["cust_color"], uso="Material definido por el usuario",
                hazard="")


def material_pool(ss):
    pool = dict(MATERIALS)
    pool[CUSTOM_LABEL] = custom_material(ss)
    return pool


def resolve_material(ss):
    mode = ss["mat_mode"]
    pool = material_pool(ss)
    if mode == "Biblioteca":
        name = ss["mat_lib"] if ss["mat_lib"] in MATERIALS else "Caliza"
        comps = [(name, 1.0, pool[name])]
    elif mode == "Personalizado":
        name = (ss["cust_name"] or "Personalizado").strip()
        comps = [(name, 1.0, pool[CUSTOM_LABEL])]
    else:
        names = [n for n in ss["mix_sel"] if n in pool] or ["Caliza"]
        ws = [float(ss.get(f"mix_{n}", 50)) for n in names]
        if sum(ws) <= 0:
            ws = [1.0] * len(names)
        comps = [(n if n != CUSTOM_LABEL else (ss["cust_name"] or "Personalizado"), w, pool[n])
                 for n, w in zip(names, ws)]
    return blend_materials(comps)


def build_params(ss):
    p = {k: ss[k] for k in OP_DEFAULTS}
    mat = resolve_material(ss)
    p.update(wi=mat["wi"] * (1 + ss["wi_adj"] / 100.0), n_rr=mat["n"], rho=mat["rho"],
             bulk=mat["bulk"], ai=mat["ai"], moist=mat["moist"], mohs=mat["mohs"])
    return p, mat


# ----------------------------------------------------------------------------
# 2. MODELO DE PROCESO
# ----------------------------------------------------------------------------
def rr_cdf(x, x63, n):
    """Distribución Rosin-Rammler: fracción pasante."""
    return 1.0 - np.exp(-(np.asarray(x, dtype=float) / x63) ** n)


def simulate(p):
    """Balance de masa en estado estable del circuito cerrado molino-zaranda."""
    F = max(p["feed_tph"], 1e-6)
    F80 = p["f80_mm"] * 1000.0  # µm
    Wi = p["wi"]
    D, L = p["D"], p["L"]
    phi = p["nc"] / 100.0
    J = p["J"] / 100.0
    a = p["aperture_mm"] * 1000.0  # µm
    n = p["n_rr"]

    nc_rpm = 42.3 / math.sqrt(D)
    rpm = phi * nc_rpm
    kw = 45.0 * D**2.5 * L * phi * J * (1 - 1.03 * J)  # potencia absorbida (empírica)
    volume = math.pi / 4 * D**2 * L
    s_max_mill = 14.0 * volume * (p["rho"] / 2.7)
    screen_cap = 25.0 * math.sqrt(p["aperture_mm"]) * p["area"] * (p["bulk"] / 1.6)
    eff_base = (p["eff0"] / 100.0) * max(0.5, 1.0 - 0.035 * max(0.0, p["moist"] - 3.0))

    S = 2.0 * F
    x_r = 2.0 * a
    eta, f, x80, x63, load, f80_eff = eff_base, 0.7, 800.0, 500.0, 0.5, F80
    for _ in range(300):
        E = kw / S
        w = min(F / S, 1.0)
        inv_feed = w / math.sqrt(F80) + (1 - w) / math.sqrt(max(x_r, 1.0))
        f80_eff = 1.0 / inv_feed**2
        inv_p = 1.0 / math.sqrt(f80_eff) + E / (10.0 * Wi)
        x80 = 1.0 / inv_p**2
        x63 = x80 / (math.log(5.0) ** (1.0 / n))
        f = float(rr_cdf(a, x63, n))
        load = S / screen_cap
        eta = max(eff_base / (1.0 + 2.0 * max(0.0, load - 0.9)), 0.05)
        S_new = min(max(F / max(f * eta, 1e-3), F), 15.0 * F)
        S = 0.6 * S + 0.4 * S_new
        x_r = x63 * (-math.log(max(0.2 * (1 - f), 1e-9))) ** (1.0 / n)

    R = S - F
    p80_prod = x63 * (-math.log(1 - 0.8 * f)) ** (1.0 / n)

    # desgaste (ecuaciones de Bond, molienda húmeda: referencia) y costos
    ai = p["ai"]
    ball_kgph = 0.159 * max(ai - 0.015, 1e-4) ** 0.33 * kw
    liner_kgph = 0.0159 * max(ai - 0.0206, 1e-4) ** 0.26 * kw
    cost_energy = kw * p["price_kwh"] / F
    cost_steel = (ball_kgph + liner_kgph) * p["price_steel"] / F
    # colector de polvo y servicios
    q_air = 1500.0 + 28.0 * S
    dp = 750.0 + 3.2 * S * (1 + p["moist"] / 12.0)
    return dict(
        F=F, S=S, R=R, cl=R / F * 100.0, kw=kw, rpm=rpm, nc_rpm=nc_rpm, E_fresh=kw / F, E_total=kw / S,
        x80=x80, x63=x63, f=f, eta=eta, eff_base=eff_base, load=load, util_mill=S / s_max_mill,
        s_max_mill=s_max_mill, screen_cap=screen_cap, p80_prod=p80_prod, f80_eff=f80_eff, x_r=x_r,
        n=n, a=a, F80=F80, unstable=(S >= 14.5 * F),
        ball_kgph=ball_kgph, liner_kgph=liner_kgph, cost_energy=cost_energy, cost_steel=cost_steel,
        cost_total=cost_energy + cost_steel, q_air=q_air, cloth=q_air / 60.0 / 1.1, dp=dp,
        fan_kw=q_air / 3600.0 * (dp + 2500.0) / 1000.0 / 0.65,
        crusher_kw=0.5 * F, css=p["f80_mm"] / 0.8, oil_t=32.0 + 0.018 * kw,
        dry_tph=F * (1 - p["moist"] / 100.0),
    )


def component_split(p, r, mat):
    """Reparto indicativo por componente (molienda preferencial en mezclas)."""
    rows = []
    for name, w, m in mat["components"]:
        wi_i = m["wi"] * (1 + p["wi_adj"] / 100.0)
        inv = 1.0 / math.sqrt(r["f80_eff"]) + r["E_total"] / (10.0 * wi_i)
        x80 = 1.0 / inv**2
        x63 = x80 / (math.log(5.0) ** (1.0 / m["n"]))
        f_i = float(rr_cdf(r["a"], x63, m["n"]))
        F_i = w * r["F"]
        S_i = min(F_i / max(f_i * r["eta"], 1e-3), 15.0 * F_i)
        rows.append(dict(name=name, w=w, wi=wi_i, mohs=m["mohs"], x80=x80, f=f_i,
                         R=S_i - F_i, cl=(S_i - F_i) / F_i * 100.0))
    tot = sum(x["R"] for x in rows) or 1.0
    for x in rows:
        x["share_R"] = x["R"] / tot
        x["enrich"] = x["share_R"] / x["w"] if x["w"] > 0 else 0.0
    return rows


def alarms(p, r, mat):
    out = []
    if not p["running"]:
        return [("info", "Planta detenida.")]
    if r["unstable"] or r["cl"] > 400:
        out.append(("error", f"Carga circulante excesiva ({r['cl']:.0f} %): el circuito está inestable. "
                             "Baja la alimentación, aumenta la abertura o limpia la zaranda."))
    elif r["cl"] > 250:
        out.append(("warning", f"Carga circulante alta ({r['cl']:.0f} %)."))
    if r["util_mill"] > 1.0:
        out.append(("error", "Molino sobrecargado: el flujo total supera su capacidad de referencia."))
    elif r["util_mill"] > 0.85:
        out.append(("warning", f"Molino cerca de su límite ({r['util_mill'] * 100:.0f} % de la capacidad)."))
    if r["load"] > 1.0:
        out.append(("warning", f"Zaranda sobrecargada ({r['load'] * 100:.0f} %): la eficiencia cae a {r['eta'] * 100:.0f} %."))
    if p["moist"] > 6:
        out.append(("warning", f"Humedad alta ({p['moist']:.1f} %): riesgo de pegado en tolva, banda y malla. "
                               "Considera secado previo."))
    elif p["moist"] > 3:
        out.append(("info", f"Humedad de {p['moist']:.1f} %: la eficiencia de la zaranda se reduce."))
    if p["ai"] > 0.5:
        out.append(("warning", f"Material muy abrasivo (Ai = {p['ai']:.2f}): desgaste acelerado de bolas y forros."))
    if mat.get("hazard"):
        out.append(("warning", mat["hazard"]))
    if r["dp"] > 1800:
        out.append(("warning", f"Pérdida de carga del filtro de mangas alta ({r['dp']:.0f} Pa): revisar limpieza por pulsos."))
    if p["nc"] > 82:
        out.append(("warning", "Velocidad del molino muy alta (> 82 % Nc): riesgo de catarata y desgaste de forros."))
    if p["nc"] < 65:
        out.append(("warning", "Velocidad baja (< 65 % Nc): poca elevación de la carga, molienda poco eficiente."))
    if p["J"] > 40:
        out.append(("warning", "Llenado de bolas > 40 %: consumo de acero y energía elevados."))
    if p["hopper_level"] < 15:
        out.append(("error", "Nivel bajo en la tolva de alimentación."))
    if p["silo_level"] > 95:
        out.append(("error", "Silo de producto casi lleno."))
    tgt = p["target_mm"] * 1000
    if r["p80_prod"] > tgt * 1.3:
        out.append(("error", f"P80 del producto ({r['p80_prod']:.0f} µm) fuera de especificación (objetivo {tgt:.0f} µm)."))
    elif r["p80_prod"] > tgt * 1.1:
        out.append(("warning", f"P80 del producto ({r['p80_prod']:.0f} µm) ligeramente sobre el objetivo ({tgt:.0f} µm)."))
    if not out:
        out.append(("success", "Operación normal: todos los indicadores dentro de rango."))
    return out


# ----------------------------------------------------------------------------
# 3. INSTRUMENTACIÓN
# ----------------------------------------------------------------------------
LEVEL_TXT = {"ok": "Normal", "warn": "Advertencia", "alarm": "Alarma", "off": "Detenido"}
OX, OY = 60, 190  # desplazamiento del bloque principal del molino (coordenadas "viejas")


def build_instruments(p, r, mat):
    run = p["running"]

    def lvl(alarm, warn):
        if not run:
            return "off"
        return "alarm" if alarm else ("warn" if warn else "ok")

    tgt = p["target_mm"] * 1000
    z = (lambda v: v if run else 0.0)
    L = []

    def add(tag, desc, val, unit, level, pos=None, target=None, sub=None, absolute=False, fmt="{:.1f}"):
        if pos is not None and not absolute:
            pos = (pos[0] + OX, pos[1] + OY)
            target = (target[0] + OX, target[1] + OY) if target else None
        L.append(dict(tag=tag, desc=desc, val=val, unit=unit, level=level, pos=pos,
                      target=target, sub=sub, fmt=fmt))

    add("JIT-100", "Potencia chancadora", z(r["crusher_kw"]), "kW", "ok" if run else "off",
        (345, 160), (322, 140), f"CSS {r['css']:.0f} mm", True, "{:.0f}")
    add("LIT-101", "Nivel tolva TK-101", p["hopper_level"], "%", lvl(p["hopper_level"] < 15, p["hopper_level"] < 30),
        (10, 72), (82, 72), "nivel tolva", fmt="{:.0f}")
    add("WIT-102", "Pesómetro alimentador", z(r["F"]), "t/h", lvl(r["util_mill"] > 1.0, r["util_mill"] > 0.85),
        (250, 212), (250, 162), "pesómetro", fmt="{:.1f}")
    add("SIC-202", "Velocidad del molino", z(r["rpm"]), "rpm", lvl(False, p["nc"] > 82 or p["nc"] < 65),
        (480, 262), (421, 276), f"{p['nc']} % Nc", fmt="{:.1f}")
    add("JIT-201", "Potencia del molino", z(r["kw"]), "kW", lvl(r["util_mill"] > 1.0, r["util_mill"] > 0.85),
        (480, 346), (427, 336), f"{r['E_fresh']:.1f} kWh/t", fmt="{:.0f}")
    add("TIT-203", "Temp. aceite lubricación", z(r["oil_t"]) if run else 25.0, "°C", lvl(False, r["oil_t"] > 60),
        (45, 392), (100, 392), "aceite", fmt="{:.0f}")
    add("WI-201", "Consumo de acero (bolas+forros)", z(r["ball_kgph"] + r["liner_kgph"]), "kg/h",
        lvl(False, p["ai"] > 0.5), (45, 316), (100, 316), "bolas+forros", fmt="{:.1f}")
    add("XI-401", "Abertura de malla", p["aperture_mm"], "mm", "ok" if run else "off",
        (395, 62), (578, 88), "abertura", fmt="{:.2f}")
    add("ET-401", "Eficiencia de la zaranda", z(r["eta"] * 100), "%", lvl(r["eta"] < 0.55, r["eta"] < 0.7),
        (500, 62), (578, 74), "eficiencia", fmt="{:.0f}")
    add("FIC-402", "Carga circulante", z(r["cl"]), "%", lvl(r["cl"] > 400 or r["unstable"], r["cl"] > 250),
        (505, 120), (578, 120), "carga circ.", fmt="{:.0f}")
    add("LIT-501", "Nivel silo TK-501", p["silo_level"], "%", lvl(p["silo_level"] > 95, p["silo_level"] > 88),
        (808, 300), (770, 300), None, fmt="{:.0f}")
    add("AIT-501", "P80 del producto (analizador láser)", r["p80_prod"], "µm",
        lvl(r["p80_prod"] > tgt * 1.3, r["p80_prod"] > tgt * 1.1), (808, 384), (782, 369), None, fmt="{:.0f}")
    add("PDIT-601", "Pérdida de carga filtro de mangas", z(r["dp"]), "Pa", lvl(r["dp"] > 2200, r["dp"] > 1800),
        (530, 100), (560, 100), "Δp filtro", True, "{:.0f}")
    add("FIT-601", "Caudal de aire del colector", z(r["q_air"]), "m³/h", "ok" if run else "off",
        (880, 70), (812, 96), f"{z(r['fan_kw']):.1f} kW", True, "{:.0f}")
    return L


# ----------------------------------------------------------------------------
# 4. DIAGRAMA INDUSTRIAL (SVG ANIMADO)
# ----------------------------------------------------------------------------
W, H = 1440, 800
COL = {"ok": "#3ddc97", "warn": "#f5b942", "alarm": "#ff5c5c", "off": "#64748b"}
C_FRESH, C_MIX, C_DISCH, C_OVER, C_PROD = "#f5b942", "#ffb36b", "#d9a05b", "#ff7a59", "#3ddc97"
C_AIR, C_OIL = "#6fb4ff", "#e0c25a"

VIEWS = {
    "Vista completa": (0, 0, 1440, 800),
    "Preparación y trituración": (0, 0, 720, 400),
    "Molino de bolas": (110, 330, 600, 333),
    "Zaranda, elevador y silo": (520, 200, 720, 400),
    "Colector de polvo": (480, 0, 640, 356),
}

CSS = """
html,body{margin:0;height:100%;background:#0a1222;font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden}
.wrap{position:relative;height:100%}
svg{width:100%;height:100%;display:block;cursor:grab}
.tb{position:absolute;right:12px;top:10px;z-index:5;display:flex;gap:6px}
.tb button{background:#14213a;color:#cfdcf2;border:1px solid #2c4166;border-radius:6px;width:30px;height:28px;font-size:15px;cursor:pointer}
.tb button:hover{background:#1c2f52}
.hint{position:absolute;left:12px;bottom:8px;font:11px 'Segoe UI',sans-serif;color:#6f84a6;z-index:5}
.pipe{fill:none;stroke:#1d2b42;stroke-width:11;stroke-linejoin:round;stroke-linecap:round}
.duct{fill:none;stroke:#1a2c4a;stroke-width:9;stroke-linejoin:round}
.air{fill:none;stroke:#6fb4ff;stroke-width:2.2;stroke-dasharray:3 9;stroke-linecap:round;animation:dash linear infinite}
.oil{fill:none;stroke:#e0c25a;stroke-width:2;stroke-dasharray:2 7;stroke-linecap:round;animation:dash linear infinite}
.flow{fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:2 11;animation:dash linear infinite}
@keyframes dash{to{stroke-dashoffset:-39}}
.belt{fill:none;stroke:#8aa0c0;stroke-width:2;stroke-dasharray:7 7;animation:belt linear infinite}
@keyframes belt{to{stroke-dashoffset:-28}}
.bucket{fill:none;stroke:#5f7596;stroke-width:16;stroke-dasharray:9 17;animation:bucket linear infinite}
@keyframes bucket{to{stroke-dashoffset:-52}}
.spin{animation:spin linear infinite}
.spinr{animation:spin linear infinite reverse}
@keyframes spin{to{transform:rotate(360deg)}}
.slosh{animation:slosh ease-in-out infinite alternate}
@keyframes slosh{from{transform:rotate(-5deg)}to{transform:rotate(5deg)}}
.rock{animation:rock ease-in-out infinite alternate}
@keyframes rock{from{transform:rotate(-2.2deg)}to{transform:rotate(2.6deg)}}
.vib{animation:vib linear infinite}
@keyframes vib{0%{transform:translate(0,0)}25%{transform:translate(1.6px,-2px)}50%{transform:translate(-1.2px,1.6px)}75%{transform:translate(1.2px,2px)}100%{transform:translate(0,0)}}
.blink{animation:blink 1s steps(2,start) infinite}
@keyframes blink{50%{opacity:.35}}
.puff{animation:puff 3s ease-out infinite;opacity:0}
@keyframes puff{0%{transform:translateY(0) scale(.6);opacity:.55}100%{transform:translateY(-38px) scale(1.7);opacity:0}}
.pulse{animation:pulse 2.4s steps(1) infinite}
@keyframes pulse{0%{fill:#2c4166}8%{fill:#6fb4ff}16%,100%{fill:#2c4166}}
.off *{animation-play-state:paused !important}
.name{font:600 11.5px 'Segoe UI',sans-serif;fill:#cfdcf2}
.sub{font:10.5px 'Segoe UI',sans-serif;fill:#7f93b3}
.val{font:700 11.5px 'Consolas','Courier New',monospace}
.stream{font:600 11px 'Consolas','Courier New',monospace;fill:#e8eefb}
.loop{font:700 8.5px 'Consolas',monospace;fill:#cfdcf2}
.big{font:700 17px 'Consolas','Courier New',monospace}
"""

JS = """
(function(){
  var svg=document.getElementById('plant');
  var FW=__W__, FH=__H__, INIT=[__X__,__Y__,__VW__,__VH__];
  var vb={x:INIT[0],y:INIT[1],w:INIT[2],h:INIT[3]};
  function clamp(){
    vb.w=Math.min(Math.max(vb.w,FW/8),FW); vb.h=vb.w*FH/FW;
    vb.x=Math.min(Math.max(vb.x,0),FW-vb.w); vb.y=Math.min(Math.max(vb.y,0),FH-vb.h);
  }
  function apply(){clamp(); svg.setAttribute('viewBox',vb.x+' '+vb.y+' '+vb.w+' '+vb.h);}
  function toUser(cx,cy){var pt=svg.createSVGPoint();pt.x=cx;pt.y=cy;return pt.matrixTransform(svg.getScreenCTM().inverse());}
  function zoomAt(cx,cy,k){var p=toUser(cx,cy);vb.x=p.x-(p.x-vb.x)*k;vb.y=p.y-(p.y-vb.y)*k;vb.w*=k;vb.h*=k;apply();}
  function center(k){var r=svg.getBoundingClientRect();zoomAt(r.left+r.width/2,r.top+r.height/2,k);}
  document.getElementById('zi').onclick=function(){center(0.75);};
  document.getElementById('zo').onclick=function(){center(1/0.75);};
  document.getElementById('zr').onclick=function(){vb={x:INIT[0],y:INIT[1],w:INIT[2],h:INIT[3]};apply();};
  svg.addEventListener('dblclick',function(){document.getElementById('zr').onclick();});
  svg.addEventListener('wheel',function(e){
    if(!(e.ctrlKey||e.metaKey||e.shiftKey)) return;
    e.preventDefault(); zoomAt(e.clientX,e.clientY,e.deltaY<0?0.85:1/0.85);
  },{passive:false});
  var drag=null;
  svg.addEventListener('mousedown',function(e){drag={x:e.clientX,y:e.clientY};svg.style.cursor='grabbing';});
  window.addEventListener('mouseup',function(){drag=null;svg.style.cursor='grab';});
  window.addEventListener('mousemove',function(e){
    if(!drag) return;
    var s=svg.getScreenCTM().a||1;
    vb.x-=(e.clientX-drag.x)/s; vb.y-=(e.clientY-drag.y)/s; drag={x:e.clientX,y:e.clientY}; apply();
  });
  apply();
})();
"""


def esc(s):
    return html.escape(str(s), quote=True)


def dur(rate, ref, slow=6.0, fast=0.5):
    if rate <= 0:
        return slow
    return float(min(slow, max(fast, 2.2 * ref / rate)))


def flow(d, color, rate, ref, width=None):
    w = width if width else 2.6 + 3.4 * min(1.0, rate / (ref * 2.5))
    return (f'<path class="pipe" d="{d}"/>'
            f'<path class="flow" d="{d}" style="stroke:{color};stroke-width:{w:.1f};'
            f'animation-duration:{dur(rate, ref):.2f}s"/>')


def belt_svg(x, y, w, d_s, direction=1, h=14, motor=True):
    """Banda transportadora con rodillos y poleas. direction=+1: izquierda->derecha."""
    x1, x2 = (x + 9, x + w - 9) if direction > 0 else (x + w - 9, x + 9)
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2:.0f}" fill="#16223a" '
         f'stroke="#8fa3c2" stroke-width="2"/>'
         f'<line x1="{x1}" y1="{y + h * 0.35:.1f}" x2="{x2}" y2="{y + h * 0.35:.1f}" class="belt" '
         f'style="animation-duration:{d_s:.2f}s"/>')
    for ix in range(int(x + 24), int(x + w - 12), 26):
        s += f'<circle cx="{ix}" cy="{y + h + 4}" r="2.6" fill="#44587a"/>'
    for px in (x + h / 2, x + w - h / 2):
        s += (f'<circle cx="{px:.0f}" cy="{y + h / 2:.0f}" r="{h / 2 + 2:.0f}" fill="#1d2b42" '
              f'stroke="#8fa3c2" stroke-width="2"/>')
    if motor:
        mx = (x + w + 2) if direction > 0 else (x - 20)
        s += (f'<rect x="{mx}" y="{y + 1}" width="18" height="{h - 2}" rx="3" fill="#1f3354" stroke="#8fa3c2" stroke-width="1.5"/>'
              f'<text x="{mx + 9}" y="{y + h - 3}" text-anchor="middle" class="loop">M</text>')
    return s


def bubble(cx, cy, tag, level, target=None, lines=()):
    loop, num = tag.split("-")
    col = COL[level]
    s = ""
    if target:
        s += (f'<line x1="{cx}" y1="{cy}" x2="{target[0]}" y2="{target[1]}" stroke="#6e809f" '
              f'stroke-width="1" stroke-dasharray="3 3"/>')
    cls = ' class="blink"' if level == "alarm" else ""
    s += (f'<g{cls}><circle cx="{cx}" cy="{cy}" r="16" fill="#0e1729" stroke="{col}" stroke-width="2"/>'
          f'<line x1="{cx - 16}" y1="{cy}" x2="{cx + 16}" y2="{cy}" stroke="{col}" stroke-width="1" opacity=".7"/>'
          f'<text class="loop" x="{cx}" y="{cy - 4}" text-anchor="middle">{loop}</text>'
          f'<text class="loop" x="{cx}" y="{cy + 11}" text-anchor="middle">{num}</text></g>')
    y = cy + 32
    for i, t in enumerate(lines):
        if i == 0:
            s += f'<text class="val" x="{cx}" y="{y}" text-anchor="middle" fill="{col}">{esc(t)}</text>'
        else:
            s += f'<text class="sub" x="{cx}" y="{y}" text-anchor="middle">{esc(t)}</text>'
        y += 12
    return s


def chord_height(J, r):
    target = 2 * math.pi * J
    lo, hi = 0.0, 2 * math.pi
    for _ in range(50):
        mid = (lo + hi) / 2
        if mid - math.sin(mid) < target:
            lo = mid
        else:
            hi = mid
    th = (lo + hi) / 2
    return r * (1 - math.cos(th / 2))


def mill_svg(cx, cy, r, J, rpm, running):
    rin = r - 14
    level_y = cy + rin - chord_height(J, rin)
    rng = np.random.default_rng(11)
    balls, tries = [], 0
    nballs = int(30 + J * 260)
    while len(balls) < nballs and tries < 6000:
        tries += 1
        x = cx + rng.uniform(-rin, rin)
        y = cy + rng.uniform(-rin, rin)
        if (x - cx) ** 2 + (y - cy) ** 2 < (rin - 6) ** 2 and y > level_y + 3:
            balls.append((x, y))
    ball_svg = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.6" fill="#8fa5c6" stroke="#2a3b57" stroke-width="1"/>'
        for x, y in balls)
    lifters = "".join(
        f'<line x1="{cx + 69 * math.cos(math.radians(k * 22.5)):.1f}" y1="{cy + 69 * math.sin(math.radians(k * 22.5)):.1f}" '
        f'x2="{cx + 80 * math.cos(math.radians(k * 22.5)):.1f}" y2="{cy + 80 * math.sin(math.radians(k * 22.5)):.1f}" '
        f'stroke="#9db1d1" stroke-width="5" stroke-linecap="round"/>' for k in range(16))
    t_spin = max(1.4, 60.0 / max(rpm, 1) * 0.55) if running else 4
    t_slosh = max(0.9, t_spin * 0.6)
    t_gear = t_spin * 1.0
    return f"""
    <g class="spin" style="transform-origin:{cx}px {cy}px;animation-duration:{t_gear:.2f}s">
      <circle cx="{cx}" cy="{cy}" r="{r + 12}" fill="none" stroke="#5c7197" stroke-width="6" stroke-dasharray="3 2.2"/>
    </g>
    <circle cx="{cx}" cy="{cy}" r="{r + 5}" fill="none" stroke="#1a2740" stroke-width="3"/>
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="#101b30" stroke="#8fa3c2" stroke-width="6"/>
    <g class="spin" style="transform-origin:{cx}px {cy}px;animation-duration:{t_spin:.2f}s">
      <circle cx="{cx}" cy="{cy}" r="{r - 9}" fill="none" stroke="#33476b" stroke-width="2" stroke-dasharray="6 5"/>
      {lifters}
    </g>
    <g transform="rotate(28 {cx} {cy})">
      <g class="slosh" style="transform-origin:{cx}px {cy}px;animation-duration:{t_slosh:.2f}s">
        {ball_svg}
      </g>
    </g>
    <circle cx="{cx}" cy="{cy}" r="5" fill="#cfdcf2"/>
    """


def fan_blades(cx, cy, rr):
    s = ""
    for k in range(6):
        ang = k * 60
        s += (f'<ellipse cx="{cx}" cy="{cy - rr * 0.55:.1f}" rx="{rr * 0.2:.1f}" ry="{rr * 0.5:.1f}" '
              f'fill="#5f7596" stroke="#9db1d1" stroke-width="1" transform="rotate({ang} {cx} {cy})"/>')
    return s


def panels(p, r, mat, inst_levels):
    run = p["running"]
    st_col = COL["ok"] if run else COL["off"]
    name = mat["name"]
    short = name if len(name) <= 34 else name[:33] + "…"
    lv_p80, lv_cl = inst_levels["p80"], inst_levels["cl"]
    # composición de la mezcla
    comp_svg = ""
    comps = mat["components"]
    if len(comps) > 1:
        x = 0.0
        bar_w = 226.0
        for i, (n, w, m) in enumerate(comps):
            comp_svg += (f'<rect x="{12 + x:.1f}" y="154" width="{w * bar_w:.1f}" height="9" fill="{m["color"]}" '
                         f'stroke="#0a1222" stroke-width="1"/>')
            x += w * bar_w
        for i, (n, w, m) in enumerate(comps[:4]):
            comp_svg += (f'<rect x="12" y="{172 + i * 13}" width="8" height="8" fill="{m["color"]}"/>'
                         f'<text class="sub" x="26" y="{180 + i * 13}">{esc(n[:26])} · {w * 100:.0f} %</text>')
    streams = [(C_FRESH, "① Alimentación fresca", r["F"]), (C_MIX, "② Alim. al molino", r["S"]),
               (C_DISCH, "③ Descarga del molino", r["S"]), (C_OVER, "④ Oversize (retorno)", r["R"]),
               (C_PROD, "⑤ Producto (undersize)", r["F"]), (C_AIR, "Aire / polvo", r["q_air"] / 1000.0)]
    st_svg = ""
    for i, (c, t, v) in enumerate(streams):
        unit = "mil m³/h" if i == 5 else "t/h"
        vv = (v if run else 0)
        st_svg += (f'<rect x="12" y="{38 + i * 18}" width="14" height="4" fill="{c}"/>'
                   f'<text class="sub" x="34" y="{43 + i * 18}">{t}</text>'
                   f'<text class="val" x="258" y="{43 + i * 18}" text-anchor="end" fill="#cfdcf2">{vv:.1f} {unit}</text>')
    props = [("Wi (Bond)", f"{p['wi']:.1f} kWh/t"), ("Densidad real", f"{p['rho']:.2f} t/m³"),
             ("Densidad aparente", f"{p['bulk']:.2f} t/m³"), ("Dureza Mohs", f"{p['mohs']:.1f}"),
             ("Abrasión Ai", f"{p['ai']:.3f}"), ("Humedad", f"{p['moist']:.1f} %")]
    pr_svg = "".join(f'<text class="sub" x="12" y="{50 + i * 14}">{a}</text>'
                     f'<text class="val" x="258" y="{50 + i * 14}" text-anchor="end" fill="#cfdcf2">{b}</text>'
                     for i, (a, b) in enumerate(props))
    h2 = 160 if len(comps) <= 1 else 236
    return f"""
    <g transform="translate(1150,20)">
      <rect width="270" height="190" rx="8" fill="#0e1729" stroke="#26385a"/>
      <circle cx="16" cy="18" r="6" fill="{st_col}" class="{'blink' if not run else ''}"/>
      <text class="name" x="30" y="22">{'PLANTA EN MARCHA' if run else 'PLANTA DETENIDA'}</text>
      <rect x="12" y="34" width="12" height="12" rx="3" fill="{mat['color']}" stroke="#cfdcf2" stroke-width=".8"/>
      <text class="name" x="30" y="44">{esc(short)}</text>
      <text class="sub" x="12" y="70">Producto P80</text>
      <text class="big" x="12" y="90" fill="{COL[lv_p80]}">{r['p80_prod']:.0f} µm</text>
      <text class="sub" x="140" y="70">Energía específica</text>
      <text class="big" x="140" y="90" fill="#cfdcf2">{r['E_fresh']:.1f} kWh/t</text>
      <text class="sub" x="12" y="118">Carga circulante</text>
      <text class="big" x="12" y="138" fill="{COL[lv_cl]}">{(r['cl'] if run else 0):.0f} %</text>
      <text class="sub" x="140" y="118">Costo operativo</text>
      <text class="big" x="140" y="138" fill="#cfdcf2">{r['cost_total']:.2f} USD/t</text>
      <text class="sub" x="12" y="166">Producción {(r['F'] if run else 0):.0f} t/h · base seca {(r['dry_tph'] if run else 0):.0f} t/h</text>
    </g>
    <g transform="translate(1150,226)">
      <rect width="270" height="{h2 + 20}" rx="8" fill="#0e1729" stroke="#26385a"/>
      <text class="name" x="12" y="20">Propiedades del material</text>
      {pr_svg}
      {comp_svg}
    </g>
    <g transform="translate(1150,{226 + h2 + 36})">
      <rect width="270" height="150" rx="8" fill="#0e1729" stroke="#26385a"/>
      <text class="name" x="12" y="20">Corrientes</text>
      {st_svg}
    </g>
    <g transform="translate(1150,{226 + h2 + 202})">
      <rect width="270" height="76" rx="8" fill="#0e1729" stroke="#26385a"/>
      <circle cx="18" cy="20" r="5" fill="{COL['ok']}"/><text class="sub" x="30" y="24">Normal</text>
      <circle cx="98" cy="20" r="5" fill="{COL['warn']}"/><text class="sub" x="110" y="24">Advertencia</text>
      <circle cx="196" cy="20" r="5" fill="{COL['alarm']}"/><text class="sub" x="208" y="24">Alarma</text>
      <text class="sub" x="12" y="46">Ctrl + rueda: zoom · arrastrar: mover</text>
      <text class="sub" x="12" y="62">Doble clic: restablecer la vista</text>
    </g>
    """


def plant_html(p, r, mat, view_name):
    run = p["running"]
    F = r["F"] if run else 0.0
    S = r["S"] if run else 0.0
    R = r["R"] if run else 0.0
    Q = r["q_air"] if run else 0.0
    ref = max(p["feed_tph"], 50.0)
    mcol = mat["color"]
    inst = build_instruments(p, r, mat)
    levels = {i["tag"]: i["level"] for i in inst}
    lv = {"p80": levels["AIT-501"], "cl": levels["FIC-402"]}
    t_air = dur(Q / 60.0, 100.0, slow=7, fast=0.5)

    # ===================== PREPARACIÓN (coordenadas absolutas) =====================
    d_fp = dur(F, ref, slow=9, fast=0.4)
    upstream = f"""
    <polygon points="350,24 460,24 435,70 375,70" fill="#101b30" stroke="#8fa3c2" stroke-width="2.5"/>
    {''.join(f'<line x1="{362 + i * 20}" y1="26" x2="{368 + i * 20}" y2="40" stroke="#5f7596" stroke-width="2"/>' for i in range(5))}
    <rect x="396" y="70" width="18" height="12" fill="#1d2b42" stroke="#8fa3c2" stroke-width="2"/>
    <text class="name" x="405" y="14" text-anchor="middle">TK-100 · Tolva de recepción (parrilla)</text>
    {belt_svg(240, 84, 208, d_fp, -1)}
    <text class="sub" x="362" y="120">FP-100 · Alimentador de placas</text>
    <polygon points="206,108 296,108 276,182 226,182" fill="#101b30" stroke="#8fa3c2" stroke-width="2.5"/>
    <line x1="215" y1="112" x2="233" y2="176" stroke="#9db1d1" stroke-width="5" stroke-linecap="round"/>
    <g class="rock" style="transform-origin:290px 112px;animation-duration:{0.5 if run else 2}s">
      <line x1="290" y1="112" x2="262" y2="176" stroke="#cfdcf2" stroke-width="5" stroke-linecap="round"/>
    </g>
    <line x1="296" y1="120" x2="316" y2="136" stroke="#8fa3c2" stroke-width="5"/>
    <g class="spin" style="transform-origin:320px 140px;animation-duration:{1.1 if run else 4}s">
      <circle cx="320" cy="140" r="20" fill="#16223a" stroke="#8fa3c2" stroke-width="3"/>
      <line x1="320" y1="122" x2="320" y2="158" stroke="#8fa3c2" stroke-width="2"/>
      <line x1="302" y1="140" x2="338" y2="140" stroke="#8fa3c2" stroke-width="2"/>
    </g>
    <text class="name" x="198" y="132" text-anchor="end">CR-100</text>
    <text class="sub" x="198" y="144" text-anchor="end">Chancadora de</text>
    <text class="sub" x="198" y="156" text-anchor="end">mandíbulas</text>
    {belt_svg(160, 190, 140, d_fp, -1, h=12)}
    <text class="sub" x="300" y="214" text-anchor="end">CV-100 · Banda a tolva</text>
    {flow("M405 72 L405 90", C_FRESH, F, ref, 4)}
    {flow("M246 98 L246 118", C_FRESH, F, ref, 4)}
    {flow("M251 182 L251 196", C_FRESH, F, ref, 4)}
    {flow("M172 200 L172 238", C_FRESH, F, ref, 4)}
    """

    # ===================== DUCTOS Y COLECTOR DE POLVO =====================
    ducts = ""
    for d in ("M368 392 L400 392 L400 168 L592 168", "M680 236 L680 170", "M710 100 L768 100"):
        ducts += f'<path class="duct" d="{d}"/><path class="air" d="{d}" style="animation-duration:{t_air:.2f}s"/>'
    bags = "".join(f'<rect x="{572 + i * 16}" y="58" width="9" height="76" rx="4" fill="#16223a" stroke="#5f7596" stroke-width="1.2"/>'
                   for i in range(8))
    pulse = "".join(f'<circle class="pulse" cx="{576 + i * 32}" cy="50" r="4" style="animation-delay:{i * 0.6:.1f}s;fill:#2c4166"/>'
                    for i in range(4))
    puffs = "".join(f'<circle class="puff" cx="790" cy="16" r="6" fill="#9fb3d1" style="animation-delay:{i}s"/>'
                    for i in range(3)) if run else ""
    dust = f"""
    <rect x="560" y="40" width="150" height="110" rx="6" fill="#101b30" stroke="#8fa3c2" stroke-width="2.5"/>
    <line x1="560" y1="56" x2="710" y2="56" stroke="#33476b" stroke-width="1.5"/>
    {bags}{pulse}
    <polygon points="560,150 710,150 650,186 620,186" fill="#101b30" stroke="#8fa3c2" stroke-width="2.5"/>
    <g class="spin" style="transform-origin:635px 196px;animation-duration:{1.6 if run else 5}s">
      <circle cx="635" cy="196" r="10" fill="#16223a" stroke="#8fa3c2" stroke-width="2"/>
      <line x1="625" y1="196" x2="645" y2="196" stroke="#8fa3c2" stroke-width="2"/>
      <line x1="635" y1="186" x2="635" y2="206" stroke="#8fa3c2" stroke-width="2"/>
    </g>
    <text class="sub" x="652" y="216">RV-601 · Polvo recuperado</text>
    <rect x="490" y="64" width="42" height="22" rx="10" fill="#16223a" stroke="#5f7596" stroke-width="1.5"/>
    <text class="sub" x="511" y="97" text-anchor="middle">AR-601</text>
    <text class="name" x="635" y="32" text-anchor="middle">BF-601 · Filtro de mangas</text>
    <g class="spin" style="transform-origin:790px 100px;animation-duration:{0.5 if run else 4}s">
      <circle cx="790" cy="100" r="24" fill="#16223a" stroke="#8fa3c2" stroke-width="2.5"/>
      {fan_blades(790, 100, 22)}
    </g>
    <circle cx="790" cy="100" r="5" fill="#cfdcf2"/>
    <rect x="780" y="26" width="20" height="50" fill="#101b30" stroke="#8fa3c2" stroke-width="2"/>
    <line x1="790" y1="76" x2="790" y2="82" stroke="#8fa3c2" stroke-width="6"/>
    {puffs}
    <text class="sub" x="790" y="140" text-anchor="middle">FN-601 · Ventilador</text>
    <text class="sub" x="826" y="44">CH-601 · Chimenea</text>
    """

    # ===================== BLOQUE PRINCIPAL (coords "viejas" + traslación) =====================
    hop_poly = "60,30 180,30 135,115 105,115"
    y_fill = 30 + (1 - p["hopper_level"] / 100.0) * 85
    t_belt = dur(F, ref, slow=9, fast=0.4)
    t_b3 = dur(S, ref, slow=9, fast=0.4)
    t_b4 = dur(R, ref, slow=9, fast=0.4)
    t_el = dur(S, ref, slow=9, fast=0.4)
    t_vib = 0.09 if run else 1
    sil_poly = "650,246 770,246 770,360 720,398 700,398 650,360"
    y_s = 246 + (1 - p["silo_level"] / 100.0) * 152

    flows = "".join([
        flow("M120 133 L120 146", C_FRESH, F, ref, 4),
        flow("M232 154 L300 154 L300 176", C_FRESH, F, ref),
        flow("M588 128 L588 176 L304 176", C_OVER, R, ref),
        flow("M300 176 L300 228", C_MIX, S, ref),
        flow("M340 398 L340 478 L860 478 L860 48 L830 64", C_DISCH, S, ref),
        flow("M710 112 L710 250", C_PROD, F, ref),
        flow("M710 414 L710 436 L640 436", C_PROD, F, ref, 4),
    ])
    oil_path = "M190 392 L255 392 L275 376"
    main = f"""
    <g transform="translate({OX},{OY})">
      {flows}
      <path class="oil" d="{oil_path}" style="animation-duration:{2.5 if run else 8}s"/>
      <!-- tolva TK-101 -->
      <clipPath id="clipH"><polygon points="{hop_poly}"/></clipPath>
      <polygon points="{hop_poly}" fill="#101b30" stroke="#8fa3c2" stroke-width="2.5"/>
      <rect x="50" y="{y_fill:.1f}" width="140" height="90" fill="{mcol}" opacity=".8" clip-path="url(#clipH)"/>
      <rect x="109" y="115" width="22" height="18" fill="#1d2b42" stroke="#8fa3c2" stroke-width="2"/>
      <line x1="68" y1="30" x2="78" y2="12" stroke="#8fa3c2" stroke-width="2"/>
      <line x1="172" y1="30" x2="162" y2="12" stroke="#8fa3c2" stroke-width="2"/>
      <!-- alimentador FE-101 + imán -->
      {belt_svg(60, 146, 205, t_belt * 0.9, 1, h=16, motor=False)}
      <rect x="150" y="122" width="52" height="12" rx="6" fill="#16223a" stroke="#c0586a" stroke-width="2"/>
      <line x1="156" y1="128" x2="196" y2="128" class="belt" style="animation-duration:{t_belt:.2f}s;stroke:#c0586a"/>
      <rect x="206" y="120" width="28" height="18" rx="3" fill="#16223a" stroke="#c0586a" stroke-width="1.5"/>
      <text x="220" y="133" text-anchor="middle" class="loop">Fe</text>
      <text class="name" x="130" y="184" text-anchor="middle">FE-101 · Alimentador de banda</text>
      <text class="sub" x="130" y="196" text-anchor="middle">MS-101 · Imán de banda (Fe)</text>
      <!-- molino ML-201, motor, reductor y lubricación -->
      <rect x="100" y="288" width="80" height="52" rx="6" fill="#16223a" stroke="#8fa3c2" stroke-width="2"/>
      <text x="140" y="320" text-anchor="middle" class="name" style="font-size:17px">M</text>
      <line x1="180" y1="314" x2="190" y2="314" stroke="#8fa3c2" stroke-width="6"/>
      <rect x="190" y="298" width="32" height="32" rx="4" fill="#1f3354" stroke="#8fa3c2" stroke-width="2"/>
      <text x="206" y="318" text-anchor="middle" class="loop">GB</text>
      <line x1="222" y1="314" x2="230" y2="314" stroke="#8fa3c2" stroke-width="5"/>
      <g class="spinr" style="transform-origin:236px 312px;animation-duration:{max(0.5, 60.0 / max(r['rpm'], 1) * 0.3) if run else 3:.2f}s">
        <circle cx="236" cy="312" r="8" fill="#16223a" stroke="#9db1d1" stroke-width="3" stroke-dasharray="3 2"/>
      </g>
      <text class="sub" x="140" y="357" text-anchor="middle">M-201 · Motor principal</text>
      <rect x="100" y="372" width="90" height="38" rx="5" fill="#16223a" stroke="{C_OIL}" stroke-width="1.5"/>
      <text class="sub" x="145" y="388" text-anchor="middle">LU-201 · Lubricación</text>
      <text class="val" x="145" y="402" text-anchor="middle" fill="{C_OIL}">{(r['oil_t'] if run else 25):.0f} °C</text>
      {mill_svg(340, 312, 88, p['J'] / 100.0, r['rpm'], run)}
      <text class="name" x="322" y="440" text-anchor="end">ML-201 · Molino de bolas</text>
      <text class="sub" x="322" y="453" text-anchor="end">{p['D']:.1f} × {p['L']:.1f} m · J = {p['J']} % · {p['nc']} % Nc</text>
      <!-- bandas -->
      {belt_svg(296, 170, 300, t_b4, -1, h=12, motor=False)}
      <text class="sub" x="446" y="198" text-anchor="middle">CV-302 · Banda de retorno (oversize)</text>
      {belt_svg(330, 472, 545, t_b3, 1, h=12, motor=False)}
      <text class="name" x="600" y="505" text-anchor="middle">CV-301 · Banda de descarga del molino</text>
      <!-- elevador -->
      <rect x="846" y="40" width="28" height="440" rx="4" fill="#101b30" stroke="#8fa3c2" stroke-width="2"/>
      <path class="bucket" d="M860 470 L860 50" style="animation-duration:{t_el:.2f}s"/>
      <rect x="842" y="22" width="36" height="16" rx="4" fill="#1f3354" stroke="#8fa3c2" stroke-width="1.5"/>
      <text x="860" y="34" text-anchor="middle" class="loop">M-301</text>
      <text class="name" x="893" y="262">EL-301</text>
      <text class="sub" x="893" y="275">Elevador de</text>
      <text class="sub" x="893" y="287">cangilones</text>
      <!-- zaranda SC-401 -->
      <rect x="578" y="46" width="268" height="104" rx="6" fill="none" stroke="#3a4d6e" stroke-width="1.5" stroke-dasharray="5 4"/>
      <polygon points="610,138 818,138 738,172 690,172" fill="#101b30" stroke="#8fa3c2" stroke-width="2"/>
      <g transform="translate(590,120) rotate(-11.8)">
        <g class="vib" style="animation-duration:{t_vib}s">
          <rect x="0" y="0" width="250" height="13" rx="3" fill="#2a3f63" stroke="#9db1d1" stroke-width="2"/>
          <line x1="6" y1="6.5" x2="244" y2="6.5" stroke="#cfdcf2" stroke-width="2" stroke-dasharray="2 4"/>
        </g>
      </g>
      <g class="vib" style="animation-duration:{t_vib}s">
        <rect x="640" y="36" width="26" height="10" rx="3" fill="#1f3354" stroke="#8fa3c2" stroke-width="1.5"/>
        <rect x="770" y="36" width="26" height="10" rx="3" fill="#1f3354" stroke="#8fa3c2" stroke-width="1.5"/>
      </g>
      <path d="M610 146 l-4 6 l8 4 l-8 4 l8 4" fill="none" stroke="#7f93b3" stroke-width="1.5"/>
      <path d="M790 146 l-4 6 l8 4 l-8 4 l8 4" fill="none" stroke="#7f93b3" stroke-width="1.5"/>
      <text class="name" x="712" y="26" text-anchor="middle">SC-401 · Zaranda vibratoria</text>
      <text class="sub" x="712" y="116" text-anchor="middle">VM-401 · motores vibradores</text>
      <!-- silo TK-501 -->
      <clipPath id="clipS"><polygon points="{sil_poly}"/></clipPath>
      <polygon points="{sil_poly}" fill="#101b30" stroke="#8fa3c2" stroke-width="2.5"/>
      <rect x="640" y="{y_s:.1f}" width="140" height="170" fill="{mcol}" opacity=".75" clip-path="url(#clipS)"/>
      <rect x="655" y="226" width="22" height="20" fill="#16223a" stroke="#8fa3c2" stroke-width="1.5"/>
      <text class="sub" x="666" y="222" text-anchor="middle">BV-501</text>
      <rect x="772" y="360" width="26" height="18" rx="3" fill="#16223a" stroke="#8fa3c2" stroke-width="1.5"/>
      <text x="785" y="372" text-anchor="middle" class="loop">SM</text>
      <g class="spin" style="transform-origin:710px 405px;animation-duration:{1.4 if run else 5}s">
        <circle cx="710" cy="405" r="10" fill="#16223a" stroke="#8fa3c2" stroke-width="2"/>
        <line x1="700" y1="405" x2="720" y2="405" stroke="#8fa3c2" stroke-width="2"/>
        <line x1="710" y1="395" x2="710" y2="415" stroke="#8fa3c2" stroke-width="2"/>
      </g>
      <text class="name" x="642" y="318" text-anchor="end">TK-501</text>
      <text class="sub" x="642" y="331" text-anchor="end">Silo de producto</text>
      <text class="sub" x="642" y="343" text-anchor="end">RV-501 · SM-501</text>
      <rect x="590" y="422" width="42" height="34" rx="4" fill="#16223a" stroke="#8fa3c2" stroke-width="1.5"/>
      <rect x="598" y="430" width="10" height="20" fill="{mcol}" opacity=".85"/>
      <rect x="614" y="430" width="10" height="20" fill="{mcol}" opacity=".85"/>
      <text class="sub" x="611" y="468" text-anchor="middle">Ensaque / despacho</text>
      <!-- etiquetas de corrientes -->
      <text class="stream" x="262" y="138" fill="{C_FRESH}">① {F:.0f} t/h</text>
      <text class="stream" x="288" y="208" text-anchor="end" fill="{C_MIX}">② {S:.0f} t/h</text>
      <text class="stream" x="446" y="166" text-anchor="middle" fill="{C_OVER}">④ {R:.0f} t/h · CL {(r['cl'] if run else 0):.0f} %</text>
      <text class="stream" x="470" y="468" text-anchor="middle" fill="{C_DISCH}">③ {S:.0f} t/h</text>
      <text class="stream" x="722" y="430" fill="{C_PROD}">⑤ {F:.0f} t/h</text>
    </g>
    <text class="name" x="252" y="236">TK-101 · Tolva</text>
    <text class="sub" x="252" y="248">de alimentación</text>
    """

    # ===================== INSTRUMENTOS Y PANELES =====================
    bub = ""
    for i in inst:
        if i["pos"] is None:
            continue
        lines = [i["fmt"].format(i["val"]) + " " + i["unit"]]
        if i["sub"]:
            lines.append(i["sub"])
        bub += bubble(i["pos"][0], i["pos"][1], i["tag"], i["level"], i["target"], lines)

    grid = ('<pattern id="g" width="30" height="30" patternUnits="userSpaceOnUse">'
            '<path d="M30 0H0V30" fill="none" stroke="#14213a" stroke-width="1"/></pattern>')
    vx, vy, vw, vh = VIEWS.get(view_name, VIEWS["Vista completa"])
    svg = f"""
    <svg id="plant" viewBox="{vx} {vy} {vw} {vh}" xmlns="http://www.w3.org/2000/svg" class="{'' if run else 'off'}">
      <defs>{grid}</defs>
      <rect width="{W}" height="{H}" fill="#0a1222"/>
      <rect width="{W}" height="{H}" fill="url(#g)"/>
      {ducts}
      {upstream}
      {main}
      {dust}
      {bub}
      {panels(p, r, mat, lv)}
    </svg>
    """
    js = (JS.replace("__W__", str(W)).replace("__H__", str(H)).replace("__X__", str(vx))
          .replace("__Y__", str(vy)).replace("__VW__", str(vw)).replace("__VH__", str(vh)))
    return (f"<style>{CSS}</style><div class='wrap'><div class='tb'><button id='zi'>+</button>"
            f"<button id='zo'>−</button><button id='zr'>⟲</button></div>{svg}"
            f"<div class='hint'>Ctrl + rueda · arrastrar · doble clic</div></div><script>{js}</script>")


# ----------------------------------------------------------------------------
# 5. GRÁFICOS
# ----------------------------------------------------------------------------
def psd_figure(p, r):
    x = np.logspace(1.5, 4.5, 200)
    n_feed = 1.3
    x63_feed = r["F80"] / (math.log(5.0) ** (1 / n_feed))
    feed = rr_cdf(x, x63_feed, n_feed) * 100
    disch = rr_cdf(x, r["x63"], r["n"]) * 100
    prod = np.where(x <= r["a"], rr_cdf(x, r["x63"], r["n"]) / max(r["f"], 1e-9) * 100, 100.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=feed, name="Alimentación fresca", line=dict(color=C_FRESH, width=3)))
    fig.add_trace(go.Scatter(x=x, y=disch, name="Descarga del molino", line=dict(color=C_DISCH, width=3)))
    fig.add_trace(go.Scatter(x=x, y=prod, name="Producto (undersize)", line=dict(color=C_PROD, width=3)))
    fig.add_vline(x=r["a"], line=dict(color="#ff5c5c", dash="dash"),
                  annotation_text=f"Abertura {p['aperture_mm']:.2f} mm", annotation_position="top left")
    fig.add_vline(x=p["target_mm"] * 1000, line=dict(color="#6fb4ff", dash="dot"),
                  annotation_text="Objetivo P80", annotation_position="bottom right")
    fig.add_hline(y=80, line=dict(color="#7f93b3", dash="dot"), annotation_text="80 %")
    fig.update_layout(template="plotly_dark", height=430, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(type="log", title="Tamaño de partícula (µm)"),
                      yaxis=dict(title="% pasante acumulado", range=[0, 102]),
                      legend=dict(orientation="h", y=1.1))
    return fig


def sweep_figures(p):
    def build(xs, key, title, xlab):
        res = [simulate({**p, key: v}) for v in xs]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=[q["p80_prod"] for q in res], name="P80 producto (µm)",
                                 line=dict(color=C_PROD, width=3)))
        fig.add_trace(go.Scatter(x=xs, y=[q["cl"] for q in res], name="Carga circulante (%)",
                                 line=dict(color=C_OVER, width=3), yaxis="y2"))
        fig.add_vline(x=p[key], line=dict(color="#cfdcf2", dash="dot"))
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10),
                          title=title, xaxis=dict(title=xlab), yaxis=dict(title="P80 (µm)"),
                          yaxis2=dict(title="Carga circulante (%)", overlaying="y", side="right"),
                          legend=dict(orientation="h", y=1.15))
        return fig
    return (build(np.linspace(0.15, 2.0, 30), "aperture_mm", "Efecto de la abertura de la zaranda", "Abertura (mm)"),
            build(np.linspace(20, 300, 30), "feed_tph", "Efecto de la alimentación fresca", "Alimentación (t/h)"))


def split_figure(rows):
    names = [x["name"] for x in rows]
    cols = [px_col(i) for i in range(len(rows))]
    fig = go.Figure()
    for i, x in enumerate(rows):
        fig.add_trace(go.Bar(name=x["name"], x=["Alimentación fresca", "Carga circulante (oversize)"],
                             y=[x["w"] * 100, x["share_R"] * 100], marker_color=cols[i]))
    fig.update_layout(template="plotly_dark", barmode="stack", height=380, margin=dict(l=10, r=10, t=40, b=10),
                      title="Composición: alimentación vs. carga circulante", yaxis=dict(title="% másico"),
                      legend=dict(orientation="h", y=1.15))
    return fig


def px_col(i):
    return ["#f5b942", "#6fb4ff", "#3ddc97", "#ff7a59", "#c792ea"][i % 5]


def cost_figure(r):
    fig = go.Figure(go.Bar(x=["Energía", "Acero (bolas + forros)"], y=[r["cost_energy"], r["cost_steel"]],
                           marker_color=["#f5b942", "#6fb4ff"]))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=40, b=10),
                      title="Costo operativo por tonelada (USD/t)", yaxis=dict(title="USD/t"))
    return fig


# ----------------------------------------------------------------------------
# 6. INTERFAZ STREAMLIT
# ----------------------------------------------------------------------------
def apply_preset():
    base = {**OP_DEFAULTS, **PRESETS[st.session_state["preset"]]}
    for k, v in base.items():
        st.session_state[k] = v


def property_table(m):
    return pd.DataFrame([
        ["Índice de trabajo Wi (Bond)", f"{m['wi']:.2f}", "kWh/t"],
        ["Densidad real", f"{m['rho']:.2f}", "t/m³"],
        ["Densidad aparente", f"{m['bulk']:.2f}", "t/m³"],
        ["Dureza Mohs", f"{m['mohs']:.1f}", "-"],
        ["Índice de abrasión Ai", f"{m['ai']:.3f}", "-"],
        ["Humedad de alimentación", f"{m['moist']:.1f}", "%"],
        ["Dispersión granulométrica n", f"{m['n']:.2f}", "-"],
    ], columns=["Propiedad", "Valor", "Unidad"])


def main():
    st.set_page_config(page_title="Planta Virtual · Molienda y Tamizaje", page_icon="🏭", layout="wide")
    for k, v in {**OP_DEFAULTS, **MAT_DEFAULTS}.items():
        st.session_state.setdefault(k, list(v) if isinstance(v, list) else v)

    st.markdown("## 🏭 Planta Virtual · Molienda y Tamizaje")
    st.caption("Recepción → chancado → tolva → molino de bolas en circuito cerrado → zaranda → silo, "
               "con colector de polvo. Elige el producto, mueve los controles y observa el proceso.")

    with st.sidebar:
        st.header("🎛️ Panel de control")
        st.toggle("Planta en marcha", key="running")
        st.selectbox("Escenario rápido", list(PRESETS.keys()), key="preset", on_change=apply_preset)

        st.subheader("1 · Producto")
        st.radio("Origen de las propiedades", ["Biblioteca", "Personalizado", "Mezcla"],
                 key="mat_mode", horizontal=True)
        mode = st.session_state["mat_mode"]
        with st.expander("Biblioteca de materiales", expanded=(mode == "Biblioteca")):
            st.selectbox("Material", list(MATERIALS.keys()), key="mat_lib")
        with st.expander("Material personalizado", expanded=(mode == "Personalizado")):
            st.text_input("Nombre", key="cust_name")
            st.number_input("Wi de Bond (kWh/t)", 3.0, 30.0, step=0.1, key="cust_wi")
            st.number_input("Densidad real (t/m³)", 0.8, 8.0, step=0.05, key="cust_rho")
            st.number_input("Densidad aparente (t/m³)", 0.4, 5.0, step=0.05, key="cust_bulk")
            st.number_input("Dureza Mohs", 1.0, 10.0, step=0.5, key="cust_mohs")
            st.number_input("Abrasión Ai", 0.0, 1.5, step=0.01, key="cust_ai")
            st.number_input("Humedad (%)", 0.0, 25.0, step=0.5, key="cust_moist")
            st.number_input("Dispersión n (Rosin-Rammler)", 0.6, 1.5, step=0.05, key="cust_n")
            st.color_picker("Color en la planta", key="cust_color")
        with st.expander("Mezcla de materiales", expanded=(mode == "Mezcla")):
            pool = list(MATERIALS.keys()) + [CUSTOM_LABEL]
            st.multiselect("Componentes (máx. 4)", pool, key="mix_sel", max_selections=4)
            for n in st.session_state["mix_sel"]:
                st.session_state.setdefault(f"mix_{n}", 50)
                st.slider(f"{n} (partes)", 0, 100, key=f"mix_{n}")
            st.caption("Las partes se normalizan a 100 % en masa. "
                       "«★ Personalizado» usa los datos de la pestaña anterior.")
        st.slider("Ajuste fino del Wi (%) por ensayo", -30, 30, key="wi_adj")

        st.subheader("2 · Alimentación")
        st.slider("Alimentación fresca (t/h)", 20.0, 300.0, step=5.0, key="feed_tph")
        st.slider("F80 de alimentación (mm)", 3.0, 40.0, step=0.5, key="f80_mm")
        st.slider("Nivel de tolva (%)", 0, 100, key="hopper_level")
        st.subheader("3 · Molino de bolas")
        st.slider("Diámetro (m)", 1.5, 5.0, step=0.1, key="D")
        st.slider("Longitud (m)", 2.0, 7.0, step=0.1, key="L")
        st.slider("Velocidad (% velocidad crítica)", 55, 90, key="nc")
        st.slider("Llenado de bolas J (%)", 10, 45, key="J")
        st.subheader("4 · Zaranda y producto")
        st.slider("Abertura de malla (mm)", 0.15, 2.0, step=0.05, key="aperture_mm")
        st.slider("Eficiencia base (%)", 40, 98, key="eff0")
        st.slider("Área de tamizado (m²)", 5.0, 60.0, step=1.0, key="area")
        st.slider("Objetivo P80 del producto (mm)", 0.1, 2.0, step=0.05, key="target_mm")
        st.slider("Nivel del silo (%)", 0, 100, key="silo_level")
        with st.expander("Costos"):
            st.number_input("Energía (USD/kWh)", 0.01, 1.0, step=0.01, key="price_kwh")
            st.number_input("Acero de bolas y forros (USD/kg)", 0.1, 10.0, step=0.1, key="price_steel")

    p, mat = build_params(st.session_state)
    r = simulate(p)
    run = p["running"]

    cols = st.columns(6)
    cols[0].metric("Producción", f"{r['F'] if run else 0:.0f} t/h")
    cols[1].metric("P80 producto", f"{r['p80_prod']:.0f} µm",
                   delta=f"{r['p80_prod'] - p['target_mm'] * 1000:+.0f} µm vs objetivo", delta_color="inverse")
    cols[2].metric("Carga circulante", f"{r['cl'] if run else 0:.0f} %")
    cols[3].metric("Potencia molino", f"{r['kw'] if run else 0:.0f} kW")
    cols[4].metric("Energía específica", f"{r['E_fresh']:.1f} kWh/t")
    cols[5].metric("Costo operativo", f"{r['cost_total']:.2f} USD/t")

    st.radio("Vista del diagrama", list(VIEWS.keys()), key="view", horizontal=True)
    components.html(plant_html(p, r, mat, st.session_state["view"]), height=640, scrolling=False)

    for kind, msg in alarms(p, r, mat):
        getattr(st, kind)(msg)

    t1, t2, t3, t4, t5, t6 = st.tabs(["📈 Granulometría", "🔬 Sensibilidad", "🧪 Material y mezcla",
                                      "📋 Balance de masa", "📟 Instrumentación", "🛠️ Desgaste y costos"])
    with t1:
        st.plotly_chart(psd_figure(p, r), width="stretch")
        st.caption("Curvas Rosin-Rammler. La línea roja es la abertura de la zaranda: lo que la supera regresa al molino.")
    with t2:
        f1, f2 = sweep_figures(p)
        a, b = st.columns(2)
        a.plotly_chart(f1, width="stretch")
        b.plotly_chart(f2, width="stretch")
    with t3:
        st.markdown(f"**{mat['name']}** · {mat['uso']}")
        st.dataframe(property_table({**mat, "wi": p["wi"]}), width="stretch", hide_index=True)
        st.caption("Wi incluye el ajuste fino del panel lateral. En mezclas: Wi, Mohs, Ai, humedad y n se promedian "
                   "por masa; las densidades, por volumen. Valores de referencia, no sustituyen ensayos.")
        if len(mat["components"]) > 1:
            rows = component_split(p, r, mat)
            df = pd.DataFrame([{
                "Componente": x["name"], "% en alimentación": round(x["w"] * 100, 1), "Wi (kWh/t)": round(x["wi"], 1),
                "P80 en descarga (µm)": round(x["x80"]), "% < abertura": round(x["f"] * 100, 1),
                "Carga circ. propia (%)": round(x["cl"]), "% en carga circulante": round(x["share_R"] * 100, 1),
                "Enriquecimiento (×)": round(x["enrich"], 2)} for x in rows])
            st.markdown("**Molienda preferencial (indicativa):** el componente más duro se acumula en la carga circulante.")
            st.dataframe(df, width="stretch", hide_index=True)
            st.plotly_chart(split_figure(rows), width="stretch")
        else:
            st.info("Selecciona el modo «Mezcla» para ver cómo se reparte cada componente en la carga circulante.")
    with t4:
        df = pd.DataFrame([
            ["① Alimentación fresca", r["F"], r["F80"] / 1000, "Tolva → alimentador"],
            ["② Alimentación al molino", r["S"], r["f80_eff"] / 1000, "Fresco + retorno"],
            ["③ Descarga del molino", r["S"], r["x80"] / 1000, "Hacia el elevador"],
            ["④ Oversize (retorno)", r["R"], r["x_r"] / 1000, "Rechazo de la zaranda"],
            ["⑤ Producto (undersize)", r["F"], r["p80_prod"] / 1000, "Hacia el silo"],
        ], columns=["Corriente", "Flujo (t/h)", "P80 / F80 (mm)", "Descripción"])
        if not run:
            df["Flujo (t/h)"] = 0.0
        df["Flujo (t/h)"] = df["Flujo (t/h)"].round(1)
        df["P80 / F80 (mm)"] = df["P80 / F80 (mm)"].round(3)
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(f"Base seca de la alimentación: {r['dry_tph']:.1f} t/h (humedad {p['moist']:.1f} %). "
                   "Modelo didáctico: no sustituye un diseño real.")
    with t5:
        rows = []
        for i in build_instruments(p, r, mat):
            rows.append({"Tag": i["tag"], "Descripción": i["desc"], "Valor": i["fmt"].format(i["val"]),
                         "Unidad": i["unit"], "Estado": LEVEL_TXT[i["level"]]})
        rows += [
            {"Tag": "MS-101", "Descripción": "Imán de banda (hierro de tramp)", "Valor": "activo", "Unidad": "-", "Estado": "Normal"},
            {"Tag": "M-201", "Descripción": "Potencia del motor del molino", "Valor": f"{r['kw'] if run else 0:.0f}", "Unidad": "kW", "Estado": "Normal"},
            {"Tag": "FN-601", "Descripción": "Potencia del ventilador", "Valor": f"{r['fan_kw'] if run else 0:.1f}", "Unidad": "kW", "Estado": "Normal"},
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with t6:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Consumo de bolas", f"{r['ball_kgph'] if run else 0:.1f} kg/h")
        c2.metric("Desgaste de forros", f"{r['liner_kgph'] if run else 0:.2f} kg/h")
        c3.metric("Costo de energía", f"{r['cost_energy']:.2f} USD/t")
        c4.metric("Costo de acero", f"{r['cost_steel']:.2f} USD/t")
        d1, d2, d3 = st.columns(3)
        d1.metric("Caudal de aire del colector", f"{r['q_air'] if run else 0:,.0f} m³/h")
        d2.metric("Área de mangas requerida", f"{r['cloth']:.0f} m²")
        d3.metric("Pérdida de carga del filtro", f"{r['dp'] if run else 0:.0f} Pa")
        st.plotly_chart(cost_figure(r), width="stretch")
        st.caption("Desgaste con las ecuaciones de Bond (referencia de molienda húmeda; en seco suele ser menor). "
                   "El dimensionamiento del colector es indicativo (relación aire/tela 1.1 m/min).")


if __name__ == "__main__":
    main()
