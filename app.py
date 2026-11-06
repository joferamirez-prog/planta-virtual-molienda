"""
Planta Virtual de Molienda y Tamizaje  ·  versión simple
=========================================================
Circuito: tolva -> alimentador + imán -> molino de bolas (circuito cerrado)
-> elevador -> zaranda -> silo de producto.

Modelo didáctico simplificado (Bond + Rosin-Rammler + balance de masa).
Las propiedades de los materiales son valores típicos de referencia.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------------------------------------------
# 1. MATERIALES Y PARÁMETROS
# ----------------------------------------------------------------------------
# wi: índice de trabajo de Bond (kWh/t) · rho: densidad real (t/m³)
# bulk: densidad aparente (t/m³) · ai: índice de abrasión de Bond
# moist: humedad de alimentación (%) · n: dispersión Rosin-Rammler
MATERIALS = {
    "Caliza": dict(wi=11.6, rho=2.70, bulk=1.50, ai=0.020, moist=2.0, n=0.95, color="#e3dcc8"),
    "Yeso": dict(wi=6.7, rho=2.30, bulk=1.20, ai=0.010, moist=4.0, n=1.00, color="#f1efe6"),
    "Clínker de cemento": dict(wi=13.5, rho=3.15, bulk=1.35, ai=0.070, moist=0.5, n=0.85, color="#7a6c5d"),
    "Arena de sílice (cuarzo)": dict(wi=12.8, rho=2.65, bulk=1.60, ai=0.750, moist=1.0, n=1.10, color="#e8c98f"),
    "Carbón mineral": dict(wi=11.4, rho=1.40, bulk=0.85, ai=0.010, moist=8.0, n=1.00, color="#2b2b30"),
}
CUSTOM_LABEL = "★ Personalizado"

DEFAULTS = dict(
    running=True,
    material="Caliza",
    feed_tph=120.0,
    f80_mm=12.0,
    D=3.0,
    L=4.5,
    nc=74,
    J=30,
    aperture_mm=0.60,
    eff0=88,
    area=25.0,
    target_mm=0.50,
    price_kwh=0.10,
    price_steel=1.40,
    # material personalizado
    cust_name="Mi material",
    cust_wi=13.0, cust_rho=2.70, cust_bulk=1.50, cust_ai=0.150,
    cust_moist=3.0, cust_n=0.95, cust_color="#c9b38a",
)
OP_KEYS = ["running", "feed_tph", "f80_mm", "D", "L", "nc", "J", "aperture_mm", "eff0", "area",
           "target_mm", "price_kwh", "price_steel"]

C_FRESH, C_MIX, C_DISCH, C_OVER, C_PROD = "#f5b942", "#ffb36b", "#d9a05b", "#ff7a59", "#3ddc97"


def resolve_material(ss):
    """Material de la biblioteca o el definido por el usuario."""
    if ss["material"] == CUSTOM_LABEL:
        return dict(name=(ss["cust_name"] or "Personalizado").strip(),
                    wi=float(ss["cust_wi"]), rho=float(ss["cust_rho"]), bulk=float(ss["cust_bulk"]),
                    ai=float(ss["cust_ai"]), moist=float(ss["cust_moist"]), n=float(ss["cust_n"]),
                    color=ss["cust_color"])
    return dict(name=ss["material"], **MATERIALS[ss["material"]])


def build_params(ss):
    p = {k: ss[k] for k in OP_KEYS}
    mat = resolve_material(ss)
    p.update(wi=mat["wi"], n_rr=mat["n"], rho=mat["rho"], bulk=mat["bulk"],
             ai=mat["ai"], moist=mat["moist"])
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
    s_max_mill = 14.0 * volume * (p["rho"] / 2.7)          # capacidad de referencia del molino
    screen_cap = 25.0 * math.sqrt(p["aperture_mm"]) * p["area"] * (p["bulk"] / 1.6)
    # la humedad por encima de 3 % empasta la malla y baja la eficiencia
    eff_base = (p["eff0"] / 100.0) * max(0.5, 1.0 - 0.035 * max(0.0, p["moist"] - 3.0))

    S = 2.0 * F          # alimentación total al molino (fresco + retorno)
    x_r = 2.0 * a
    eta, f, x80, x63, load, f80_eff = eff_base, 0.7, 800.0, 500.0, 0.5, F80
    for _ in range(300):
        E = kw / S
        w = min(F / S, 1.0)
        inv_feed = w / math.sqrt(F80) + (1 - w) / math.sqrt(max(x_r, 1.0))
        f80_eff = 1.0 / inv_feed**2
        inv_p = 1.0 / math.sqrt(f80_eff) + E / (10.0 * Wi)   # Bond
        x80 = 1.0 / inv_p**2
        x63 = x80 / (math.log(5.0) ** (1.0 / n))
        f = float(rr_cdf(a, x63, n))                          # fracción < abertura
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
    return dict(
        F=F, S=S, R=R, cl=R / F * 100.0, kw=kw, rpm=rpm, E_fresh=kw / F,
        x80=x80, x63=x63, f=f, eta=eta, load=load, util_mill=S / s_max_mill,
        p80_prod=p80_prod, f80_eff=f80_eff, x_r=x_r, n=n, a=a, F80=F80,
        unstable=(S >= 14.5 * F),
        dry_tph=F * (1 - p["moist"] / 100.0),
        ball_kgph=ball_kgph, liner_kgph=liner_kgph,
        cost_energy=cost_energy, cost_steel=cost_steel, cost_total=cost_energy + cost_steel,
    )


def alarms(p, r):
    if not p["running"]:
        return [("info", "Planta detenida.")]
    out = []
    if r["unstable"] or r["cl"] > 400:
        out.append(("error", f"Carga circulante excesiva ({r['cl']:.0f} %): el circuito está inestable. "
                             "Baja la alimentación, aumenta la abertura o limpia la zaranda."))
    elif r["cl"] > 250:
        out.append(("warning", f"Carga circulante alta ({r['cl']:.0f} %)."))
    if r["util_mill"] > 1.0:
        out.append(("error", "Molino sobrecargado: el flujo total supera su capacidad de referencia."))
    if r["load"] > 1.0:
        out.append(("warning", f"Zaranda sobrecargada ({r['load'] * 100:.0f} %): "
                               f"la eficiencia cae a {r['eta'] * 100:.0f} %."))
    if p["moist"] > 6:
        out.append(("warning", f"Humedad alta ({p['moist']:.1f} %): riesgo de pegado en tolva, banda y malla. "
                               "Considera secado previo."))
    elif p["moist"] > 3:
        out.append(("info", f"Humedad de {p['moist']:.1f} %: la eficiencia de la zaranda se reduce."))
    if p["ai"] > 0.5:
        out.append(("warning", f"Material muy abrasivo (Ai = {p['ai']:.2f}): desgaste acelerado de bolas y forros."))
    tgt = p["target_mm"] * 1000
    if r["p80_prod"] > tgt * 1.3:
        out.append(("error", f"P80 del producto ({r['p80_prod']:.0f} µm) fuera de especificación "
                             f"(objetivo {tgt:.0f} µm)."))
    elif r["p80_prod"] > tgt * 1.1:
        out.append(("warning", f"P80 del producto ({r['p80_prod']:.0f} µm) ligeramente sobre el objetivo "
                               f"({tgt:.0f} µm)."))
    if not out:
        out.append(("success", "Operación normal: todos los indicadores dentro de rango."))
    return out


# ----------------------------------------------------------------------------
# 3. DIAGRAMA (SVG ANIMADO)
# ----------------------------------------------------------------------------
CSS = """
html,body{margin:0;height:100%;background:#0a1222;font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden}
.wrap{position:relative;height:100%}
svg{width:100%;height:100%;display:block}
.pipe{fill:none;stroke:#1d2b42;stroke-width:11;stroke-linejoin:round;stroke-linecap:round}
.flow{fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:2 11;animation:dash linear infinite}
@keyframes dash{to{stroke-dashoffset:-39}}
.belt{fill:none;stroke:#8aa0c0;stroke-width:2;stroke-dasharray:7 7;animation:belt linear infinite}
@keyframes belt{to{stroke-dashoffset:-28}}
.bucket{fill:none;stroke:#5f7596;stroke-width:16;stroke-dasharray:9 17;animation:bucket linear infinite}
@keyframes bucket{to{stroke-dashoffset:-52}}
.spin{animation:spin linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.slosh{animation:slosh ease-in-out infinite alternate}
@keyframes slosh{from{transform:rotate(-5deg)}to{transform:rotate(5deg)}}
.vib{animation:vib linear infinite}
@keyframes vib{0%{transform:translate(0,0)}25%{transform:translate(1.6px,-2px)}50%{transform:translate(-1.2px,1.6px)}75%{transform:translate(1.2px,2px)}100%{transform:translate(0,0)}}
.off *{animation-play-state:paused !important}
.name{font:600 11.5px 'Segoe UI',sans-serif;fill:#cfdcf2}
.sub{font:10.5px 'Segoe UI',sans-serif;fill:#7f93b3}
.stream{font:600 11px 'Consolas','Courier New',monospace;fill:#e8eefb}
.loop{font:700 8.5px 'Consolas',monospace;fill:#cfdcf2}
"""


def dur(rate, ref, slow=6.0, fast=0.5):
    if rate <= 0:
        return slow
    return float(min(slow, max(fast, 2.2 * ref / rate)))


def flow(d, color, rate, ref, width=None):
    w = width if width else 2.6 + 3.4 * min(1.0, rate / (ref * 2.5))
    return (f'<path class="pipe" d="{d}"/>'
            f'<path class="flow" d="{d}" style="stroke:{color};stroke-width:{w:.1f};'
            f'animation-duration:{dur(rate, ref):.2f}s"/>')


def belt_svg(x, y, w, d_s, direction=1, h=14):
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
    return f"""
    <g class="spin" style="transform-origin:{cx}px {cy}px;animation-duration:{t_spin:.2f}s">
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


def plant_html(p, r, mat):
    run = p["running"]
    F = r["F"] if run else 0.0
    S = r["S"] if run else 0.0
    R = r["R"] if run else 0.0
    ref = max(p["feed_tph"], 50.0)
    mcol = mat["color"]

    hop_poly = "60,30 180,30 135,115 105,115"
    y_fill = 30 + (1 - 0.70) * 85                       # tolva al 70 % (fijo)
    sil_poly = "650,246 770,246 770,360 720,398 700,398 650,360"
    y_s = 246 + (1 - 0.50) * 152                        # silo al 50 % (fijo)
    t_belt = dur(F, ref, slow=9, fast=0.4)
    t_b3 = dur(S, ref, slow=9, fast=0.4)
    t_b4 = dur(R, ref, slow=9, fast=0.4)
    t_el = dur(S, ref, slow=9, fast=0.4)
    t_vib = 0.09 if run else 1

    flows = "".join([
        flow("M120 133 L120 146", C_FRESH, F, ref, 4),
        flow("M232 154 L300 154 L300 176", C_FRESH, F, ref),
        flow("M588 128 L588 176 L304 176", C_OVER, R, ref),
        flow("M300 176 L300 228", C_MIX, S, ref),
        flow("M340 398 L340 478 L860 478 L860 48 L830 64", C_DISCH, S, ref),
        flow("M710 112 L710 250", C_PROD, F, ref),
    ])

    main = f"""
    {flows}
    <!-- tolva TK-101 -->
    <clipPath id="clipH"><polygon points="{hop_poly}"/></clipPath>
    <polygon points="{hop_poly}" fill="#101b30" stroke="#8fa3c2" stroke-width="2.5"/>
    <rect x="50" y="{y_fill:.1f}" width="140" height="90" fill="{mcol}" opacity=".8" clip-path="url(#clipH)"/>
    <rect x="109" y="115" width="22" height="18" fill="#1d2b42" stroke="#8fa3c2" stroke-width="2"/>
    <line x1="68" y1="30" x2="78" y2="12" stroke="#8fa3c2" stroke-width="2"/>
    <line x1="172" y1="30" x2="162" y2="12" stroke="#8fa3c2" stroke-width="2"/>
    <text class="name" x="192" y="46">TK-101 · Tolva</text>
    <text class="sub" x="192" y="58">de alimentación</text>
    <!-- alimentador FE-101 + imán -->
    {belt_svg(60, 146, 205, t_belt * 0.9, 1, h=16)}
    <rect x="150" y="122" width="52" height="12" rx="6" fill="#16223a" stroke="#c0586a" stroke-width="2"/>
    <line x1="156" y1="128" x2="196" y2="128" class="belt" style="animation-duration:{t_belt:.2f}s;stroke:#c0586a"/>
    <text class="name" x="130" y="184" text-anchor="middle">FE-101 · Alimentador de banda</text>
    <text class="sub" x="130" y="196" text-anchor="middle">Imán de banda (retira hierro)</text>
    <!-- molino ML-201 y motor -->
    <line x1="180" y1="314" x2="242" y2="314" stroke="#8fa3c2" stroke-width="6"/>
    <rect x="100" y="288" width="80" height="52" rx="6" fill="#16223a" stroke="#8fa3c2" stroke-width="2"/>
    <text x="140" y="320" text-anchor="middle" class="name" style="font-size:17px">M</text>
    <text class="sub" x="140" y="357" text-anchor="middle">Motor principal</text>
    {mill_svg(340, 312, 88, p['J'] / 100.0, r['rpm'], run)}
    <text class="name" x="322" y="440" text-anchor="end">ML-201 · Molino de bolas</text>
    <text class="sub" x="322" y="453" text-anchor="end">{p['D']:.1f} × {p['L']:.1f} m · J = {p['J']} % · {p['nc']} % Nc</text>
    <!-- bandas -->
    {belt_svg(296, 170, 300, t_b4, -1, h=12)}
    <text class="sub" x="446" y="198" text-anchor="middle">Banda de retorno (oversize)</text>
    {belt_svg(330, 472, 545, t_b3, 1, h=12)}
    <text class="name" x="600" y="505" text-anchor="middle">Banda de descarga del molino</text>
    <!-- elevador -->
    <rect x="846" y="40" width="28" height="440" rx="4" fill="#101b30" stroke="#8fa3c2" stroke-width="2"/>
    <path class="bucket" d="M860 470 L860 50" style="animation-duration:{t_el:.2f}s"/>
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
    <text class="name" x="712" y="26" text-anchor="middle">SC-401 · Zaranda vibratoria</text>
    <!-- silo TK-501 -->
    <clipPath id="clipS"><polygon points="{sil_poly}"/></clipPath>
    <polygon points="{sil_poly}" fill="#101b30" stroke="#8fa3c2" stroke-width="2.5"/>
    <rect x="640" y="{y_s:.1f}" width="140" height="170" fill="{mcol}" opacity=".75" clip-path="url(#clipS)"/>
    <text class="name" x="642" y="318" text-anchor="end">TK-501</text>
    <text class="sub" x="642" y="331" text-anchor="end">Silo de producto</text>
    <!-- etiquetas de corrientes -->
    <text class="stream" x="262" y="138" fill="{C_FRESH}">① {F:.0f} t/h</text>
    <text class="stream" x="288" y="208" text-anchor="end" fill="{C_MIX}">② {S:.0f} t/h</text>
    <text class="stream" x="446" y="166" text-anchor="middle" fill="{C_OVER}">④ {R:.0f} t/h · CL {(r['cl'] if run else 0):.0f} %</text>
    <text class="stream" x="470" y="468" text-anchor="middle" fill="{C_DISCH}">③ {S:.0f} t/h</text>
    <text class="stream" x="722" y="214" fill="{C_PROD}">⑤ {F:.0f} t/h</text>
    """

    grid = ('<pattern id="g" width="30" height="30" patternUnits="userSpaceOnUse">'
            '<path d="M30 0H0V30" fill="none" stroke="#14213a" stroke-width="1"/></pattern>')
    svg = f"""
    <svg id="plant" viewBox="30 0 970 520" xmlns="http://www.w3.org/2000/svg" class="{'' if run else 'off'}">
      <defs>{grid}</defs>
      <rect x="-200" y="-200" width="1600" height="1000" fill="#0a1222"/>
      <rect x="-200" y="-200" width="1600" height="1000" fill="url(#g)"/>
      {main}
    </svg>
    """
    return f"<style>{CSS}</style><div class='wrap'>{svg}</div>"


# ----------------------------------------------------------------------------
# 4. GRÁFICOS
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


def cost_figure(r):
    fig = go.Figure(go.Bar(x=["Energía", "Acero (bolas + forros)"], y=[r["cost_energy"], r["cost_steel"]],
                           marker_color=["#f5b942", "#6fb4ff"]))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=40, b=10),
                      title="Costo operativo por tonelada (USD/t)", yaxis=dict(title="USD/t"))
    return fig


# ----------------------------------------------------------------------------
# 5. INTERFAZ STREAMLIT
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Planta Virtual · Molienda y Tamizaje", page_icon="🏭", layout="wide")
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

    st.markdown("## 🏭 Planta Virtual · Molienda y Tamizaje")
    st.caption("Tolva → molino de bolas en circuito cerrado → zaranda → silo. "
               "Elige el material, mueve los controles y observa el proceso.")

    with st.sidebar:
        st.header("🎛️ Panel de control")
        st.toggle("Planta en marcha", key="running")

        st.subheader("1 · Material")
        st.selectbox("Material", list(MATERIALS.keys()) + [CUSTOM_LABEL], key="material")
        with st.expander("Datos del material personalizado",
                         expanded=(st.session_state["material"] == CUSTOM_LABEL)):
            st.text_input("Nombre", key="cust_name")
            st.number_input("Wi de Bond (kWh/t)", 3.0, 30.0, step=0.1, key="cust_wi")
            st.number_input("Densidad real (t/m³)", 0.8, 8.0, step=0.05, key="cust_rho")
            st.number_input("Densidad aparente (t/m³)", 0.4, 5.0, step=0.05, key="cust_bulk")
            st.number_input("Abrasión Ai", 0.0, 1.5, step=0.01, key="cust_ai")
            st.number_input("Humedad (%)", 0.0, 25.0, step=0.5, key="cust_moist")
            st.number_input("Dispersión n (Rosin-Rammler)", 0.6, 1.5, step=0.05, key="cust_n")
            st.color_picker("Color en la planta", key="cust_color")
            st.caption("Se usan solo si eliges «★ Personalizado» arriba.")
        m = resolve_material(st.session_state)
        st.caption(f"Wi = {m['wi']:.1f} kWh/t · densidad {m['rho']:.2f} t/m³ · "
                   f"humedad {m['moist']:.1f} % · Ai = {m['ai']:.3f}")

        st.subheader("2 · Alimentación")
        st.slider("Alimentación fresca (t/h)", 20.0, 300.0, step=5.0, key="feed_tph")
        st.slider("F80 de alimentación (mm)", 3.0, 40.0, step=0.5, key="f80_mm")

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

    diagram = plant_html(p, r, mat)
    if hasattr(st, "iframe"):          # Streamlit reciente
        st.iframe(diagram, height=560)
    else:                              # versiones anteriores
        components.html(diagram, height=560, scrolling=False)

    for kind, msg in alarms(p, r):
        getattr(st, kind)(msg)

    t1, t2, t3, t4 = st.tabs(["📈 Granulometría", "📋 Balance de masa", "🔬 Sensibilidad",
                              "🛠️ Desgaste y costos"])
    with t1:
        st.plotly_chart(psd_figure(p, r), width="stretch")
        st.caption("Curvas Rosin-Rammler. La línea roja es la abertura de la zaranda: "
                   "lo que la supera regresa al molino.")
    with t2:
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
                   "Balance en estado estable: producto = alimentación fresca; "
                   "carga circulante = oversize ÷ alimentación fresca. Modelo didáctico.")
    with t3:
        f1, f2 = sweep_figures(p)
        a, b = st.columns(2)
        a.plotly_chart(f1, width="stretch")
        b.plotly_chart(f2, width="stretch")
    with t4:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Consumo de bolas", f"{r['ball_kgph'] if run else 0:.1f} kg/h")
        c2.metric("Desgaste de forros", f"{r['liner_kgph'] if run else 0:.2f} kg/h")
        c3.metric("Costo de energía", f"{r['cost_energy']:.2f} USD/t")
        c4.metric("Costo de acero", f"{r['cost_steel']:.2f} USD/t")
        st.plotly_chart(cost_figure(r), width="stretch")
        st.caption("Desgaste con las ecuaciones de Bond (referencia de molienda húmeda; en seco suele ser menor).")


if __name__ == "__main__":
    main()
