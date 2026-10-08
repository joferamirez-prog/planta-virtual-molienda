"""
Planta Virtual de Molienda y Tamizaje
Circuito cerrado: Tolva -> Alimentador -> Molino de bolas -> Elevador -> Zaranda
(oversize recircula al molino, undersize va al silo de producto)

Modelo didáctico simplificado (Bond + Rosin-Rammler + balance de masa).
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------------------------------------------
# 1. MODELO DE PROCESO
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    running=True,
    feed_tph=120.0,
    f80_mm=12.0,
    wi=15.0,
    D=3.0,
    L=4.5,
    nc=74,
    J=30,
    aperture_mm=0.60,
    eff0=88,
    area=25.0,
    n_rr=0.9,
    target_mm=0.50,
    hopper_level=70,
    silo_level=40,
)

PRESETS = {
    "Condición base": {},
    "Alimentación alta (160 t/h)": dict(feed_tph=160.0),
    "Mineral duro (Wi = 20)": dict(wi=20.0),
    "Malla fina (0.40 mm)": dict(aperture_mm=0.40, target_mm=0.30),
    "Zaranda sucia (eficiencia 60 %)": dict(eff0=60),
    "Molino con poca carga de bolas": dict(J=20),
}


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
    # Potencia absorbida (modelo empírico simplificado, kW)
    kw = 45.0 * D**2.5 * L * phi * J * (1 - 1.03 * J)
    volume = math.pi / 4 * D**2 * L
    s_max_mill = 14.0 * volume  # t/h que el molino puede manejar (referencia)
    screen_cap = 25.0 * math.sqrt(p["aperture_mm"]) * p["area"]  # t/h

    S = 2.0 * F
    x_r = 2.0 * a
    eta = p["eff0"] / 100.0
    f = 0.7
    x80 = 800.0
    x63 = 500.0
    load = 0.5
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
        eta = (p["eff0"] / 100.0) / (1.0 + 2.0 * max(0.0, load - 0.9))
        eta = max(eta, 0.05)
        f_eff = max(f * eta, 1e-3)
        S_new = min(max(F / f_eff, F), 15.0 * F)
        S = 0.6 * S + 0.4 * S_new
        x_r = x63 * (-math.log(max(0.2 * (1 - f), 1e-9))) ** (1.0 / n)

    R = S - F
    cl = R / F * 100.0
    p80_prod = x63 * (-math.log(1 - 0.8 * f)) ** (1.0 / n)
    util_mill = S / s_max_mill
    return dict(
        F=F, S=S, R=R, cl=cl, kw=kw, rpm=rpm, nc_rpm=nc_rpm, E_fresh=kw / F,
        E_total=kw / S, x80=x80, x63=x63, f=f, eta=eta, load=load,
        util_mill=util_mill, s_max_mill=s_max_mill, screen_cap=screen_cap,
        p80_prod=p80_prod, f80_eff=f80_eff, x_r=x_r, n=n, a=a, F80=F80,
        unstable=(S >= 14.5 * F),
    )


def alarms(p, r):
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
        out.append(("warning", f"Molino cerca de su límite ({r['util_mill']*100:.0f} % de la capacidad)."))
    if r["load"] > 1.0:
        out.append(("warning", f"Zaranda sobrecargada ({r['load']*100:.0f} %): la eficiencia cae a {r['eta']*100:.0f} %."))
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
# 2. DIAGRAMA INDUSTRIAL (SVG ANIMADO)
# ----------------------------------------------------------------------------
COL = {"ok": "#3ddc97", "warn": "#f5b942", "alarm": "#ff5c5c", "off": "#64748b"}
C_FRESH, C_MIX, C_DISCH, C_OVER, C_PROD = "#f5b942", "#ffb36b", "#d9a05b", "#ff7a59", "#3ddc97"

CSS = """
html,body{margin:0;background:transparent;font-family:'Segoe UI',system-ui,sans-serif}
svg{width:100%;height:auto;display:block}
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
.blink{animation:blink 1s steps(2,start) infinite}
@keyframes blink{50%{opacity:.35}}
.off *{animation-play-state:paused !important}
.tag{font:600 10px 'Consolas','Courier New',monospace;fill:#9fb3d1}
.name{font:600 11px 'Segoe UI',sans-serif;fill:#cfdcf2}
.sub{font:10px 'Segoe UI',sans-serif;fill:#7f93b3}
.val{font:700 11px 'Consolas','Courier New',monospace}
.stream{font:600 10.5px 'Consolas','Courier New',monospace;fill:#e8eefb}
.loop{font:700 8.5px 'Consolas',monospace;fill:#cfdcf2}
"""


def dur(rate, ref, slow=6.0, fast=0.5):
    """Duración de animación: más flujo => más rápido."""
    if rate <= 0:
        return slow
    return float(min(slow, max(fast, 2.2 * ref / rate)))


def flow(d, color, rate, ref, width=None):
    w = width if width else 2.6 + 3.4 * min(1.0, rate / (ref * 2.5))
    return (f'<path class="pipe" d="{d}"/>'
            f'<path class="flow" d="{d}" style="stroke:{color};stroke-width:{w:.1f};'
            f'animation-duration:{dur(rate, ref):.2f}s"/>')


def bubble(cx, cy, loop, num, level, target=None, lines=()):
    col = COL[level]
    s = ""
    if target:
        s += f'<line x1="{cx}" y1="{cy}" x2="{target[0]}" y2="{target[1]}" stroke="#6e809f" stroke-width="1" stroke-dasharray="3 3"/>'
    cls = ' class="blink"' if level == "alarm" else ""
    s += (f'<g{cls}><circle cx="{cx}" cy="{cy}" r="16" fill="#0e1729" stroke="{col}" stroke-width="2"/>'
          f'<line x1="{cx-16}" y1="{cy}" x2="{cx+16}" y2="{cy}" stroke="{col}" stroke-width="1" opacity=".7"/>'
          f'<text class="loop" x="{cx}" y="{cy-4}" text-anchor="middle">{loop}</text>'
          f'<text class="loop" x="{cx}" y="{cy+11}" text-anchor="middle">{num}</text></g>')
    y = cy + 32
    for i, t in enumerate(lines):
        if i == 0:
            s += f'<text class="val" x="{cx}" y="{y}" text-anchor="middle" fill="{col}">{t}</text>'
        else:
            s += f'<text class="sub" x="{cx}" y="{y}" text-anchor="middle">{t}</text>'
        y += 12
    return s


def chord_height(J, r):
    """Altura del segmento circular (llenado J en fracción de área)."""
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


def mill_svg(cx, cy, r, J, rpm, running, kw):
    rin = r - 14
    h = chord_height(J, rin)
    level_y = cy + rin - h
    rng = np.random.default_rng(11)
    balls = []
    tries = 0
    nballs = int(30 + J * 260)
    while len(balls) < nballs and tries < 6000:
        tries += 1
        x = cx + rng.uniform(-rin, rin)
        y = cy + rng.uniform(-rin, rin)
        if (x - cx) ** 2 + (y - cy) ** 2 < (rin - 6) ** 2 and y > level_y + 3:
            balls.append((x, y))
    ball_svg = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.6" fill="#8fa5c6" stroke="#2a3b57" stroke-width="1"/>' for x, y in balls)
    # polvo/mineral entre bolas (relleno)
    fill = (f'<path d="M {cx - rin} {cy} A {rin} {rin} 0 0 0 {cx + rin} {cy} Z" fill="#2b2a2a" opacity="0"/>')
    lifters = "".join(
        f'<line x1="{cx + 69*math.cos(math.radians(k*22.5)):.1f}" y1="{cy + 69*math.sin(math.radians(k*22.5)):.1f}" '
        f'x2="{cx + 80*math.cos(math.radians(k*22.5)):.1f}" y2="{cy + 80*math.sin(math.radians(k*22.5)):.1f}" '
        f'stroke="#9db1d1" stroke-width="5" stroke-linecap="round"/>' for k in range(16))
    t_spin = max(1.4, 60.0 / max(rpm, 1) * 0.55) if running else 4
    t_slosh = max(0.9, t_spin * 0.6)
    return f"""
    <circle cx="{cx}" cy="{cy}" r="{r+9}" fill="none" stroke="#1a2740" stroke-width="3"/>
    <circle cx="{cx}" cy="{cy}" r="{r}" fill="#101b30" stroke="#8fa3c2" stroke-width="6"/>
    <g class="spin" style="transform-origin:{cx}px {cy}px;animation-duration:{t_spin:.2f}s">
      <circle cx="{cx}" cy="{cy}" r="{r-9}" fill="none" stroke="#33476b" stroke-width="2" stroke-dasharray="6 5"/>
      {lifters}
    </g>
    <g transform="rotate(28 {cx} {cy})">
      <g class="slosh" style="transform-origin:{cx}px {cy}px;animation-duration:{t_slosh:.2f}s">
        {ball_svg}
      </g>
    </g>
    {fill}
    <circle cx="{cx}" cy="{cy}" r="5" fill="#cfdcf2"/>
    """


def plant_svg(p, r):
    run = p["running"]
    F = r["F"] if run else 0.0
    S = r["S"] if run else 0.0
    R = r["R"] if run else 0.0
    ref = max(p["feed_tph"], 50.0)

    # ---- niveles de alarma por instrumento
    def lvl(cond_alarm, cond_warn):
        if not run:
            return "off"
        return "alarm" if cond_alarm else ("warn" if cond_warn else "ok")

    tgt = p["target_mm"] * 1000
    lv_hopper = lvl(p["hopper_level"] < 15, p["hopper_level"] < 30)
    lv_feed = lvl(r["util_mill"] > 1.0, r["util_mill"] > 0.85)
    lv_speed = lvl(False, p["nc"] > 82 or p["nc"] < 65)
    lv_kw = lvl(r["util_mill"] > 1.0, r["util_mill"] > 0.85)
    lv_eff = lvl(r["eta"] < 0.55, r["eta"] < 0.7)
    lv_cl = lvl(r["cl"] > 400 or r["unstable"], r["cl"] > 250)
    lv_silo = lvl(p["silo_level"] > 95, p["silo_level"] > 88)
    lv_p80 = lvl(r["p80_prod"] > tgt * 1.3, r["p80_prod"] > tgt * 1.1)
    st_col = COL["ok"] if run else COL["off"]

    # ---- tolva con nivel
    hop_poly = "60,30 180,30 135,115 105,115"
    lvl_h = p["hopper_level"] / 100.0
    y_fill = 30 + (1 - lvl_h) * 85
    hopper = f"""
    <clipPath id="clipH"><polygon points="{hop_poly}"/></clipPath>
    <polygon points="{hop_poly}" fill="#101b30" stroke="#8fa3c2" stroke-width="2.5"/>
    <rect x="50" y="{y_fill:.1f}" width="140" height="90" fill="{C_FRESH}" opacity=".55" clip-path="url(#clipH)"/>
    <rect x="109" y="115" width="22" height="18" fill="#1d2b42" stroke="#8fa3c2" stroke-width="2"/>
    <text class="name" x="120" y="16" text-anchor="middle">TK-101 · Tolva de alimentación</text>
    """

    # ---- alimentador de banda
    t_belt = dur(F, ref, slow=9, fast=0.4)
    belt1 = f"""
    <rect x="60" y="146" width="205" height="16" rx="8" fill="#16223a" stroke="#8fa3c2" stroke-width="2"/>
    <line x1="68" y1="150" x2="257" y2="150" class="belt" style="animation-duration:{t_belt*0.9:.2f}s"/>
    <circle cx="68" cy="154" r="9" fill="#1d2b42" stroke="#8fa3c2" stroke-width="2"/>
    <circle cx="257" cy="154" r="9" fill="#1d2b42" stroke="#8fa3c2" stroke-width="2"/>
    <text class="name" x="130" y="182" text-anchor="middle">FE-101 · Alimentador de banda</text>
    """

    # ---- banda de descarga y retorno
    t_b3 = dur(S, ref, slow=9, fast=0.4)
    belt3 = f"""
    <rect x="330" y="472" width="545" height="12" rx="6" fill="#16223a" stroke="#8fa3c2" stroke-width="2"/>
    <line x1="338" y1="475" x2="868" y2="475" class="belt" style="animation-duration:{t_b3:.2f}s"/>
    <text class="name" x="600" y="505" text-anchor="middle">CV-301 · Banda de descarga del molino</text>
    """
    t_b4 = dur(R, ref, slow=9, fast=0.4)
    belt4 = f"""
    <rect x="296" y="170" width="300" height="12" rx="6" fill="#16223a" stroke="#8fa3c2" stroke-width="2"/>
    <line x1="590" y1="173" x2="304" y2="173" class="belt" style="animation-duration:{t_b4:.2f}s"/>
    <text class="sub" x="446" y="198" text-anchor="middle">CV-302 · Banda de retorno (oversize)</text>
    """

    # ---- elevador
    t_el = dur(S, ref, slow=9, fast=0.4)
    elev = f"""
    <rect x="846" y="40" width="28" height="440" rx="4" fill="#101b30" stroke="#8fa3c2" stroke-width="2"/>
    <path class="bucket" d="M860 470 L860 50" style="animation-duration:{t_el:.2f}s"/>
    <text class="name" x="893" y="262">EL-301</text>
    <text class="sub" x="893" y="275">Elevador de</text>
    <text class="sub" x="893" y="287">cangilones</text>
    """

    # ---- zaranda
    t_vib = 0.09 if run else 1
    screen = f"""
    <rect x="578" y="46" width="268" height="104" rx="6" fill="none" stroke="#3a4d6e" stroke-width="1.5" stroke-dasharray="5 4"/>
    <polygon points="610,138 818,138 738,172 690,172" fill="#101b30" stroke="#8fa3c2" stroke-width="2"/>
    <g transform="translate(590,120) rotate(-11.8)">
      <g class="vib" style="animation-duration:{t_vib}s">
        <rect x="0" y="0" width="250" height="13" rx="3" fill="#2a3f63" stroke="#9db1d1" stroke-width="2"/>
        <line x1="6" y1="6.5" x2="244" y2="6.5" stroke="#cfdcf2" stroke-width="2" stroke-dasharray="2 4"/>
      </g>
    </g>
    <path d="M610 146 l-4 6 l8 4 l-8 4 l8 4" fill="none" stroke="#7f93b3" stroke-width="1.5"/>
    <path d="M790 146 l-4 6 l8 4 l-8 4 l8 4" fill="none" stroke="#7f93b3" stroke-width="1.5"/>
    <text class="name" x="712" y="32" text-anchor="middle">SC-401 · Zaranda vibratoria</text>
    """

    # ---- silo de producto
    sil_poly = "650,246 770,246 770,360 720,398 700,398 650,360"
    lvl_s = p["silo_level"] / 100.0
    y_s = 246 + (1 - lvl_s) * 152
    silo = f"""
    <clipPath id="clipS"><polygon points="{sil_poly}"/></clipPath>
    <polygon points="{sil_poly}" fill="#101b30" stroke="#8fa3c2" stroke-width="2.5"/>
    <rect x="640" y="{y_s:.1f}" width="140" height="170" fill="{C_PROD}" opacity=".45" clip-path="url(#clipS)"/>
    <rect x="703" y="398" width="14" height="14" fill="#1d2b42" stroke="#8fa3c2" stroke-width="2"/>
    <text class="name" x="642" y="318" text-anchor="end">TK-501</text>
    <text class="sub" x="642" y="331" text-anchor="end">Silo de producto</text>
    """

    # ---- motor
    motor = f"""
    <rect x="105" y="288" width="85" height="52" rx="6" fill="#16223a" stroke="#8fa3c2" stroke-width="2"/>
    <text x="147" y="319" text-anchor="middle" class="name" style="font-size:16px">M</text>
    <line x1="190" y1="314" x2="252" y2="314" stroke="#8fa3c2" stroke-width="6"/>
    <text class="sub" x="147" y="357" text-anchor="middle">M-201 · Motor principal</text>
    """

    # ---- flujos (tuberías/chutes animados)
    flows = "".join([
        flow("M120 133 L120 146", C_FRESH, F, ref, width=4),
        flow("M262 154 L300 154 L300 176", C_FRESH, F, ref),
        flow("M588 128 L588 176 L304 176", C_OVER, R, ref),
        flow("M300 176 L300 228", C_MIX, S, ref),
        flow("M340 398 L340 478 L860 478 L860 48 L830 64", C_DISCH, S, ref),
        flow("M710 112 L710 250", C_PROD, F, ref),
        flow("M710 398 L710 442", C_PROD, F, ref, width=4),
    ])

    mill = mill_svg(340, 312, 88, p["J"] / 100.0, r["rpm"], run, r["kw"])
    mill_lbl = f"""
    <text class="name" x="322" y="432" text-anchor="end">ML-201 · Molino de bolas</text>
    <text class="sub" x="322" y="445" text-anchor="end">{p['D']:.1f} × {p['L']:.1f} m · J = {p['J']} %</text>
    """

    # ---- etiquetas de corrientes
    streams = f"""
    <text class="stream" x="262" y="136" fill="{C_FRESH}">① {F:.0f} t/h</text>
    <text class="stream" x="288" y="208" text-anchor="end" fill="{C_MIX}">② {S:.0f} t/h</text>
    <text class="stream" x="446" y="166" text-anchor="middle" fill="{C_OVER}">④ {R:.0f} t/h · CL {r['cl'] if run else 0:.0f} %</text>
    <text class="stream" x="470" y="468" text-anchor="middle" fill="{C_DISCH}">③ {S:.0f} t/h</text>
    <text class="stream" x="722" y="430" fill="{C_PROD}">⑤ {F:.0f} t/h</text>
    <text class="sub" x="722" y="442">A despacho / ensaque</text>
    """

    # ---- instrumentos
    inst = "".join([
        bubble(215, 72, "LIT", "101", lv_hopper, (170, 72),
               (f"{p['hopper_level']} %", "nivel tolva")),
        bubble(215, 206, "WIT", "102", lv_feed, (215, 162),
               (f"{F:.1f} t/h", "pesómetro")),
        bubble(147, 250, "SIC", "201", lv_speed, (147, 288),
               ()),
        bubble(480, 262, "SIC", "202", lv_speed, (421, 276),
               (f"{r['rpm'] if run else 0:.1f} rpm", f"{p['nc']} % Nc")),
        bubble(480, 346, "JIT", "201", lv_kw, (427, 336),
               (f"{r['kw'] if run else 0:.0f} kW", f"{r['E_fresh']:.1f} kWh/t")),
        bubble(395, 62, "XI", "401", "ok" if run else "off", (578, 88),
               (f"{p['aperture_mm']:.2f} mm", "abertura")),
        bubble(500, 62, "ET", "401", lv_eff, (578, 74),
               (f"{r['eta']*100 if run else 0:.0f} %", "eficiencia")),
        bubble(505, 120, "FIC", "402", lv_cl, (578, 120),
               (f"{r['cl'] if run else 0:.0f} %", "carga circ.")),
        bubble(808, 300, "LIT", "501", lv_silo, (770, 300),
               (f"{p['silo_level']} %",)),
        bubble(808, 384, "AIT", "501", lv_p80, (762, 372),
               (f"P80 {r['p80_prod']:.0f} µm",)),
    ])
    # SIC-201 duplicado en el motor: se elimina para no saturar (se muestra solo SIC-202)
    inst = inst.replace(bubble(147, 250, "SIC", "201", lv_speed, (147, 288), ()), "")

    # ---- panel lateral
    status_txt = "EN MARCHA" if run else "DETENIDA"
    panel = f"""
    <g transform="translate(935,36)">
      <rect x="0" y="0" width="152" height="150" rx="8" fill="#0e1729" stroke="#26385a"/>
      <circle cx="16" cy="18" r="6" fill="{st_col}" class="{'blink' if not run else ''}"/>
      <text class="name" x="30" y="22">{status_txt}</text>
      <text class="sub" x="12" y="46">Producto P80</text>
      <text class="val" x="12" y="62" fill="{COL[lv_p80]}" style="font-size:16px">{r['p80_prod']:.0f} µm</text>
      <text class="sub" x="12" y="84">Energía específica</text>
      <text class="val" x="12" y="100" fill="#cfdcf2" style="font-size:16px">{r['E_fresh']:.1f} kWh/t</text>
      <text class="sub" x="12" y="122">Carga circulante</text>
      <text class="val" x="12" y="138" fill="{COL[lv_cl]}" style="font-size:16px">{r['cl'] if run else 0:.0f} %</text>
    </g>
    <g transform="translate(935,205)">
      <rect x="0" y="0" width="152" height="112" rx="8" fill="#0e1729" stroke="#26385a"/>
      <text class="name" x="12" y="20">Corrientes</text>
      <rect x="12" y="30" width="14" height="4" fill="{C_FRESH}"/><text class="sub" x="34" y="35">① Alimentación fresca</text>
      <rect x="12" y="46" width="14" height="4" fill="{C_MIX}"/><text class="sub" x="34" y="51">② Alim. al molino</text>
      <rect x="12" y="62" width="14" height="4" fill="{C_DISCH}"/><text class="sub" x="34" y="67">③ Descarga molino</text>
      <rect x="12" y="78" width="14" height="4" fill="{C_OVER}"/><text class="sub" x="34" y="83">④ Oversize (retorno)</text>
      <rect x="12" y="94" width="14" height="4" fill="{C_PROD}"/><text class="sub" x="34" y="99">⑤ Producto (undersize)</text>
    </g>
    <g transform="translate(935,335)">
      <rect x="0" y="0" width="152" height="82" rx="8" fill="#0e1729" stroke="#26385a"/>
      <text class="name" x="12" y="20">Estado instrumentos</text>
      <circle cx="18" cy="36" r="5" fill="{COL['ok']}"/><text class="sub" x="30" y="40">Normal</text>
      <circle cx="18" cy="52" r="5" fill="{COL['warn']}"/><text class="sub" x="30" y="56">Advertencia</text>
      <circle cx="18" cy="68" r="5" fill="{COL['alarm']}"/><text class="sub" x="30" y="72">Alarma</text>
    </g>
    """

    grid = ('<pattern id="g" width="30" height="30" patternUnits="userSpaceOnUse">'
            '<path d="M30 0H0V30" fill="none" stroke="#14213a" stroke-width="1"/></pattern>')

    svg = f"""
    <svg viewBox="0 0 1100 520" xmlns="http://www.w3.org/2000/svg" class="{'' if run else 'off'}">
      <defs>{grid}</defs>
      <rect width="1100" height="520" rx="12" fill="#0a1222"/>
      <rect width="1100" height="520" rx="12" fill="url(#g)"/>
      {flows}
      {belt1}{belt4}{belt3}{elev}
      {hopper}{motor}{mill}{mill_lbl}{screen}{silo}
      {streams}
      {inst}
      {panel}
    </svg>
    """
    return f"<style>{CSS}</style>{svg}"


# ----------------------------------------------------------------------------
# 3. GRÁFICOS
# ----------------------------------------------------------------------------
def psd_figure(p, r):
    x = np.logspace(1.5, 4.5, 200)  # µm
    F80 = r["F80"]
    n_feed = 1.3
    x63_feed = F80 / (math.log(5.0) ** (1 / n_feed))
    feed = rr_cdf(x, x63_feed, n_feed) * 100
    disch = rr_cdf(x, r["x63"], r["n"]) * 100
    f = r["f"]
    prod = np.where(x <= r["a"], rr_cdf(x, r["x63"], r["n"]) / max(f, 1e-9) * 100, 100.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=feed, name="Alimentación fresca", line=dict(color=C_FRESH, width=3)))
    fig.add_trace(go.Scatter(x=x, y=disch, name="Descarga del molino", line=dict(color=C_DISCH, width=3)))
    fig.add_trace(go.Scatter(x=x, y=prod, name="Producto (undersize)", line=dict(color=C_PROD, width=3)))
    fig.add_vline(x=r["a"], line=dict(color="#ff5c5c", dash="dash"),
                  annotation_text=f"Abertura {p['aperture_mm']:.2f} mm", annotation_position="top left")
    fig.add_hline(y=80, line=dict(color="#7f93b3", dash="dot"), annotation_text="80 %")
    fig.update_layout(template="plotly_dark", height=430, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(type="log", title="Tamaño de partícula (µm)"),
                      yaxis=dict(title="% pasante acumulado", range=[0, 102]),
                      legend=dict(orientation="h", y=1.1))
    return fig


def sweep_figures(p):
    ap = np.linspace(0.15, 2.0, 30)
    res_a = [simulate({**p, "aperture_mm": a}) for a in ap]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ap, y=[q["p80_prod"] for q in res_a], name="P80 producto (µm)",
                              line=dict(color=C_PROD, width=3)))
    fig1.add_trace(go.Scatter(x=ap, y=[q["cl"] for q in res_a], name="Carga circulante (%)",
                              line=dict(color=C_OVER, width=3), yaxis="y2"))
    fig1.add_vline(x=p["aperture_mm"], line=dict(color="#cfdcf2", dash="dot"))
    fig1.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10),
                       title="Efecto de la abertura de la zaranda",
                       xaxis=dict(title="Abertura (mm)"), yaxis=dict(title="P80 (µm)"),
                       yaxis2=dict(title="Carga circulante (%)", overlaying="y", side="right"),
                       legend=dict(orientation="h", y=1.15))

    fe = np.linspace(20, 300, 30)
    res_f = [simulate({**p, "feed_tph": q}) for q in fe]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fe, y=[q["p80_prod"] for q in res_f], name="P80 producto (µm)",
                              line=dict(color=C_PROD, width=3)))
    fig2.add_trace(go.Scatter(x=fe, y=[q["cl"] for q in res_f], name="Carga circulante (%)",
                              line=dict(color=C_OVER, width=3), yaxis="y2"))
    fig2.add_vline(x=p["feed_tph"], line=dict(color="#cfdcf2", dash="dot"))
    fig2.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=40, b=10),
                       title="Efecto de la alimentación fresca",
                       xaxis=dict(title="Alimentación (t/h)"), yaxis=dict(title="P80 (µm)"),
                       yaxis2=dict(title="Carga circulante (%)", overlaying="y", side="right"),
                       legend=dict(orientation="h", y=1.15))
    return fig1, fig2


# ----------------------------------------------------------------------------
# 4. INTERFAZ STREAMLIT
# ----------------------------------------------------------------------------
def apply_preset():
    name = st.session_state["preset"]
    base = {**DEFAULTS, **PRESETS[name]}
    for k, v in base.items():
        st.session_state[k] = v


def main():
    st.set_page_config(page_title="Planta Virtual · Molienda y Tamizaje", page_icon="🏭", layout="wide")
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

    st.markdown("## 🏭 Planta Virtual · Molienda y Tamizaje")
    st.caption("Circuito cerrado: tolva → alimentador → molino de bolas → elevador → zaranda → silo "
               "(el oversize recircula al molino). Mueve los controles y observa el proceso.")

    with st.sidebar:
        st.header("🎛️ Panel de control")
        st.selectbox("Escenario rápido", list(PRESETS.keys()), key="preset", on_change=apply_preset)
        st.toggle("Planta en marcha", key="running")

        with st.expander("Alimentación", expanded=True):
            st.slider("Alimentación fresca (t/h)", 20.0, 300.0, step=5.0, key="feed_tph")
            st.slider("F80 de alimentación (mm)", 3.0, 40.0, step=0.5, key="f80_mm")
            st.slider("Índice de trabajo Wi (kWh/t)", 8.0, 24.0, step=0.5, key="wi")
            st.slider("Nivel de tolva (%)", 0, 100, key="hopper_level")
        with st.expander("Molino de bolas", expanded=True):
            st.slider("Diámetro (m)", 1.5, 5.0, step=0.1, key="D")
            st.slider("Longitud (m)", 2.0, 7.0, step=0.1, key="L")
            st.slider("Velocidad (% velocidad crítica)", 55, 90, key="nc")
            st.slider("Llenado de bolas J (%)", 10, 45, key="J")
        with st.expander("Zaranda", expanded=True):
            st.slider("Abertura de malla (mm)", 0.15, 2.0, step=0.05, key="aperture_mm")
            st.slider("Eficiencia base (%)", 40, 98, key="eff0")
            st.slider("Área de tamizado (m²)", 5.0, 60.0, step=1.0, key="area")
        with st.expander("Producto / avanzado"):
            st.slider("Objetivo P80 del producto (mm)", 0.1, 2.0, step=0.05, key="target_mm")
            st.slider("Nivel del silo (%)", 0, 100, key="silo_level")
            st.slider("Dispersión granulométrica n (Rosin-Rammler)", 0.6, 1.5, step=0.05, key="n_rr")

    p = {k: st.session_state[k] for k in DEFAULTS}
    r = simulate(p)
    run = p["running"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Producción", f"{r['F'] if run else 0:.0f} t/h")
    c2.metric("P80 producto", f"{r['p80_prod']:.0f} µm",
              delta=f"{r['p80_prod'] - p['target_mm']*1000:+.0f} µm vs objetivo", delta_color="inverse")
    c3.metric("Carga circulante", f"{r['cl'] if run else 0:.0f} %")
    c4.metric("Potencia molino", f"{r['kw'] if run else 0:.0f} kW")
    c5.metric("Energía específica", f"{r['E_fresh']:.1f} kWh/t")

    components.html(plant_svg(p, r), height=560, scrolling=False)

    for kind, msg in alarms(p, r):
        getattr(st, kind)(msg)

    tab1, tab2, tab3 = st.tabs(["📈 Granulometría", "🔬 Sensibilidad", "📋 Balance de masa"])
    with tab1:
        st.plotly_chart(psd_figure(p, r), width="stretch")
        st.caption("Curvas Rosin-Rammler. La línea roja es la abertura de la zaranda: "
                   "todo lo que supera ese tamaño regresa al molino.")
    with tab2:
        f1, f2 = sweep_figures(p)
        a, b = st.columns(2)
        a.plotly_chart(f1, width="stretch")
        b.plotly_chart(f2, width="stretch")
    with tab3:
        df = pd.DataFrame([
            ["① Alimentación fresca", r["F"], r["F80"] / 1000, "Tolva → alimentador"],
            ["② Alimentación al molino", r["S"], r["f80_eff"] / 1000, "Fresco + retorno"],
            ["③ Descarga del molino", r["S"], r["x80"] / 1000, "Hacia el elevador"],
            ["④ Oversize (retorno)", r["R"], r["x_r"] / 1000, "Rechazo de la zaranda"],
            ["⑤ Producto (undersize)", r["F"], r["p80_prod"] / 1000, "Hacia el silo"],
        ], columns=["Corriente", "Flujo (t/h)", "P80 / F80 (mm)", "Descripción"])
        if not run:
            df["Flujo (t/h)"] = 0.0
        st.dataframe(df.style.format({"Flujo (t/h)": "{:.1f}", "P80 / F80 (mm)": "{:.3f}"}),
                     width="stretch", hide_index=True)
        st.caption("Modelo didáctico: potencia del molino empírica, energía por Bond, distribución Rosin-Rammler "
                   "y eficiencia de zaranda con penalización por sobrecarga. No sustituye un diseño real.")


if __name__ == "__main__":
    main()
