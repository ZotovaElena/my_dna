"""Shared building blocks of the report: palette, HTML helpers, simple charts, page shell (CSS + JS)."""

import html
import json
import math
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRCH37_LEN = {  # chromosome lengths (bp)
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276, "5": 180915260, "6": 171115067,
    "7": 159138663, "8": 146364022, "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
    "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753, "17": 81195210, "18": 78077248,
    "19": 59128983, "20": 63025520, "21": 48129895, "22": 51304566, "X": 155270560,
}


# Categorical palette (light mode), fixed slot order, from scripts/report/color_palette.txt: the four hues
# (indigo, raspberry, green, chartreuse), then a lighter / darker step of each and slate. Same values as the CSS below.
SLOTS = ["#413AC5", "#C53A79", "#2D9E6B", "#BEC53A", "#8691E5", "#777C21", "#F2A3BF", "#595F61"]
LIGHT_SLOTS = {"#BEC53A", "#8691E5", "#F2A3BF"}  # take dark labels


SURFACE, INK, INK2, MUTED, GRID, TRACK = "#ffffff", "#191B1C", "#595F61", "#7B8285", "#E4E7EA", "#EDEFF0"

# Module accent colours (light, dark) and icons, used by the page header, tabs and overview cards.
MODULE_ACCENT = {"index": ("#413AC5", "#8691E5"), "genome": ("#413AC5", "#8691E5"),
                 "health": ("#C53A79", "#EC5E9A"), "drugs": ("#2D9E6B", "#3AC586"),
                 "traits": ("#777C21", "#BEC53A"), "ethnicity": ("#6169DC", "#B2BBEE"),
                 "ancient": ("#8F2856", "#F2A3BF"), "matches": ("#1F764F", "#49F1A5")}
ICONS = {  # 24×24 stroke icons
    "index": '<path d="M7 3c0 6 10 6 10 12s-10 6-10 6M17 3c0 6-10 6-10 12s10 6 10 6M8.5 7h7M8.5 17h7"/>',
    "genome": '<path d="M7 3c0 6 10 6 10 12s-10 6-10 6M17 3c0 6-10 6-10 12s10 6 10 6M8.5 7h7M8.5 17h7"/>',
    "health": '<path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z"/><path d="M8 12h2l1-2 2 4 1-2h2"/>',
    "drugs": '<rect x="3" y="8.5" width="18" height="7" rx="3.5" transform="rotate(-35 12 12)"/><path d="m10 8 4 6.6"/>',
    "traits": '<circle cx="12" cy="12" r="3"/><path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/>',
    "ethnicity": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.2 3 14.8 0 18M12 3c-3 3.2-3 14.8 0 18"/>',
    "ancient": '<path d="M5 21h14M6 18h12M7 18V9M10.3 18V9M13.7 18V9M17 18V9M4 9h16L12 4z"/>',
    "matches": '<circle cx="9" cy="12" r="5"/><circle cx="15" cy="12" r="5"/>',
}


def icon(module: str, cls: str = "ico") -> str:
    return (f'<svg class="{cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS.get(module, ICONS["index"])}</svg>')


def helix_svg(rungs: int = 14) -> str:
    """Decorative double helix for the page header; rungs cycle through the four palette hues."""
    w, h, amp, mid = 440, 150, 46, 75
    xs = np.linspace(0, w, 160)
    a = " ".join(f"{x:.1f},{mid + amp * math.sin(x / w * 2.6 * math.pi):.1f}" for x in xs)
    b = " ".join(f"{x:.1f},{mid - amp * math.sin(x / w * 2.6 * math.pi):.1f}" for x in xs)
    lines = []
    for i in range(rungs):
        x = (i + 0.5) * w / rungs
        y = amp * math.sin(x / w * 2.6 * math.pi)
        lines.append(f'<line class="h{i % 4}" x1="{x:.1f}" y1="{mid + y:.1f}" x2="{x:.1f}" y2="{mid - y:.1f}"/>'
                     f'<circle class="h{i % 4}" cx="{x:.1f}" cy="{mid + y:.1f}" r="4.5"/>'
                     f'<circle class="h{(i + 2) % 4}" cx="{x:.1f}" cy="{mid - y:.1f}" r="4.5"/>')
    return (f'<svg class="helix" viewBox="0 0 {w} {h}" aria-hidden="true"><polyline class="strand" points="{a}"/>'
            f'<polyline class="strand s-b" points="{b}"/>{"".join(lines)}</svg>')


SUPERPOP_NAMES = {"AFR": "African", "AMR": "Admixed American", "EAS": "East Asian", "EUR": "European",
                  "SAS": "South Asian"}


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def esc(x) -> str:
    return html.escape(str(x), quote=True).replace("\n", "&#10;")


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "–"
    return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%"


def tile(label: str, value: str, note: str = "") -> str:
    note = f'<div class="tile-note">{esc(note)}</div>' if note else ""
    return f'<div class="tile"><div class="tile-label">{esc(label)}</div><div class="tile-value">{esc(value)}</div>{note}</div>'


def legend(items: list[tuple[str, int]]) -> str:
    keys = "".join(f'<span class="key"><i class="sw s{slot}"></i>{esc(name)}</span>' for name, slot in items)
    return f'<div class="legend">{keys}</div>'


def seg_bar(parts: list[tuple[str, float, int, str]], cls: str = "", named: bool = False) -> str:
    """Tall segmented bar: one whole split into parts (name, share, slot, tooltip). Zero parts draw nothing.
    named=True prints the part's name in its block (when colours can't carry identity)."""
    blocks = "".join(
        f'<div class="seg s{slot}" style="flex:{share:.5f} 1 0" tabindex="0" data-tip="{esc(tip)}">'
        f'<span>{esc(name) + " " if named else ""}{pct(share)}</span></div>'
        for name, share, slot, tip in parts if share > 0.0005)
    label = "; ".join(f"{name} {pct(share)}" for name, share, _, _ in parts)
    return f'<div class="segbar {cls}" role="img" aria-label="{esc(label)}">{blocks}</div>'


def hbars(rows: list[tuple[str, float, str, str]], xmax: float | None = None, note: str = "") -> str:
    """Horizontal bars, one series: (label, value, value_text, tooltip)."""
    xmax = xmax or max(v for _, v, _, _ in rows) or 1
    body = "".join(
        f'<div class="hb-label">{esc(label)}</div>'
        f'<div class="hb-track"><div class="hb-bar s1" style="width:calc((100% - 56px) * {max(v / xmax, 0):.4f});flex:none" '
        f'tabindex="0" data-tip="{esc(tip)}"></div><span class="hb-val">{esc(text)}</span></div>'
        for label, v, text, tip in rows)
    note = f'<div class="chart-note">{esc(note)}</div>' if note else ""
    return f'<div class="hbars">{body}</div>{note}'


def columns(rows: list[tuple[str, float, str]], height: int = 160, y_label: str = "") -> str:
    """Vertical columns, one series: (x label, value, tooltip)."""
    ymax = max(v for _, v, _ in rows) or 1
    ticks = nice_ticks(0, ymax, 4)
    top = ticks[-1]
    grid = "".join(f'<div class="col-grid" style="bottom:{t / top * 100:.2f}%"><span>{fmt_num(t)}</span></div>'
                   for t in ticks)
    cols = "".join(
        f'<div class="col"><div class="col-bar s1" style="height:{v / top * 100:.2f}%" tabindex="0" '
        f'data-tip="{esc(tip)}"></div></div>' for _, v, tip in rows)
    labels = "".join(f'<div class="col-x">{esc(x)}</div>' for x, _, _ in rows)
    y_label = f'<div class="chart-note">{esc(y_label)}</div>' if y_label else ""
    return (f'<div class="cols-wrap"><div class="cols" style="height:{height}px">{grid}{cols}</div>'
            f'<div class="cols-x">{labels}</div></div>{y_label}')


def nice_ticks(lo: float, hi: float, n: int) -> list[float]:
    step = 10 ** math.floor(math.log10((hi - lo) / n or 1))
    for m in (1, 2, 2.5, 5, 10):
        if (hi - lo) / (m * step) <= n:
            step *= m
            break
    return list(np.arange(lo, math.ceil(hi / step) * step + step / 2, step))


def fmt_num(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v / 1e6:g}M"
    if abs(v) >= 1e4:
        return f"{v / 1e3:g}K"
    return f"{v:,.0f}" if v == int(v) else f"{v:g}"


def table(df: pd.DataFrame, cls: str = "", row_attrs=None) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        attrs = row_attrs(i) if row_attrs else ""
        rows.append(f"<tr {attrs}>" + "".join(f"<td>{esc(v)}</td>" for v in r) + "</tr>")
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def details(summary: str, content: str) -> str:
    return f"<details><summary>{esc(summary)}</summary>{content}</details>"


def section(sec_id: str, title: str, lede: str, body: str) -> str:
    lede = f'<p class="lede">{lede}</p>' if lede else ""
    return f'<section id="{sec_id}"><h2>{esc(title)}</h2>{lede}{body}</section>'


def card(title: str, body: str, sub: str = "") -> str:
    sub = f'<div class="card-sub">{sub}</div>' if sub else ""
    return f'<div class="card"><h3>{esc(title)}</h3>{sub}{body}</div>'


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def label_color(hex_color: str) -> str:
    return INK if hex_color in LIGHT_SLOTS else "white"


def fig_stacked(rows: list[tuple[str, list, list, list]], slots: dict[str, int] | None, title: str, path: Path):
    """Horizontal 100% bars: (row label, sources, weights, se)."""
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.6), facecolor=SURFACE)
    style_ax(ax)
    ax.grid(False)
    for i, (label, src, w, se) in enumerate(rows):
        left = 0
        for j, (s, x, e) in enumerate(zip(src, w, se)):
            color = SLOTS[(slots[s] if slots else j + 1) - 1]
            ax.barh(i, x - 0.004, left=left, height=0.62, color=color, edgecolor="none")
            text = f"{x * 100:.0f}%" if slots else f"{s} {x * 100:.0f}%"
            if x > (0.07 if slots else 0.18):
                ax.text(left + 0.01, i, text, va="center", fontsize=9, color=label_color(color))
            left += x
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    if slots:
        used = sorted({s for r in rows for s in r[1]}, key=slots.get)
        handles = [plt.Rectangle((0, 0), 1, 1, color=SLOTS[slots[s] - 1]) for s in used]
        ax.legend(handles, used, ncol=min(len(used), 6), frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
                  fontsize=9, labelcolor=INK, handlelength=1, handleheight=1)
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


CSS = """
:root{color-scheme:light;--page:#F5F6FC;--surface:#ffffff;--ink:#191B1C;--ink2:#595F61;--muted:#7B8285;
--grid:#E4E7EA;--axis:#C2C9CC;--track:#EDEFF0;--border:rgba(13,11,59,.10);--link:#413AC5;
--s1:#413AC5;--s2:#C53A79;--s3:#2D9E6B;--s4:#BEC53A;--s5:#8691E5;--s6:#777C21;--s7:#F2A3BF;--s8:#595F61;
--on4:#191B1C;--on-seg:#ffffff;--good:#3AC586;--good-ink:#1F764F;--crit:#C53A79;--pt:#C2C9CC;
--q1:#B2BBEE;--q2:#8691E5;--q3:#413AC5;--q4:#26217C;
--indigo:#413AC5;--rasp:#C53A79;--lime:#BEC53A;--mint:#3AC586;
--indigo-ink:#26217C;--rasp-ink:#8F2856;--lime-ink:#555815;--mint-ink:#1F764F;
--indigo-soft:#E0E5F8;--rasp-soft:#FADBE5;--lime-soft:#F3F6C9;--mint-soft:#E4FEEE;
--shadow:0 1px 2px rgba(13,11,59,.05),0 6px 20px rgba(13,11,59,.06)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0D0B3B;--surface:#17154A;
--ink:#EDEFF0;--ink2:#C2C9CC;--muted:#9BA3A8;--grid:rgba(178,187,238,.13);--axis:rgba(178,187,238,.28);
--track:rgba(178,187,238,.11);--border:rgba(178,187,238,.16);--link:#B2BBEE;
--s1:#8691E5;--s2:#EC5E9A;--s3:#3AC586;--s4:#BEC53A;--s5:#B2BBEE;--s6:#989E2D;--s7:#F2A3BF;--s8:#9BA3A8;
--on4:#0D0B3B;--on-seg:#0D0B3B;--good-ink:#49F1A5;--crit:#EC5E9A;--pt:#4A4F80;
--q1:#26217C;--q2:#413AC5;--q3:#8691E5;--q4:#B2BBEE;
--indigo:#8691E5;--rasp:#EC5E9A;--lime:#BEC53A;--mint:#3AC586;
--indigo-ink:#B2BBEE;--rasp-ink:#F2A3BF;--lime-ink:#E5EE48;--mint-ink:#49F1A5;
--indigo-soft:rgba(134,145,229,.16);--rasp-soft:rgba(236,94,154,.16);--lime-soft:rgba(190,197,58,.16);
--mint-soft:rgba(58,197,134,.16);--shadow:0 1px 2px rgba(0,0,0,.25),0 8px 24px rgba(0,0,0,.25)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0D0B3B;--surface:#17154A;
--ink:#EDEFF0;--ink2:#C2C9CC;--muted:#9BA3A8;--grid:rgba(178,187,238,.13);--axis:rgba(178,187,238,.28);
--track:rgba(178,187,238,.11);--border:rgba(178,187,238,.16);--link:#B2BBEE;
--s1:#8691E5;--s2:#EC5E9A;--s3:#3AC586;--s4:#BEC53A;--s5:#B2BBEE;--s6:#989E2D;--s7:#F2A3BF;--s8:#9BA3A8;
--on4:#0D0B3B;--on-seg:#0D0B3B;--good-ink:#49F1A5;--crit:#EC5E9A;--pt:#4A4F80;
--q1:#26217C;--q2:#413AC5;--q3:#8691E5;--q4:#B2BBEE;
--indigo:#8691E5;--rasp:#EC5E9A;--lime:#BEC53A;--mint:#3AC586;
--indigo-ink:#B2BBEE;--rasp-ink:#F2A3BF;--lime-ink:#E5EE48;--mint-ink:#49F1A5;
--indigo-soft:rgba(134,145,229,.16);--rasp-soft:rgba(236,94,154,.16);--lime-soft:rgba(190,197,58,.16);
--mint-soft:rgba(58,197,134,.16);--shadow:0 1px 2px rgba(0,0,0,.25),0 8px 24px rgba(0,0,0,.25)}
/* module accent: --acc-l / --acc-d set per page (body data-mod) */
body,.card.mod{--acc:var(--acc-l,#413AC5)}nav.tabs a{--dot:var(--dot-l)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) :is(body,.card.mod){--acc:var(--acc-d,#8691E5)}
:root:not([data-theme="light"]) nav.tabs a{--dot:var(--dot-d)}}
:root[data-theme="dark"] :is(body,.card.mod){--acc:var(--acc-d,#8691E5)}:root[data-theme="dark"] nav.tabs a{--dot:var(--dot-d)}
body,.card.mod{--acc-soft:color-mix(in srgb,var(--acc) 13%,var(--surface));--acc-ink:color-mix(in srgb,var(--acc) 82%,var(--ink))}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--link)}
::selection{background:color-mix(in srgb,var(--acc) 30%,transparent)}
/* header: tinted band with soft colour blobs and a helix */
header.hero{position:relative;overflow:hidden;border-bottom:1px solid var(--border);
background:radial-gradient(ellipse 50% 90% at 92% 0%,color-mix(in srgb,var(--acc) 26%,transparent),transparent 70%),
radial-gradient(ellipse 35% 80% at 70% 120%,color-mix(in srgb,var(--rasp) 14%,transparent),transparent 70%),
radial-gradient(ellipse 30% 70% at 100% 100%,color-mix(in srgb,var(--mint) 16%,transparent),transparent 70%),
linear-gradient(180deg,var(--acc-soft),var(--page))}
.hero-in{position:relative;z-index:1;max-width:1120px;margin:0 auto;padding:34px 16px 26px}
.eyebrow{display:inline-flex;align-items:center;gap:8px;font-size:12.5px;font-weight:700;letter-spacing:.08em;
text-transform:uppercase;color:var(--acc-ink);background:color-mix(in srgb,var(--acc) 12%,var(--surface));
border:1px solid color-mix(in srgb,var(--acc) 30%,transparent);border-radius:99px;padding:4px 12px 4px 8px;margin-bottom:12px}
.eyebrow .ico{width:18px;height:18px}
header.hero h1{margin:0;font-size:34px;line-height:1.15;font-weight:750;letter-spacing:-.02em;max-width:720px}
header.hero .sub{color:var(--ink2);margin-top:8px;max-width:640px}
.helix{position:absolute;right:max(16px,calc(50% - 560px));top:50%;transform:translateY(-50%) rotate(-8deg);width:400px;
height:auto;opacity:.9;pointer-events:none}
.helix .strand{fill:none;stroke:color-mix(in srgb,var(--acc) 55%,transparent);stroke-width:3;stroke-linecap:round}
.helix .strand.s-b{stroke:color-mix(in srgb,var(--rasp) 45%,transparent)}
.helix line{stroke-width:2.5;opacity:.55}.helix circle{stroke:var(--surface);stroke-width:1.5}
.helix .h0{stroke:var(--indigo);fill:var(--indigo)}.helix .h1{stroke:var(--rasp);fill:var(--rasp)}
.helix .h2{stroke:var(--mint);fill:var(--mint)}.helix .h3{stroke:var(--lime);fill:var(--lime)}
.helix circle.h0,.helix circle.h1,.helix circle.h2,.helix circle.h3{stroke:var(--surface)}
body.home header.hero .hero-in{padding:52px 16px 40px}body.home header.hero h1{font-size:44px}
body.home header.hero h1 em{font-style:normal;background:linear-gradient(90deg,var(--indigo),var(--rasp) 55%,var(--mint));
-webkit-background-clip:text;background-clip:text;color:transparent}
@media (max-width:860px){.helix{opacity:.22;width:340px;right:-60px}}
/* top nav: four-hue stripe, coloured dot per module */
nav.tabs{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--page) 88%,transparent);
backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border-bottom:1px solid var(--border)}
nav.tabs::before{content:"";display:block;height:4px;
background:linear-gradient(90deg,var(--indigo) 0 25%,var(--rasp) 25% 50%,var(--lime) 50% 75%,var(--mint) 75%)}
nav.tabs div{max-width:1120px;margin:0 auto;padding:0 16px;display:flex;gap:0 2px;align-items:center;overflow-x:auto;
scrollbar-width:none}
nav.tabs .brand{display:inline-flex;align-items:center;gap:6px;font-weight:800;margin-right:14px;white-space:nowrap;
letter-spacing:-.01em}
nav.tabs .brand .ico{width:20px;height:20px;color:var(--indigo)}
nav.tabs a{display:inline-flex;align-items:center;gap:6px;color:var(--ink2);text-decoration:none;padding:11px 10px 9px;
border-bottom:3px solid transparent;white-space:nowrap;font-size:14.5px}
nav.tabs a i{width:8px;height:8px;border-radius:50%;background:var(--dot);opacity:.85}
nav.tabs a:hover{color:var(--ink);background:color-mix(in srgb,var(--dot) 9%,transparent)}
nav.tabs a.on{color:var(--ink);border-bottom-color:var(--dot);font-weight:650}
.subnav div{max-width:1120px;margin:0 auto;padding:16px 16px 0;display:flex;gap:8px;flex-wrap:wrap;font-size:13.5px}
.subnav a{color:var(--ink2);text-decoration:none;padding:4px 12px;border-radius:99px;background:var(--surface);
border:1px solid var(--border);transition:background .15s,color .15s,border-color .15s}
.subnav a:hover{color:var(--acc-ink);background:var(--acc-soft);border-color:color-mix(in srgb,var(--acc) 35%,transparent)}
.subnav a.more{color:var(--surface);background:var(--acc);border-color:var(--acc);font-weight:600}
.subnav a.more:hover{color:var(--surface);background:var(--acc-ink)}
main{max-width:1120px;margin:0 auto;padding:0 16px 56px}
section{margin-top:44px;scroll-margin-top:64px}
h2{font-size:22px;font-weight:700;margin:0 0 6px;letter-spacing:-.01em;display:flex;align-items:center;gap:10px}
h2::before{content:"";flex:none;width:8px;height:22px;border-radius:4px;
background:linear-gradient(180deg,var(--acc),color-mix(in srgb,var(--acc) 35%,var(--surface)))}
h3{font-size:15px;font-weight:650;margin:0 0 2px}
.lede{color:var(--ink2);margin:0 0 16px;max-width:820px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px;margin-bottom:16px;min-width:0;
box-shadow:var(--shadow)}
.card-sub{color:var(--ink2);font-size:13.5px;margin-bottom:14px}
.card p{margin:0 0 10px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px}
.grid2>.card{margin-bottom:0}
section>.grid2{margin-bottom:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:20px}
.tile{--h:var(--indigo);--hi:var(--indigo-ink);position:relative;overflow:hidden;border:1px solid var(--border);border-radius:14px;
padding:16px 16px 14px;box-shadow:var(--shadow);background:linear-gradient(160deg,color-mix(in srgb,var(--h) 13%,var(--surface)),var(--surface) 70%)}
.tile::before{content:"";position:absolute;inset:0 0 auto 0;height:4px;background:var(--h)}
.tile:nth-child(4n+2){--h:var(--rasp);--hi:var(--rasp-ink)}.tile:nth-child(4n+3){--h:var(--mint);--hi:var(--mint-ink)}
.tile:nth-child(4n+4){--h:var(--lime);--hi:var(--lime-ink)}
.tile-label{color:var(--ink2);font-size:12.5px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
.tile-value{font-size:23px;line-height:1.25;font-weight:750;margin-top:6px;color:var(--hi);letter-spacing:-.01em}
.tile-note{color:var(--muted);font-size:12.5px}
.muted{color:var(--muted)}
.chart-note{color:var(--muted);font-size:12.5px;margin-top:8px}
.s1{background:var(--s1)}.s2{background:var(--s2)}.s3{background:var(--s3)}.s4{background:var(--s4)}
.s5{background:var(--s5)}.s6{background:var(--s6)}.s7{background:var(--s7)}.s8{background:var(--s8)}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;color:var(--ink2);margin:10px 0}
.key{display:inline-flex;align-items:center;gap:6px}
.sw{width:10px;height:10px;border-radius:3px;display:inline-block}
/* segmented bar */
.segbar{display:flex;gap:3px;height:30px;width:100%}
.seg{border-radius:6px;display:flex;align-items:center;padding-left:8px;min-width:2px;overflow:hidden;cursor:default}
.seg span{color:var(--on-seg);font-size:12.5px;font-weight:600;white-space:nowrap;font-variant-numeric:tabular-nums}
.seg.s4 span,.seg.s5 span,.seg.s7 span{color:var(--on4)}
.segbar.faded{opacity:.35}
.segbar-empty{height:28px;display:flex;align-items:center;color:var(--muted);font-size:13px;border:1px dashed var(--axis);
border-radius:4px;padding-left:8px}
/* horizontal bars */
.hbars{display:grid;grid-template-columns:fit-content(45%) minmax(0,1fr);gap:6px 12px;align-items:center;font-size:13.5px}
.hb-label{color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hb-track{display:flex;align-items:center;gap:8px;min-width:0}
.hb-bar{height:14px;border-radius:0 7px 7px 0;min-width:2px;background:linear-gradient(90deg,color-mix(in srgb,var(--s1) 55%,var(--surface)),var(--s1))!important}
.hb-val{color:var(--ink2);font-size:12.5px;font-variant-numeric:tabular-nums;white-space:nowrap}
/* columns */
.cols-wrap{padding-left:44px}
.cols{position:relative;display:flex;align-items:flex-end;gap:2px;border-bottom:1px solid var(--axis)}
.col{flex:1;height:100%;display:flex;align-items:flex-end;justify-content:center;z-index:1}
.col-bar{width:min(24px,80%);border-radius:6px 6px 0 0;min-height:1px;background:linear-gradient(0deg,color-mix(in srgb,var(--s1) 55%,var(--surface)),var(--s1))!important}
.col-grid{position:absolute;left:0;right:0;border-top:1px solid var(--grid)}
.col-grid span{position:absolute;right:calc(100% + 6px);top:-8px;font-size:11px;color:var(--muted);
font-variant-numeric:tabular-nums;white-space:nowrap}
.cols-x{display:flex;gap:2px;margin-top:4px}
.col-x{flex:1;text-align:center;font-size:11px;color:var(--muted);min-width:0;overflow:hidden}
/* ideogram */
.ideo{display:grid;grid-template-columns:28px 1fr;gap:5px 8px;align-items:center}
.ideo-label{color:var(--muted);font-size:12px;text-align:right;font-variant-numeric:tabular-nums}
.ideo-track{position:relative;height:12px;background:var(--track);border-radius:6px;overflow:hidden}
.roh-seg{position:absolute;top:0;bottom:0;border-radius:2px}
/* meter */
.meter-row{margin-bottom:18px}
.meter-head{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}
.meter-val{margin-left:auto;font-weight:600}
.meter{position:relative;height:8px;border-radius:4px;
background:linear-gradient(90deg,color-mix(in srgb,var(--mint) 45%,var(--track)),var(--track) 35% 65%,color-mix(in srgb,var(--rasp) 45%,var(--track)))}
.meter-mid{position:absolute;left:50%;top:-4px;bottom:-4px;width:1px;background:var(--axis)}
.meter-dot{position:absolute;top:50%;width:16px;height:16px;margin:-8px 0 0 -8px;border-radius:50%;background:var(--s1);
box-shadow:0 0 0 2px var(--surface)}
.meter-scale{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-top:4px}
/* status */
.status{padding:10px 14px;border-radius:10px;background:var(--indigo-soft);border-left:4px solid var(--indigo);font-size:14px}
.status.good{background:var(--mint-soft);border-left-color:var(--mint)}.status.crit{background:var(--rasp-soft);border-left-color:var(--rasp)}
.status.good b{color:var(--good-ink)}.status.crit b{color:var(--crit)}
/* qpAdm model rows */
.models{display:flex;flex-direction:column;gap:14px;margin-top:6px}
.mr{display:grid;grid-template-columns:70px 1fr minmax(140px,max-content);gap:6px 12px;align-items:center;
padding-bottom:14px;border-bottom:1px solid var(--grid)}
.mr:last-child{border-bottom:0}
.mr-name{grid-column:1/-1;font-weight:600;font-size:14px}
.mr-target{color:var(--ink2);font-size:13px}
.mr-p{font-size:13px;color:var(--ink2);font-variant-numeric:tabular-nums;white-space:nowrap}
.badge{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;font-weight:600;margin-right:4px}
.badge.fits{color:var(--good-ink);background:var(--mint-soft);box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--mint) 45%,transparent)}
.badge.rejected{color:var(--ink2);background:var(--track)}
.findings{margin:0;padding-left:20px}.findings li{margin-bottom:6px}.findings li::marker{color:var(--acc)}
/* p-value plot */
.pp{display:grid;grid-template-columns:minmax(120px,max-content) 1fr;gap:4px 12px;align-items:center;font-size:13px}
.pp-label{color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:280px}
.pp-track{position:relative;height:20px}
.pp-grid{position:absolute;top:0;bottom:0;width:1px;background:var(--grid)}
.pp-thr{position:absolute;top:-2px;bottom:-2px;border-left:1px dashed var(--ink2)}
.pp-dot{position:absolute;top:50%;width:11px;height:11px;margin:-5.5px 0 0 -5.5px;border-radius:50%;
box-shadow:0 0 0 2px var(--surface);z-index:1}
.pp-dot.open{background:var(--surface)!important;border:2px solid currentColor}
.pp-dot.s1.open{color:var(--s1)}.pp-dot.s2.open{color:var(--s2)}.pp-dot.s4.open{color:var(--s4)}
.pp-axis{position:relative;height:30px;font-size:11px;color:var(--muted)}
.pp-axis span{position:absolute;transform:translateX(-50%);top:2px;font-variant-numeric:tabular-nums}
.pp-axis .pp-thr-label{top:15px;color:var(--ink2)}
/* svg charts */
.svg-row{gap:12px}
.svg-fig{margin:0;min-width:0}
.svg-fig svg{width:100%;height:auto;display:block}
.svg-fig figcaption{color:var(--muted);font-size:12.5px;margin-top:4px}
.pt{fill:var(--pt);opacity:.8}.pt.hi{fill:var(--s1);opacity:.75}
.pt:hover{opacity:1;stroke:var(--ink);stroke-width:1}
.pt.c1{fill:var(--s1);opacity:.7}.pt.c2{fill:var(--s2);opacity:.7}.pt.c3{fill:var(--s3);opacity:.7}.pt.c4{fill:var(--s4);opacity:.7}.pt.c5{fill:var(--s5);opacity:.7}.pt.c6{fill:var(--s6);opacity:.7}.pt.c7{fill:var(--s7);opacity:.7}.pt.c8{fill:var(--s8);opacity:.7}
.me{fill:var(--ink);stroke:var(--surface);stroke-width:2}
.me-label{fill:var(--ink);font-size:13px;font-weight:600;paint-order:stroke;stroke:var(--surface);stroke-width:4px}
.grp{fill:var(--ink2);font-size:11px;font-weight:600;paint-order:stroke;stroke:var(--surface);stroke-width:3px;cursor:default}
.axis{stroke:var(--axis);stroke-width:1}.f4-cat{font-weight:600;color:var(--ink);margin-top:8px}.f4-c{position:absolute;top:50%;width:7px;height:7px;margin:-3.5px 0 0 -3.5px;border-radius:50%;opacity:.75}.f4-you{position:absolute;top:50%;width:13px;height:13px;margin:-6.5px 0 0 -6.5px;border-radius:50%;background:var(--ink);box-shadow:0 0 0 2px var(--surface);z-index:2}.sub-h{font-size:14px;margin:18px 0 6px}tr.hl td{background:var(--track)}.aps{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,340px),1fr));gap:12px}.card.ap{margin:0}.card.ap .fact b{font-weight:500;text-align:right;overflow-wrap:anywhere}.gridl{stroke:var(--grid);stroke-width:1}.ax{fill:var(--muted);font-size:11px}
/* tables */
.table-wrap{overflow-x:auto;max-height:560px;overflow-y:auto;border:1px solid var(--border);border-radius:12px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{position:sticky;top:0;background:color-mix(in srgb,var(--acc) 9%,var(--surface));text-align:left;font-weight:650;color:var(--acc-ink);
border-bottom:2px solid color-mix(in srgb,var(--acc) 40%,transparent);padding:7px 10px;white-space:nowrap}
td{padding:6px 10px;border-bottom:1px solid var(--grid);vertical-align:top;font-variant-numeric:tabular-nums}
table.dense td{padding:4px 10px;font-size:13px}
tbody tr:nth-child(even) td{background:color-mix(in srgb,var(--acc) 3%,transparent)}
tr:hover td{background:color-mix(in srgb,var(--acc) 9%,transparent)!important}
details{margin:4px 0 16px}summary{cursor:pointer;color:var(--acc-ink);font-weight:600;margin-bottom:10px}
summary::marker{color:var(--acc)}
.filters{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center;margin-bottom:12px;font-size:14px}
select{font:inherit;padding:3px 6px;border-radius:6px;border:1px solid var(--axis);background:var(--surface);color:var(--ink)}
/* qpAdm: glance rows, heatmap, forest plot */
.mr.glance{border-bottom:0;padding-bottom:0}
.hm-row-wrap{display:flex;flex-wrap:wrap;gap:24px 40px;overflow-x:auto;padding-bottom:4px}
.hm-wrap{min-width:0}
.hm-title{font-weight:600;font-size:13.5px;margin-bottom:6px}
.hm{display:grid;grid-auto-rows:26px;gap:2px;align-items:stretch}
.hm-col{height:auto;display:flex;align-items:flex-end;justify-content:center;padding-bottom:4px}
.hm-col span{writing-mode:vertical-rl;transform:rotate(180deg);font-size:11.5px;color:var(--ink2);white-space:nowrap}
.hm>.hm-col{height:96px}
.hm{grid-template-rows:96px}
.hm-row{font-size:12px;color:var(--ink2);display:flex;align-items:center;justify-content:flex-end;padding-right:8px;white-space:nowrap}
.hm-cell{border-radius:4px;cursor:default}
.hm-cell.diag{box-shadow:inset 0 0 0 1px var(--axis)}
.hm-empty{}
.q1{background:var(--q1)}.q2{background:var(--q2)}.q3{background:var(--q3)}.q4{background:var(--q4)}
.rej{background:var(--track)}
.inv{background:repeating-linear-gradient(45deg,var(--track) 0 3px,var(--grid) 3px 6px)}
.hm-cell.best,.hm-sw.best{outline:2px solid var(--ink);outline-offset:-2px}
.hm-sw{width:12px;height:12px;border-radius:3px}
.hm-cell:hover{outline:2px solid var(--ink2);outline-offset:-2px}
.fp-track{position:relative;height:26px}
.fp-ci{position:absolute;height:2px;margin-top:-1px;border-radius:1px;opacity:.55}
.fp-dot{position:absolute;width:10px;height:10px;margin:-5px 0 0 -5px;border-radius:50%;box-shadow:0 0 0 2px var(--surface);z-index:1}
.fp-mean{position:absolute;top:-2px;bottom:-2px;border-left:1px dashed var(--ink2)}
.fp .pp-label{max-width:340px}
pre.g25{white-space:pre;overflow-x:auto;font-size:12px;background:var(--indigo-soft);padding:10px;border-radius:8px;margin:6px 0}
.fit-h{margin:14px 0 4px;font-size:14px}
.tile-value{overflow-wrap:anywhere}
section>.tiles{margin-bottom:16px}
td.nw{white-space:nowrap}
/* overview: module cards */
.mods{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:16px}
.card.mod{position:relative;overflow:hidden;display:block;color:inherit;text-decoration:none;margin:0;padding-top:22px;
background:linear-gradient(170deg,var(--acc-soft),var(--surface) 55%);transition:transform .18s,box-shadow .18s,border-color .18s}
.card.mod::before{content:"";position:absolute;inset:0 0 auto 0;height:5px;background:linear-gradient(90deg,var(--acc),color-mix(in srgb,var(--acc) 30%,var(--surface)))}
.card.mod:hover{transform:translateY(-3px);border-color:color-mix(in srgb,var(--acc) 50%,transparent);
box-shadow:0 14px 34px color-mix(in srgb,var(--acc) 22%,transparent)}
.card.mod h3{font-size:18px;display:flex;align-items:center;gap:10px;margin-bottom:6px}
.mod-ico{flex:none;display:grid;place-items:center;width:38px;height:38px;border-radius:11px;color:var(--acc-ink);
background:color-mix(in srgb,var(--acc) 16%,var(--surface));box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--acc) 30%,transparent)}
.mod-ico .ico{width:22px;height:22px}
.card.mod .go{margin-left:auto;color:var(--acc-ink);transition:transform .18s}.card.mod:hover .go{transform:translateX(4px)}
.card.mod .fact b{color:var(--acc-ink)}
.fact{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-top:1px solid var(--grid);font-size:13.5px}
.fact span{color:var(--ink2)}.fact b{font-weight:600;text-align:right}
/* theme switch and sortable tables */
.theme-btn{margin-left:auto;font:inherit;font-size:13px;color:var(--ink2);background:var(--surface);border:1px solid var(--border);
border-radius:99px;padding:3px 12px;cursor:pointer;white-space:nowrap}
.theme-btn:hover{color:var(--ink)}
table.sortable th{cursor:pointer;user-select:none}
table.sortable th[aria-sort]::after{content:" ▾";color:var(--muted)}
table.sortable th[aria-sort="ascending"]::after{content:" ▴"}
/* tooltip */
#tip{position:fixed;z-index:10;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
border-top:3px solid var(--acc);
border-radius:8px;padding:7px 10px;font-size:12.5px;box-shadow:0 4px 16px rgba(0,0,0,.14);max-width:280px;
font-variant-numeric:tabular-nums}
#tip .t1{font-weight:600}#tip div:not(.t1){color:var(--ink2)}
[data-tip]:focus-visible{outline:2px solid var(--ink);outline-offset:1px}
footer{color:var(--muted);font-size:12.5px;border-top:1px solid var(--border);
background:linear-gradient(180deg,var(--page),color-mix(in srgb,var(--acc) 7%,var(--page)))}
footer::before{content:"";display:block;height:4px;background:linear-gradient(90deg,var(--indigo),var(--rasp),var(--lime),var(--mint))}
footer div{max-width:1120px;margin:0 auto;padding:22px 16px 40px;display:flex;align-items:center;gap:10px}
footer .ico{width:18px;height:18px;color:var(--acc)}
@media (max-width:560px){.mr{grid-template-columns:60px 1fr}.mr-p{grid-column:2}header.hero h1,body.home header.hero h1{font-size:26px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""


JS = """
const tip=document.getElementById('tip');
function place(x,y){const r=tip.getBoundingClientRect();let L=x+14,T=y+14;
 if(L+r.width>innerWidth-8)L=x-r.width-14;if(T+r.height>innerHeight-8)T=y-r.height-14;
 tip.style.left=Math.max(8,L)+'px';tip.style.top=Math.max(8,T)+'px'}
function show(el,x,y){tip.replaceChildren(...el.dataset.tip.split('\\n').map((l,i)=>{const d=document.createElement('div');
 d.textContent=l;if(!i)d.className='t1';return d}));tip.hidden=false;place(x,y)}
document.addEventListener('pointerover',e=>{const el=e.target.closest('[data-tip]');el?show(el,e.clientX,e.clientY):tip.hidden=true});
document.addEventListener('pointermove',e=>{if(!tip.hidden)place(e.clientX,e.clientY)});
document.addEventListener('focusin',e=>{const el=e.target.closest('[data-tip]');if(el){const r=el.getBoundingClientRect();
 show(el,r.right,r.top)}});
document.addEventListener('focusout',()=>tip.hidden=true);
// Hide in-bar labels that don't fit their block (never clip them)
function fitLabels(){document.querySelectorAll('.seg span').forEach(s=>{s.style.visibility='';
 if(s.offsetWidth+12>s.parentElement.clientWidth)s.style.visibility='hidden'})}
addEventListener('resize',fitLabels);fitLabels();
// Table filters: a control with data-filter="key" keeps the rows of its table (data-table, default the qpAdm
// .models-table) whose data-key matches; a checkbox keeps rows with data-key="1". The count goes to data-count.
const filters=[...document.querySelectorAll('[data-filter]')];
function applyFilters(){const groups={};
 filters.forEach(f=>{const k=f.dataset.table||'.models-table';(groups[k]=groups[k]||[]).push(f)});
 for(const [sel,fs] of Object.entries(groups)){const t=document.querySelector(sel);if(!t)continue;
  const rows=t.querySelectorAll('tbody tr');let n=0;
  rows.forEach(r=>{const ok=fs.every(f=>f.type==='checkbox'?(!f.checked||r.dataset[f.dataset.filter]==='1')
   :(!f.value||r.dataset[f.dataset.filter]===f.value));r.hidden=!ok;n+=ok});
  const c=document.getElementById(fs[0].dataset.count||'row-count');
  if(c)c.textContent=n+' of '+rows.length+' '+(fs[0].dataset.noun||'models')}}
filters.forEach(f=>f.addEventListener('change',applyFilters));if(filters.length)applyFilters();
// Sortable tables: click a header; numbers (incl. 1.2e-05) sort numerically
document.querySelectorAll('table.sortable').forEach(t=>t.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
 const asc=th.getAttribute('aria-sort')!=='ascending';t.querySelectorAll('th').forEach(h=>h.removeAttribute('aria-sort'));
 th.setAttribute('aria-sort',asc?'ascending':'descending');const body=t.tBodies[0];
 const val=r=>{const s=r.cells[i].textContent.trim(),n=parseFloat(s);return isNaN(n)||!/^[-+\\d.]/.test(s)?s:n};
 [...body.rows].sort((a,b)=>{const x=val(a),y=val(b);const c=typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y));
  return asc?c:-c}).forEach(r=>body.appendChild(r))})));
// Theme: auto (system) -> light -> dark
const themeBtn=document.querySelector('.theme-btn');
function setTheme(m){if(m==='auto')delete document.documentElement.dataset.theme;else document.documentElement.dataset.theme=m;
 if(themeBtn)themeBtn.textContent='Theme: '+m;try{localStorage.setItem('theme',m)}catch(e){}}
if(themeBtn){let m='auto';try{m=localStorage.getItem('theme')||'auto'}catch(e){}setTheme(m);
 themeBtn.addEventListener('click',()=>{const order=['auto','light','dark'];
  setTheme(order[(order.indexOf(themeBtn.textContent.replace('Theme: ',''))+1)%3])})}
"""


# Report modules, one page each, in tab order: (page file stem, tab label). A page that would be empty is
# left out of the tabs; build_report.py sets TABS to the pages it writes.
MODULES = [("index", "Overview"), ("genome", "Genome"), ("health", "Health"), ("drugs", "Drugs"),
           ("traits", "Traits"), ("ethnicity", "Ethnicity"), ("ancient", "Ancient"), ("matches", "Matches")]
TABS = [m for m, _ in MODULES]


def page(title: str, subtitle: str, nav: list[tuple[str, str]], body: str, active: str = "", head: str = "",
         extra: list[tuple[str, str]] | None = None) -> str:
    """One report page: module tabs (active = this page's module), section links, and extra links to detail
    pages that belong to the module (href, label)."""
    on = ' class="on" aria-current="page"'
    tabs = "".join(f'<a href="{m}.html"{on if m == active else ""} style="--dot-l:{MODULE_ACCENT[m][0]};--dot-d:{MODULE_ACCENT[m][1]}"><i></i>{esc(t)}</a>'
                   for m, t in MODULES if m in TABS)
    acc_l, acc_d = MODULE_ACCENT.get(active, MODULE_ACCENT["index"])
    label = dict(MODULES).get(active, "")
    eyebrow = "" if active == "index" else f'<div class="eyebrow">{icon(active)}{esc(label)}</div>'
    h1 = esc(title).replace("My DNA", "My <em>DNA</em>") if active == "index" else esc(title)
    links = "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in nav)
    links += "".join(f'<a class="more" href="{h}">{esc(t)} →</a>' for h, t in extra or [])
    sub = f'<div class="subnav"><div>{links}</div></div>' if links else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style>{head}
<script>try{{const m=localStorage.getItem('theme');if(m&&m!=='auto')document.documentElement.dataset.theme=m}}catch(e){{}}</script></head>
<body data-mod="{esc(active)}"{' class="home"' if active == "index" else ""} style="--acc-l:{acc_l};--acc-d:{acc_d}">
<nav class="tabs"><div><span class="brand">{icon("index")}My DNA</span>{tabs}<button class="theme-btn" type="button">Theme: auto</button></div></nav>
<header class="hero"><div class="hero-in">{eyebrow}<h1>{h1}</h1><div class="sub">{esc(subtitle)}</div></div>{helix_svg()}</header>
{sub}<main>{body}</main>
<footer><div>{icon("index")}Generated {date.today().isoformat()} by scripts/report/build_report.py. Not a medical test.</div></footer>
<div id="tip" hidden></div><script>{JS}</script></body></html>"""
