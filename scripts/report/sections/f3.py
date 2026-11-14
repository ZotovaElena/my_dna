"""Ancient module, outgroup f3: the ranking of ancient individuals, its periods/regions/cultures, the map."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from report.core import card, details, esc, hbars, nice_ticks, page, section, table

# Archaeological period from the AADR group label (its period tokens: _EBA, _Mesolithic, _IronGates, _Viking, ...), the
# individual's date where the label has none. Labels beat fixed date cut-offs, which would put Anatolian farmers
# (8.3k years) in the Mesolithic. With several tokens (Sweden_LN_Viking) the one closest to the date wins.
PERIODS = [("Palaeolithic", "before ~11,700 years ago"), ("Mesolithic", "hunter-gatherers after the Ice Age"),
           ("Neolithic", "first farmers"), ("Copper Age", "Eneolithic, Chalcolithic, Bell Beaker"),
           ("Bronze Age", "incl. Yamnaya, Corded Ware"), ("Iron Age", "incl. Roman and Late Antiquity"),
           ("Medieval", "Migration period to ~1500 CE"), ("Post-medieval", "after ~1500 CE")]


PERIOD_NAMES = [p for p, _ in PERIODS]


PERIOD_TOKENS = [
    ("Palaeolithic", r"UP|EUP|LUP|Paleolithic|Palaeolithic|E[pP]i[pP]al?eolithic|Magdalenian|Gravettian|Aurignacian|"
                     r"Natufian|Azilian|Epigravettian|LGM"),
    ("Mesolithic", r"Mesolithic|Meso|IronGates|EMesolithic|LMesolithic"),
    ("Neolithic", r"N|EN|MN|LN|MLN|ENMN|MNLN|EMN|Neolithic|PPN|PPNA|PPNB|PPNC|PN|LBK|TRB|GlobularAmphora|PittedWare"),
    ("Copper Age", r"C|EC|LC|MC|EarlyC|LateC|MiddleC|Chalcolithic|Eneolithic|CA|ChL|Copper|BellBeaker|Afanasievo"),
    ("Bronze Age", r"BA|EBA|MBA|LBA|MLBA|EMBA|Bronze"),
    ("Iron Age", r"IA|EIA|MIA|LIA|PreRoman|Roman|EarlyRoman|LateRoman|ImperialRoman|Hellenistic|Antiquity|"
                 r"LateAntiquity|Classical|Imperial|Republican|Punic|Etruscan|LaTene|Hallstatt|Scythian|Sarmatian|Xiongnu"),
    ("Medieval", r"Medieval|EarlyMedieval|EMedieval|HighMedieval|LateMedieval|Viking|VikingAge|Migration|"
                 r"MigrationPeriod|Avar|EarlyAvar|LateAvar|MiddleAvar|MiddleLateAvar|Conqueror|Saxon|AngloSaxon|Slav|"
                 r"EarlySlav|Langobard|Lombards|Merovingian|Byzantine|Islamic|GoldenHorde|Mongol|Khazar|Norse|EarlyNorse|"
                 r"LateNorse|Crusader|EarlyArpadian|Arpadian"),
    ("Post-medieval", r"EarlyModern|PostMedieval|Modern|Colonial|Industrial|Napoleonic")]


PERIOD_RX = [(name, re.compile(rf"^(?:{rx})$")) for name, rx in PERIOD_TOKENS]


PERIOD_DATE = [(11700, "Palaeolithic"), (8000, "Mesolithic"), (5000, "Neolithic"), (2800, "Bronze Age"),
               (1500, "Iron Age"), (450, "Medieval")]


# Kept for the ancient page's "by period" text: name -> span shown in its headings
F3_PERIODS = [(None, None, name, span) for name, span in PERIODS]


def period_by_date(bp: float) -> str:
    for lim, name in PERIOD_DATE:
        if bp > lim:
            return name
    return "Post-medieval"


def classify_period(group: str, bp: float) -> str:
    by_date = period_by_date(bp) if np.isfinite(bp) else None
    hits = {name for t in re.split(r"[_\-.]", str(group)) for name, rx in PERIOD_RX if rx.match(t)}
    if not hits and re.search(r"(^|_)HG($|_)", str(group)):
        hits = {"Palaeolithic" if bp > 11700 else "Mesolithic"}
    if not hits:
        return by_date or "Post-medieval"
    if by_date is None or len(hits) == 1:
        return min(hits, key=PERIOD_NAMES.index)
    k = PERIOD_NAMES.index(by_date)
    return min(hits, key=lambda h: (abs(PERIOD_NAMES.index(h) - k), PERIOD_NAMES.index(h)))


REGIONS = {
    "Eastern Europe": ["Ukraine", "Belarus", "Moldova", "Crimea"],
    "Baltic": ["Estonia", "Latvia", "Lithuania"],
    "Scandinavia & North Atlantic": ["Denmark", "Sweden", "Norway", "Finland", "Iceland", "Faroe Islands", "Greenland"],
    "Britain & Ireland": ["United Kingdom", "Ireland", "Channel Islands"],
    "Western Europe": ["France", "Belgium", "Netherlands", "Luxembourg"],
    "Central Europe": ["Germany", "Poland", "Czechia", "Slovakia", "Austria", "Switzerland", "Hungary"],
    "Iberia": ["Spain", "Portugal", "Gibraltar"],
    "Italy & Malta": ["Italy", "Malta"],
    "Balkans": ["Serbia", "Croatia", "Romania", "Bulgaria", "Greece", "Albania", "Montenegro", "North Macedonia",
                "Slovenia", "Bosnia-Herzegovina"],
    "Caucasus": ["Armenia", "Georgia", "Azerbaijan"],
    "Anatolia & Near East": ["Turkey", "Israel", "Lebanon", "Jordan", "Syria", "Iraq", "Iran", "Yemen", "Bahrain"],
    "Central & South Asia": ["Kazakhstan", "Kyrgyzstan", "Turkmenistan", "Uzbekistan", "Tajikistan", "Afghanistan",
                             "Pakistan", "India", "Nepal"],
    "East & Southeast Asia": ["China", "Mongolia", "Republic of Korea", "Japan", "Taiwan", "Vietnam", "Thailand", "Laos",
                              "Malaysia", "Indonesia"],
    "Africa": ["Kenya", "Tunisia", "Sudan", "Morocco", "South Africa", "Tanzania", "Malawi", "Algeria", "Cameroon",
               "Democratic Republic of the Congo", "Botswana", "Saint Helena", "Egypt", "Ethiopia", "Uganda", "Zambia",
               "Canary Islands"],
    "Oceania": ["Guam", "Papua New Guinea", "Vanuatu", "Palau", "Federated States of Micronesia", "Australia", "Tonga",
                "Solomon Islands", "French Polynesia"],
}


REGION_OF = {c: r for r, cs in REGIONS.items() for c in cs}


REGION_NAMES = ["Eastern Europe", "Baltic", "Scandinavia & North Atlantic", "Britain & Ireland", "Western Europe",
                "Central Europe", "Iberia", "Italy & Malta", "Balkans", "Caucasus", "Anatolia & Near East",
                "Siberia", "Central & South Asia", "East & Southeast Asia", "Africa", "Americas", "Oceania", "Unknown"]


def region_of(country: str, lat: float, lon: float) -> str:
    if country == "Russia":
        if np.isfinite(lon) and lon >= 60:
            return "Siberia"
        if np.isfinite(lat) and lat < 45 and 37 <= lon <= 50:
            return "Caucasus"
        return "Eastern Europe"
    if country in REGION_OF:
        return REGION_OF[country]
    return "Unknown" if country in ("..", "") or pd.isna(country) else "Americas"


# Archaeological cultures for the map's population filter: name, regex on the AADR group label
CULTURES = [("Hunter-gatherers (all)", r"(^|_)HG($|_)|Mesolithic|IronGates|Paleolithic|Palaeolithic|Magdalenian"),
            ("Iron Gates", r"IronGates"), ("Magdalenian", r"Magdalenian"), ("Gravettian", r"Gravettian"),
            ("Natufian", r"Natufian"), ("Pitted Ware", r"PittedWare|_PWC"), ("Comb Ceramic", r"CombCeramic|_CCC"),
            ("LBK", r"LBK"), ("Funnel Beaker", r"TRB|FunnelBeaker"), ("Globular Amphora", r"GlobularAmphora|_GAC"),
            ("Trypillia", r"Trypillia|Tripolye|Cucuteni"), ("Khvalynsk", r"Khvalynsk"), ("Yamnaya", r"Yamnaya"),
            ("Afanasievo", r"Afanasievo"), ("Corded Ware", r"CordedWare|Corded_Ware|_CWC"),
            ("Bell Beaker", r"BellBeaker|Bell_Beaker|_BB($|_)"), ("Fatyanovo", r"Fatyanovo"), ("Catacomb", r"Catacomb"),
            ("Únětice", r"Unetice"), ("Sintashta", r"Sintashta"), ("Srubnaya", r"Srubnaya"),
            ("Andronovo", r"Andronovo|Alakul|Fedorovo"), ("Okunevo", r"Okunevo"), ("Hallstatt", r"Hallstatt"),
            ("La Tène", r"LaTene"), ("Scythian", r"Scyth"), ("Sarmatian", r"Sarmat"), ("Xiongnu", r"Xiongnu"),
            ("Roman", r"Roman"), ("Huns", r"Hun($|_|nic)"), ("Avars", r"Avar"), ("Anglo-Saxon", r"Saxon"),
            ("Langobards", r"Langobard|Lombard"), ("Slavs", r"Slav"), ("Viking", r"Viking"),
            ("Hungarian Conquerors", r"Conqueror"), ("Golden Horde", r"GoldenHorde"), ("Mongol era", r"Mongol")]


CULTURE_RX = [(n, re.compile(rx)) for n, rx in CULTURES]


ALTAI_BOX = {"lat": (48, 56), "lon": (78, 100)}   # Altai, Altai Krai, Tuva, Minusinsk basin


def load_f3(out: Path) -> pd.DataFrame | None:
    path = out / "f3" / "outgroup_f3_ancient.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["period"] = pd.Categorical([classify_period(g, b) for g, b in zip(df["group"], df["date_bp"])],
                                  categories=PERIOD_NAMES, ordered=True)
    df["region"] = [region_of(c, la, lo) for c, la, lo in zip(df["country"], df["lat"], df["lon"])]
    df["altai"] = df["lat"].between(*ALTAI_BOX["lat"]) & df["lon"].between(*ALTAI_BOX["lon"])
    top = df.iloc[0]
    # tied with the closest: difference to it within 2 combined standard errors
    df["tied_top"] = (top["f3"] - df["f3"]) < 2 * np.sqrt(top["se"] ** 2 + df["se"] ** 2)
    return df


def f3_groups(df: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """Inverse-variance mean f3 per AADR group label (>= min_n individuals): far less noisy than single genomes."""
    w = 1 / df["se"] ** 2
    g = df.assign(w=w, wf=w * df["f3"]).groupby("group").agg(
        n=("f3", "size"), w=("w", "sum"), wf=("wf", "sum"), date_bp=("date_bp", "median"),
        country=("country", lambda x: x.mode().iat[0]), altai=("altai", "mean"),
        period=("period", lambda x: x.mode().iat[0]))
    g = g[g["n"] >= min_n].assign(f3=lambda x: x["wf"] / x["w"], se=lambda x: 1 / np.sqrt(x["w"]))
    g = g.sort_values("f3", ascending=False).reset_index()
    g["rank"] = np.arange(1, len(g) + 1)
    g["period"] = pd.Categorical(g["period"].astype(str), categories=PERIOD_NAMES, ordered=True)
    return g.assign(iid=g["group"], group=g["n"].map(lambda n: f"{n} people"), locality="", mt_hg="",
                    snps_shared=0, altai=g["altai"] > 0.5)


def age_txt(bp: float) -> str:
    if not np.isfinite(bp):
        return ""
    return f"{bp / 1000:.1f}k yrs" if bp >= 1000 else f"{bp:.0f} yrs"


def f3_dots(rows: pd.DataFrame, lo: float, hi: float, n_label: int = 60) -> str:
    """One row per individual: f3 dot ± 1.96 SE on a zoomed axis."""
    pos = lambda v: (min(max(v, lo), hi) - lo) / (hi - lo) * 100
    ticks = nice_ticks(lo, hi, 5)
    grid = "".join(f'<div class="pp-grid" style="left:{pos(t):.2f}%"></div>' for t in ticks if lo <= t <= hi)
    out = []
    for _, r in rows.iterrows():
        lab = f"{r['iid']} · {r['group']}"
        place = f"{esc(r['locality'])}, {esc(r['country'])}" if r["locality"] else esc(r["country"])
        tip = (f"{esc(r['iid'])} ({esc(r['group'])})&#10;{place}&#10;{age_txt(r['date_bp'])} ago"
               f"&#10;f3 = {r['f3']:.4f} ± {r['se']:.4f}&#10;rank {int(r['rank'])}"
               + (f" · {int(r['snps_shared']):,} SNPs" if r["snps_shared"] else "")
               + (f"&#10;mtDNA {esc(r['mt_hg'])}" if str(r["mt_hg"]) not in ("..", "nan", "") else ""))
        cls = "s2" if r.get("altai") else "s1"
        out.append(f'<div class="pp-label" title="{esc(lab)}">{esc(lab[:n_label])} '
                   f'<span class="muted">{age_txt(r["date_bp"])}, {esc(r["country"])}</span></div>'
                   f'<div class="fp-track">{grid}<div class="fp-ci {cls}" style="left:{pos(r.f3 - 1.96 * r.se):.2f}%;'
                   f'width:{pos(r.f3 + 1.96 * r.se) - pos(r.f3 - 1.96 * r.se):.2f}%;top:50%"></div>'
                   f'<div class="fp-dot {cls}" style="left:{pos(r.f3):.2f}%;top:50%" tabindex="0" data-tip="{tip}"></div></div>')
    axis = "".join(f'<span style="left:{pos(t):.2f}%">{t:.3f}</span>' for t in ticks if lo <= t <= hi)
    return f'<div class="pp fp">{"".join(out)}<div></div><div class="pp-axis">{axis}</div></div>'


def f3_age_svg(df: pd.DataFrame, n_top: int = 20) -> str:
    """f3 against age (log scale) for every individual; closest ones and the Altai region highlighted."""
    W, H, L, R, T, B = 960, 360, 56, 20, 14, 44
    x0, x1 = np.log10(150), np.log10(120000)
    lo, hi = np.nanpercentile(df["f3"], 0.5), df["f3"].max()
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad
    X = lambda v: L + (np.log10(np.clip(v, 150, 120000)) - x0) / (x1 - x0) * (W - L - R)
    Y = lambda v: T + (1 - (np.clip(v, lo, hi) - lo) / (hi - lo)) * (H - T - B)
    g = ""
    for t in nice_ticks(lo, hi, 5):
        if lo <= t <= hi:
            g += (f'<line class="gridl" x1="{L}" x2="{W - R}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                  f'<text class="ax" x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:.3f}</text>')
    for v, lbl in [(300, "300"), (1000, "1k"), (3000, "3k"), (10000, "10k"), (30000, "30k"), (100000, "100k")]:
        g += f'<text class="ax" x="{X(v):.1f}" y="{H - B + 16}" text-anchor="middle">{lbl}</text>'
    g += (f'<line class="axis" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>'
          f'<text class="ax" x="{(L + W - R) / 2:.0f}" y="{H - 6}" text-anchor="middle">years before present (log scale)</text>')
    bg = df.iloc[n_top:]
    bg = bg[~bg["altai"]]
    dots = "".join(f'<circle cx="{X(a):.0f}" cy="{Y(f):.0f}" r="2"/>' for a, f in zip(bg["date_bp"], bg["f3"]))
    hl = ""
    for cls, d in (("s2", df[df["altai"]]), ("s1", df.head(n_top))):
        for _, r in d.iterrows():
            hl += (f'<circle cx="{X(r.date_bp):.1f}" cy="{Y(r.f3):.1f}" r="4" style="fill:var(--{cls[0]}{cls[1:]})" '
                   f'stroke="var(--surface)" stroke-width="1" tabindex="0" data-tip="{esc(r.iid)} ({esc(r.group)})&#10;'
                   f'{esc(r.country)}, {age_txt(r.date_bp)} ago&#10;f3 = {r.f3:.4f} ± {r.se:.4f} (rank {int(r["rank"])})"/>')
    return (f'<figure class="svg-fig"><svg viewBox="0 0 {W} {H}" role="img" aria-label="f3 against age">{g}'
            f'<g style="fill:var(--muted);opacity:.28">{dots}</g>{hl}</svg></figure>')


def f3_table(d: pd.DataFrame, cls: str = "dense sortable") -> str:
    t = pd.DataFrame({"Rank": d["rank"].astype(int), "Individual": d["iid"], "Group": d["group"],
                      "Place": d["locality"].astype(str).str.split("(").str[0].str.strip() + ", " + d["country"].astype(str),
                      "Age": d["date_bp"].map(age_txt), "f3": d["f3"].map(lambda v: f"{v:.4f}"),
                      "± SE": d["se"].map(lambda v: f"{v:.4f}"), "SNPs": d["snps_shared"].map(lambda v: f"{int(v):,}"),
                      "mtDNA": d["mt_hg"].replace({"..": "", "nan": ""}).fillna("")})
    return table(t, cls)


def f3_specific(df: pd.DataFrame, n_show: int = 15) -> str:
    """Your f3 relative to the control Eastern Europeans', per ancient group, centred on the median group.
    Genotype errors in your calls shrink all your f3 by about the same factor, which the ratio and centring remove."""
    if "f3_ctrl" not in df:
        return ""
    d = df[df["f3_ctrl"] > 0].assign(ratio=lambda x: x["f3"] / x["f3_ctrl"], w=lambda x: (x["f3_ctrl"] / x["se"]) ** 2)
    g = d.assign(wr=d["w"] * d["ratio"]).groupby("group").agg(n=("ratio", "size"), w=("w", "sum"), wr=("wr", "sum"),
                                                                date_bp=("date_bp", "median"),
                                                                country=("country", lambda x: x.mode().iat[0]))
    g = g[g["n"] >= 3].assign(ratio=lambda x: x["wr"] / x["w"], se=lambda x: 1 / np.sqrt(x["w"]))
    med = g["ratio"].median()
    g["rel"] = g["ratio"] / med - 1
    g["rel_se"] = g["se"] / med
    g = g.sort_values("rel", ascending=False).reset_index()
    pick = pd.concat([g.head(n_show), g.tail(n_show)])
    lo = (pick["rel"] - 2 * pick["rel_se"]).min()
    hi = (pick["rel"] + 2 * pick["rel_se"]).max()
    pos = lambda v: (min(max(v, lo), hi) - lo) / (hi - lo) * 100
    ticks = nice_ticks(lo, hi, 6)
    grid = "".join(f'<div class="pp-grid" style="left:{pos(t):.2f}%"></div>' for t in ticks if lo <= t <= hi)
    grid += f'<div class="fp-mean" style="left:{pos(0):.2f}%"></div>'
    rows = []
    for i, (_, r) in enumerate(pick.iterrows()):
        if i == n_show:
            rows.append('<div class="pp-label f4-cat">…</div><div></div>')
        cls = "s1" if r.rel > 0 else "s2"
        rows.append(f'<div class="pp-label">{esc(r.group)} <span class="muted">{age_txt(r.date_bp)}, {esc(r.country)}, '
                    f'n={int(r.n)}</span></div><div class="fp-track">{grid}'
                    f'<div class="fp-ci {cls}" style="left:{pos(r.rel - 1.96 * r.rel_se):.2f}%;width:'
                    f'{pos(r.rel + 1.96 * r.rel_se) - pos(r.rel - 1.96 * r.rel_se):.2f}%;top:50%"></div>'
                    f'<div class="fp-dot {cls}" style="left:{pos(r.rel):.2f}%;top:50%" tabindex="0" data-tip="'
                    f'{esc(r.group)}&#10;your f3 / controls\' f3 = {r.ratio:.4f} ({r.rel * 100:+.2f}% vs median group)'
                    f'&#10;{int(r.n)} individuals"></div></div>')
    axis = "".join(f'<span style="left:{pos(t):.2f}%">{t * 100:+.1f}%</span>' for t in ticks if lo <= t <= hi)
    chart = (f'<div class="pp fp">{"".join(rows)}<div></div><div class="pp-axis">{axis}</div></div>'
             f'<div class="chart-note">{len(g):,} ancient groups with ≥ 3 people. 0 = the median group. Top: groups you '
             f'share relatively more with than typical Russians, Belarusians and Ukrainians do; bottom: less.</div>')
    # With many groups, the largest |Z| expected by chance alone (two-sided, 5% family-wise)
    zmax = float(np.abs((g["ratio"] - med) / g["se"]).max())
    z_chance = float(np.sqrt(2 * np.log(len(g) / 0.05)))
    shift = (med - 1) * 100
    if zmax < z_chance:
        verdict = (f'<p class="status good"><b>No ancient group stands out.</b> The largest deviation is |Z| = {zmax:.1f}; '
                   f'with {len(g):,} groups, chance alone gives up to about {z_chance:.1f}. The groups at the top of the list '
                   f'(scattered across Africa, Tibet and the Americas) confirm it is noise. Outgroup f3 cannot tell you apart '
                   f'from a typical Eastern European; the f4 test on the overview page is the sensitive one.</p>'
                   f'<p class="chart-note">Your f3 is {abs(shift):.1f}% {"lower" if shift < 0 else "higher"} than the controls\' '
                   f'for every ancient group alike: genotype errors in low-coverage calls, removed by the ratio.</p>')
        return verdict + details("Show the groups at both ends anyway", chart)
    return (f'<p class="status"><b>Largest deviation |Z| = {zmax:.1f}</b> (chance level ≈ {z_chance:.1f}).</p>' + chart)


def f3_page(df: pd.DataFrame) -> str:
    n = len(df)
    top = df.head(30)
    lo = (top["f3"] - 2.2 * top["se"]).min()
    hi = (top["f3"] + 2.2 * top["se"]).max()
    tied = int(df["tied_top"].sum())
    intro = ("<p>Outgroup f3(Mbuti; you, X) measures the genetic drift you share with ancient individual X since both "
             "lineages split from an African outgroup: the higher, the more of your history you share with that person's "
             "population. It is the valid version of GEDmatch's “Ancient Calculator”. It says nothing about direct descent; "
             "and with one genome on each side, neighbouring values overlap. You enter as <code>me_cram</code> "
             f"(one read per SNP, like the ancient genomes). {n:,} ancient individuals with ≥ 30,000 shared SNPs.</p>")
    secs = [section("about", "How to read this", "", intro + (
        "<p><b>Why hunter-gatherers come out on top.</b> f3 counts shared drift, not recent kinship. About half of your "
        "ancestry goes back to Western and Eastern hunter-gatherers (qpAdm), and those small, isolated populations "
        "carried a lot of drift that you inherited. Northeast Europeans therefore share the most f3 with Mesolithic "
        "Baltic, Scandinavian and Iron Gates foragers (Lazaridis et al. 2014 saw the same), followed by Baltic, "
        "Viking-age and Early Medieval groups that are closer to you in time. Read the ranking as “whose population "
        "history overlaps most with yours”, not “your ancestor”.</p>"))]
    grp = f3_groups(df)
    gt = grp.head(30)
    secs.append(section("groups", "Closest ancient populations",
                        f"Mean of the individuals in each AADR group with at least 3 people ({len(grp):,} groups).",
                        card("Top 30 populations", f3_dots(gt, (gt["f3"] - 2.2 * gt["se"]).min(),
                                                           (gt["f3"] + 2.2 * gt["se"]).max()),
                             "Dot = precision-weighted mean f3, line = 95% interval. Hover for details.")))
    spec = f3_specific(df)
    if spec:
        secs.append(section("specific", "What is specific to you", (
            "The same f3 was computed for 20 AADR Russians, Belarusians and Ukrainians (one random allele per SNP, like "
            "you). Dividing yours by theirs removes what all Eastern Europeans share and leaves where you differ. Single "
            "genomes differ only slightly, so read this as a direction, not a ranking."), card(
            "Your f3 relative to typical Eastern Europeans", spec)))
    secs.append(section("closest", "Closest single individuals",
                        f"Single genomes are noisy: {tied} individuals are statistically tied with the first one "
                        "(within 2 standard errors), so the exact order inside the top 30 means little. A single individual far above "
                        "its own group (e.g. TOR.8 vs TOR.6 from the same site) more likely carries some modern-European "
                        "contamination than special closeness: contamination makes an ancient genome look like you.",
                        card("Top 30 individuals", f3_dots(top, lo, hi),
                             "Dot = f3, line = 95% interval. Orange: Altai/southern Siberia. Hover for place, date, haplogroup.")))
    rows = []
    for lo_, hi_, name, span in F3_PERIODS:
        d = df[df["period"] == name]
        if d.empty:
            continue
        gp = grp[grp["period"] == name].head(3)
        gl = "; ".join(f"{r.iid} ({r.f3:.4f})" for r in gp.itertuples())
        rows.append(f'<h3 class="sub-h">{esc(name)} <span class="muted">({esc(span)}, {len(d):,} individuals)</span></h3>'
                    + (f'<p class="chart-note">Closest populations: {esc(gl)}</p>' if gl else "")
                    + f3_table(d.head(5), "dense"))
    secs.append(section("periods", "Closest in each period", "The five closest individuals of each age.", card(
        "By period", "".join(rows))))
    secs.append(section("time", "f3 through time", "", card(
        "Every ancient individual", f3_age_svg(df),
        "Grey: all individuals. Blue: the 20 closest. Orange: Altai and southern Siberia. Shared drift rises toward the "
        "present for people whose populations contributed to yours.")))
    alt = df[df["altai"]]
    if len(alt):
        ag = f3_groups(alt, 2).head(3)
        recent = alt[alt["date_bp"] < 3000]
        secs.append(section("altai", "Altai and southern Siberia", esc(
            "Ancient people of the region where your family settled about a century ago. The closest of them are "
            + ", ".join(ag["iid"]) + f" (best individual rank {int(alt['rank'].min()):,} of {n:,}): Steppe-derived "
            "Bronze Age groups that came from the west. Iron Age and later locals"
            + (f" (best rank {int(recent['rank'].min()):,})" if len(recent) else "")
            + " carry East Asian ancestry you don't, so they rank far lower."),
            card(f"{len(alt):,} individuals (ranks among all {n:,})", f3_table(alt.head(40)))))
    secs.append(section("all", "Top 500", "", card("Sortable", f3_table(df.head(500)))))
    nav = [("about", "How to read"), ("groups", "Populations")] + ([("specific", "Specific to you")] if "f3_ctrl" in df else []) + [
           ("closest", "Individuals"), ("periods", "By period"),
           ("time", "Through time")]
    nav += ([("altai", "Altai")] if len(alt) else []) + [("all", "Top 500")]
    return page("Ancient individuals closest to you", "Outgroup f3(Mbuti; you, X), AADR v66", nav, "".join(secs),
                active="ancient", extra=[("map.html", "Map"), ("qpadm.html", "qpAdm details")])


def sec_f3_teaser(df: pd.DataFrame | None) -> str:
    if df is None:
        return ""
    grp = f3_groups(df)
    best = grp.groupby("period", observed=True).head(1).set_index("period")
    rows = [(f"{p}: {r.iid}", r.f3, f"{r.f3:.4f}", f"{r.iid}, {r.country}\n{age_txt(r.date_bp)} ago, {r.n} people\n"
             f"f3 = {r.f3:.4f} ± {r.se:.4f}, rank {int(r['rank'])} of {len(grp):,} populations") for p, r in best.iterrows()]
    base = min(v for _, v, _, _ in rows) - 0.002
    body = card("Closest ancient population in each period", hbars([(a, v - base, t, tip) for a, v, t, tip in rows]),
                f'Outgroup f3(Mbuti; you, X), mean per AADR group, {len(df):,} individuals (bars start at {base:.3f}; '
                'hunter-gatherer groups score high because about half your ancestry comes from them). '
                'Details: <a href="ancient_individuals.html">ancient individuals page</a> and the '
                '<a href="map.html">interactive map</a>.')
    return section("f3", "Closest ancient populations (outgroup f3)", "", body)


LEAFLET = "https://unpkg.com/leaflet@1.9.4/dist/"


MAP_CSS = """
.mc{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:flex-end;margin:6px 0 10px}
.mc label{display:flex;flex-direction:column;gap:3px;font-size:12.5px;color:var(--ink2)}
.mc input[type=search]{font:inherit;padding:3px 8px;border-radius:6px;border:1px solid var(--axis);background:var(--surface);
color:var(--ink);min-width:190px}
.mc button,.chip{font:inherit;font-size:13px;color:var(--ink2);background:var(--surface);border:1px solid var(--axis);
border-radius:999px;padding:2px 10px;cursor:pointer}
.mc button:hover,.chip:hover{color:var(--ink)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px}
.chip{display:inline-flex;align-items:center;gap:6px}
.chip[aria-pressed="false"]{opacity:.45;text-decoration:line-through}
.chip .sw{width:9px;height:9px;border-radius:50%}
.chip .n{color:var(--muted);font-size:12px}
#map{height:min(72vh,680px);min-height:420px;border-radius:10px;border:1px solid var(--border);background:var(--track)}
#map-off{padding:24px;color:var(--ink2)}
.map-status{font-size:13.5px;color:var(--ink2);margin:8px 0 6px;font-variant-numeric:tabular-nums}
.map-status b{color:var(--ink)}
.leaflet-container{font:inherit}
.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:var(--surface);color:var(--ink);
box-shadow:0 4px 16px rgba(0,0,0,.18)}
.leaflet-popup-content{margin:12px 14px;font-size:13px;line-height:1.45;font-variant-numeric:tabular-nums;min-width:240px}
.leaflet-popup-content h4{margin:0 0 2px;font-size:14.5px}
.leaflet-popup-content .pk{color:var(--muted)}
.leaflet-popup-content table{font-size:12.5px;margin-top:6px}
.leaflet-popup-content td{padding:1px 8px 1px 0;border:0}
.site-list{margin-top:8px;border-top:1px solid var(--grid);padding-top:6px;max-height:170px;overflow:auto}
.site-list div{cursor:pointer;color:var(--ink2)}.site-list div:hover{color:var(--ink)}.site-list .cur{color:var(--ink);font-weight:600}
.leaflet-control-attribution{background:var(--surface)!important;color:var(--muted)!important}
.leaflet-control-attribution a{color:var(--link)!important}
.leaflet-bar a{background:var(--surface);color:var(--ink);border-color:var(--border)}
#map-table tbody tr{cursor:pointer}#map-table tbody tr:hover td{background:var(--track)}
"""


MAP_JS = r"""
(function(){
const M=JSON.parse(document.getElementById('map-data').textContent);
const R=M.rows.map(r=>({id:r[0],iid:r[1],g:M.groups[r[2]],loc:M.locs[r[3]],c:M.countries[r[4]],lat:r[5],lon:r[6],bp:r[7],
 f3:r[8],se:r[9],rank:r[10],snps:r[11],mt:r[12],y:r[13],pub:M.pubs[r[14]],per:r[15],reg:r[16],cul:r[17],diff:r[18]}));
R.forEach(r=>{r.hay=[r.id,r.iid,r.g,r.loc,r.c,r.mt,r.y,r.pub].join(' ').toLowerCase();r.site=r.lat+','+r.lon});
R.sort((a,b)=>b.rank-a.rank);                          // best last: drawn on top
const N=M.n_total,$=id=>document.getElementById(id);
const fmtAge=bp=>{const y=1950-bp,cal=y<0?Math.round(-y).toLocaleString()+' BCE':Math.round(y)+' CE';
 return (bp>=1000?(bp/1000).toFixed(1)+'k':Math.round(bp))+' years ago (c. '+cal+')'};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let pal={};
function readPal(){pal={pt:css('--pt'),surface:css('--surface'),q:[1,2,3,4].map(k=>css('--q'+k)),s:[1,2,3,4,5,6,7,8].map(k=>css('--s'+k))}}
readPal();
const mode=$('m-color');
const tier=r=>r.rank<=20?3:r.rank<=100?2:r.rank<=500?1:r.rank<=2000?0:-1;
const colorOf=r=>mode.value==='period'?pal.s[r.per]:(tier(r)<0?pal.pt:pal.q[tier(r)]);
const radius=r=>r.rank<=20?7:r.rank<=100?5.5:r.rank<=500?4.5:3.5;
function legend(){const L=$('map-legend');
 const k=(c,t)=>'<span class="key"><i class="sw" style="background:'+c+';border-radius:50%"></i>'+t+'</span>';
 const sz='<span class="key"><i class="sw" style="width:14px;height:14px;border-radius:50%;background:var(--ink2)"></i>'+
  '<i class="sw" style="width:10px;height:10px;border-radius:50%;background:var(--ink2)"></i>'+
  '<i class="sw" style="width:7px;height:7px;border-radius:50%;background:var(--ink2)"></i>Larger = closer (top 20, 100, 500)</span>';
 L.innerHTML=mode.value==='period'?'<span class="key">Colour = period, as on the buttons above</span>'+sz
  :k(pal.q[3],'Top 20')+k(pal.q[2],'21–100')+k(pal.q[1],'101–500')+k(pal.q[0],'501–2,000')+k(pal.pt,'Rank over 2,000')}

// --- map
let map=null,layer=null,tiles=null;
const dark=()=>{const t=document.documentElement.dataset.theme;return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches};
// Basemaps without an API key: Esri gray canvas (light/dark); OpenStreetMap if Esri tiles fail to load
const ESRI='https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/';
const PROVIDERS=[{url:()=>ESRI+(dark()?'World_Dark_Gray_Base':'World_Light_Gray_Base')+'/MapServer/tile/{z}/{y}/{x}',
  attr:'Tiles &copy; <a href="https://www.esri.com">Esri</a> &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors',max:16},
 {url:()=>'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  attr:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',max:18}];
let prov=0,tileErr=0;
const tileUrl=()=>PROVIDERS[prov].url();
const EUROPE=[[33,-12],[68,80]];
if(window.L){
 map=L.map('map',{preferCanvas:true,worldCopyJump:true,minZoom:2}).fitBounds(EUROPE);
 const addTiles=()=>{tileErr=0;const p=PROVIDERS[prov];
  tiles=L.tileLayer(p.url(),{maxZoom:p.max,maxNativeZoom:p.max,attribution:p.attr}).addTo(map);
  tiles.on('tileerror',()=>{if(++tileErr===6&&prov<PROVIDERS.length-1){map.removeLayer(tiles);prov++;addTiles()}})};
 addTiles();
 const renderer=L.canvas({padding:.5,tolerance:3});
 layer=L.layerGroup().addTo(map);
 R.forEach(r=>{r.m=L.circleMarker([r.lat,r.lon],{renderer,radius:radius(r),weight:1.2,color:pal.surface,
  fillColor:colorOf(r),fillOpacity:r.rank<=2000?.9:.55});
  r.m.on('mouseover',e=>{const o=e.originalEvent;show({dataset:{tip:r.iid+'\n'+r.g+'\n'+fmtAge(r.bp)+'\nRank '+r.rank.toLocaleString()+
   ' of '+N.toLocaleString()+' · f3 '+r.f3.toFixed(4)}},o.clientX,o.clientY)});
  r.m.on('mouseout',()=>{tip.hidden=true});
  r.m.on('click',()=>openPop(r))})
}else{$('map').innerHTML='<div id="map-off">The map needs an internet connection (Leaflet and the map tiles load '+
 'from the web). The filters and the table below still work.</div>'}
function recolor(){readPal();legend();R.forEach(r=>r.m&&r.m.setStyle({fillColor:colorOf(r),color:pal.surface}))}
mode.addEventListener('change',recolor);
function themeChanged(){if(tiles)tiles.setUrl(tileUrl());recolor()}
new MutationObserver(themeChanged).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',themeChanged);

// --- filters
const st={per:new Set(M.periods.map((_,i)=>i)),reg:-1,top:0,cul:-1,q:''};
const chips=$('m-periods');
chips.innerHTML=M.periods.map((p,i)=>'<button type="button" class="chip" aria-pressed="true" data-i="'+i+'" title="'+
 M.period_notes[i]+'"><i class="sw" style="background:var(--s'+(i+1)+')"></i>'+p+' <span class="n">'+
 M.period_n[i].toLocaleString()+'</span></button>').join('');
chips.addEventListener('click',e=>{const b=e.target.closest('.chip');if(!b)return;const i=+b.dataset.i;
 if(e.altKey||e.shiftKey){st.per=new Set([i])}else if(st.per.has(i))st.per.delete(i);else st.per.add(i);
 chips.querySelectorAll('.chip').forEach(c=>c.setAttribute('aria-pressed',st.per.has(+c.dataset.i)));apply()});
$('m-only').addEventListener('click',()=>{const all=st.per.size===M.periods.length;
 st.per=new Set(all?[]:M.periods.map((_,i)=>i));chips.querySelectorAll('.chip').forEach(c=>c.setAttribute('aria-pressed',!all));apply()});
const sel=(id,key)=>$(id).addEventListener('change',e=>{st[key]=+e.target.value;apply()});
sel('m-region','reg');sel('m-top','top');sel('m-culture','cul');
let qt;$('m-q').addEventListener('input',e=>{clearTimeout(qt);qt=setTimeout(()=>{st.q=e.target.value.trim().toLowerCase();apply()},150)});
$('m-reset').addEventListener('click',()=>{st.per=new Set(M.periods.map((_,i)=>i));st.reg=-1;st.top=0;st.cul=-1;st.q='';
 ['m-region','m-culture'].forEach(i=>$(i).value='-1');$('m-top').value='0';$('m-q').value='';
 chips.querySelectorAll('.chip').forEach(c=>c.setAttribute('aria-pressed','true'));apply();if(map)map.fitBounds(EUROPE)});
$('m-europe').addEventListener('click',()=>map&&map.fitBounds(EUROPE));
$('m-world').addEventListener('click',()=>map&&map.setView([30,30],2));
$('m-fit').addEventListener('click',()=>{if(!map)return;const v=R.filter(pass);if(v.length)map.fitBounds(L.latLngBounds(v.map(r=>[r.lat,r.lon])).pad(.15))});
const pass=r=>st.per.has(r.per)&&(st.reg<0||r.reg===st.reg)&&(!st.top||r.rank<=st.top)&&(st.cul<0||r.cul.includes(st.cul))&&
 (!st.q||r.hay.includes(st.q));
let shown=[];
function apply(){shown=R.filter(pass);
 if(layer){layer.clearLayers();shown.forEach(r=>layer.addLayer(r.m))}
 const best=shown.length?shown[shown.length-1]:null;
 $('map-status').innerHTML=shown.length?'<b>'+shown.length.toLocaleString()+'</b> of '+R.length.toLocaleString()+
  ' individuals shown · closest: <b>'+best.iid+'</b> ('+best.g+', rank '+best.rank.toLocaleString()+')':'No individual matches these filters.';
 table()}

// --- table of the closest in view
function table(){const rows=shown.slice(-100).reverse();
 $('map-tbody').innerHTML=rows.map(r=>'<tr data-k="'+R.indexOf(r)+'"><td>'+r.rank.toLocaleString()+'</td><td>'+r.iid+'</td><td>'+
  r.g+'</td><td>'+(r.loc.split('(')[0].trim()||'–')+', '+r.c+'</td><td>'+(r.bp>=1000?(r.bp/1000).toFixed(1)+'k':r.bp)+
  '</td><td>'+M.periods[r.per]+'</td><td>'+r.f3.toFixed(4)+'</td><td>'+r.snps.toLocaleString()+'</td><td>'+(r.mt||'')+
  '</td><td>'+(r.y||'')+'</td></tr>').join('');
 $('map-tcount').textContent=shown.length>100?'the closest 100 of '+shown.length.toLocaleString():'all '+shown.length}
$('map-tbody').addEventListener('click',e=>{const tr=e.target.closest('tr');if(!tr)return;const r=R[+tr.dataset.k];
 if(map){map.setView([r.lat,r.lon],Math.max(map.getZoom(),6));openPop(r);$('map').scrollIntoView({behavior:'smooth',block:'center'})}});

// --- popup: the individual, then everyone shown at the same site
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function popHtml(r){const site=shown.filter(x=>x.site===r.site).sort((a,b)=>a.rank-b.rank);
 const row=(k,v)=>v?'<tr><td class="pk">'+k+'</td><td>'+v+'</td></tr>':'';
 let h='<h4>'+esc(r.iid)+'</h4><div>'+esc(r.g)+'</div><table>'+
  row('Site',esc(r.loc)+', '+esc(r.c))+row('Age',fmtAge(r.bp))+row('Period',M.periods[r.per]+' · '+M.regions[r.reg])+
  row('f3',r.f3.toFixed(4)+' ± '+r.se.toFixed(4))+row('Rank','<b>'+r.rank.toLocaleString()+'</b> of '+N.toLocaleString())+
  (r.diff===null?'':row('vs E. Europeans',(r.diff>=0?'+':'')+r.diff.toFixed(4)))+
  row('Shared SNPs',r.snps.toLocaleString())+row('mtDNA',esc(r.mt))+row('Y-DNA',esc(r.y))+row('Study',esc(r.pub))+
  row('AADR ID',esc(r.id))+'</table>';
 if(site.length>1)h+='<div class="site-list"><div class="pk" style="cursor:default">'+site.length+' individuals shown at this site:</div>'+
  site.slice(0,40).map(x=>'<div data-k="'+R.indexOf(x)+'" class="'+(x===r?'cur':'')+'">#'+x.rank.toLocaleString()+' '+esc(x.iid)+
  ' · '+(x.bp>=1000?(x.bp/1000).toFixed(1)+'k':x.bp)+' · '+esc(x.g)+'</div>').join('')+'</div>';
 return h}
let pop=null;
function openPop(r){if(!map)return;tip.hidden=true;pop=L.popup({maxWidth:340,autoPanPadding:[30,30]}).setLatLng([r.lat,r.lon])
 .setContent(popHtml(r)).openOn(map)}
document.addEventListener('click',e=>{const d=e.target.closest('.site-list [data-k]');if(d&&pop)pop.setContent(popHtml(R[+d.dataset.k]))});
legend();apply();
})();
"""


def map_page(df: pd.DataFrame) -> str:
    d = df[df["lat"].notna() & df["lon"].notna()].copy()
    n_total, n_off = len(df), int(len(df) - len(d))

    def codes(values) -> tuple[list[str], list[int]]:
        cats = list(dict.fromkeys(values))
        idx = {v: i for i, v in enumerate(cats)}
        return cats, [idx[v] for v in values]

    clean = lambda s: "" if pd.isna(s) or str(s) in ("..", "nan", "n/a") else str(s)
    groups, gi = codes(d["group"].astype(str).tolist())
    locs, li = codes(d["locality"].map(clean).tolist())
    countries, ci = codes(d["country"].map(clean).tolist())
    pubs, pi = codes(d["pub"].map(clean).tolist())
    cul_of = {g: [k for k, (_, rx) in enumerate(CULTURE_RX) if rx.search(g)] for g in groups}
    diff = d["f3_diff"] if "f3_diff" in d else pd.Series(np.nan, index=d.index)
    rows = []
    for r, g, l, c, p_, dv in zip(d.itertuples(), gi, li, ci, pi, diff):
        y = clean(r.y_hg)
        rows.append([r.id, r.iid, g, l, c, round(float(r.lat), 4), round(float(r.lon), 4), int(r.date_bp),
                     round(float(r.f3), 5), round(float(r.se), 5), int(r.rank), int(r.snps_shared), clean(r.mt_hg),
                     "female" if y.startswith("n/a") else y, p_, PERIOD_NAMES.index(str(r.period)),
                     REGION_NAMES.index(r.region), cul_of[groups[g]], None if pd.isna(dv) else round(float(dv), 5)])
    cul_n = [sum(1 for r in rows if k in r[17]) for k in range(len(CULTURES))]
    reg_n = d["region"].value_counts()
    per_n = d["period"].value_counts()
    data = {"rows": rows, "groups": groups, "locs": locs, "countries": countries, "pubs": pubs, "n_total": n_total,
            "periods": PERIOD_NAMES, "period_notes": [n for _, n in PERIODS],
            "period_n": [int(per_n.get(p, 0)) for p in PERIOD_NAMES], "regions": REGION_NAMES}
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

    reg_opts = "".join(f'<option value="{i}">{esc(r)} ({int(reg_n[r]):,})</option>'
                       for i, r in enumerate(REGION_NAMES) if reg_n.get(r, 0))
    cul_opts = "".join(f'<option value="{k}">{esc(n)} ({cul_n[k]:,})</option>'
                       for k, (n, _) in enumerate(CULTURES) if cul_n[k] >= 3)
    top_opts = "".join(f'<option value="{v}">{t}</option>' for v, t in
                       [(0, f"All {n_total:,}"), (20, "Top 20"), (100, "Top 100"), (500, "Top 500"), (1000, "Top 1,000"),
                        (2000, "Top 2,000")])
    controls = (
        '<div class="chips" id="m-periods" role="group" aria-label="Periods"></div>'
        '<div class="mc">'
        f'<label>Region<select id="m-region"><option value="-1">All regions</option>{reg_opts}</select></label>'
        f'<label>Genetic affinity<select id="m-top">{top_opts}</select></label>'
        f'<label>Culture<select id="m-culture"><option value="-1">All cultures</option>{cul_opts}</select></label>'
        '<label>Search<input type="search" id="m-q" placeholder="ID, group, site, haplogroup…"></label>'
        '<label>Colour by<select id="m-color"><option value="period">Period</option>'
        '<option value="rank">Affinity (rank)</option></select></label>'
        '<button type="button" id="m-only">All / no periods</button>'
        '<button type="button" id="m-fit">Zoom to shown</button><button type="button" id="m-europe">Europe</button>'
        '<button type="button" id="m-world">World</button><button type="button" id="m-reset">Reset</button></div>')
    top = df.iloc[0]
    tied = int(df["tied_top"].sum()) if "tied_top" in df else 0
    lede = (f"Every ancient individual in the outgroup-f3 comparison ({n_total:,} genomes from the AADR), placed at its "
            "burial site. f3(Mbuti; you, X) measures the genetic drift you share with that person's population; rank 1 "
            f"is the highest ({esc(top['iid'])}, {esc(str(top['group']))}). Click a point for details and for everyone "
            "buried at the same site; shift-click a period to show only that period.")
    body = (controls + '<div class="map-status" id="map-status"></div><div class="legend" id="map-legend"></div>'
            '<div id="map"></div>')
    notes = (f"<p><b>Read ranks loosely.</b> Single ancient genomes are noisy: {tied} individuals are statistically tied "
             "with rank 1, and neighbouring ranks overlap. Groups of individuals (the populations on the "
             "<a href=\"ancient_individuals.html\">ancient individuals page</a>) are more reliable than any single point.</p>"
             "<p><b>Why hunter-gatherers rank high.</b> f3 measures shared drift, and small, isolated foraging "
             "populations carried a lot of it. Read the map as “whose population history overlaps with yours”, "
             "not as a map of your ancestors.</p>"
             f"<p><b>Data.</b> Coordinates, dates, groups and haplogroups from the AADR v66 annotation. Periods come from "
             "the group label (_EBA, _Mesolithic, _Viking, …), or from the date where the label has none. "
             f"{n_off} individuals without coordinates are left out. Basemap: Esri gray canvas (© Esri, HERE, Garmin, © OpenStreetMap contributors), no API key needed.</p>")
    tbl = ('<div class="table-wrap"><table class="dense" id="map-table"><thead><tr><th>Rank</th><th>Individual</th>'
           '<th>Group</th><th>Site</th><th>Age</th><th>Period</th><th>f3</th><th>SNPs</th><th>mtDNA</th><th>Y-DNA</th>'
           '</tr></thead><tbody id="map-tbody"></tbody></table></div>')
    secs = (section("explore", "Map of your ancient matches", lede, card("Ancient individuals by outgroup f3", body))
            + section("list", "Closest individuals in view", "",
                      card("Individuals matching the filters", tbl,
                           'Showing <span id="map-tcount"></span>, by rank. Click a row to find it on the map.'))
            + section("notes", "How to read the map", "", card("Notes", notes)))
    secs += (f'<script type="application/json" id="map-data">{blob}</script>'
             f'<script>{MAP_JS}</script>')
    head = (f'<link rel="stylesheet" href="{LEAFLET}leaflet.css" crossorigin="">'
            f'<script src="{LEAFLET}leaflet.js" crossorigin=""></script><style>{MAP_CSS}</style>')
    return page("Map of ancient matches", f"Outgroup f3(Mbuti; you, X), {n_total:,} ancient individuals, AADR v66",
                [("explore", "Map"), ("list", "Closest in view"), ("notes", "How to read")], secs, active="ancient",
                head=head, extra=[("ancient_individuals.html", "Ancient individuals"), ("qpadm.html", "qpAdm details")])
