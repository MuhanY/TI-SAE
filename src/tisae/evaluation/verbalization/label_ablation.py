"""Verbalization label ablation: isolates what the dictionary labels contribute.

Conditions (episode 48h raw-values table appended to ALL; baseline cells come
from the episode file's attribution_cells, suffix att/tsr switches the baseline
method from IG to attention/TSR):
  A / Aatt / Atsr        baseline cells, flat, +per-cell importance   (CELL framing)
  C                      TI-SAE concepts + label + description        (CONCEPT framing)
  D                      TI-SAE concepts, no label                    (CONCEPT framing)
  Dp                     D + cross-episode characterization, no label (CONCEPT framing)
  i / iatt / itsr        LLM-grouped baseline cells, no label         (CONCEPT framing)
  iia / iiaatt / iiatsr  baseline cells grouped by channel            (CONCEPT framing)

Comparisons (12): D/Dp vs each baseline's cells; full C vs the LLM-summarized
and channel-grouped variants of each baseline.
Verbalizer + i-grouping = Claude Sonnet 4.6 ; pairwise judge = GPT-4o.

Requires episodes from extract_episodes.py and the latent dictionary from
tisae.dictionary. Resumable; API keys required.
"""
import argparse, json, random
from pathlib import Path
from collections import defaultdict

from tisae.evaluation.verbalization.build_prompts import (
    PromptBuilder, VERBALIZER_SYSTEM, PAIRWISE_DIM_RUBRICS, tabulate_raw)
from tisae.dictionary.llm_api import build_client
from tisae.dictionary.llm_labeler import parse_llm_json
from tisae.models.ontology import VARIABLE_CHANNEL_MAP, CHANNEL_DEFINITIONS

CHAN = {c.idx: c.name for c in CHANNEL_DEFINITIONS}
def var_chan(v):
    chs = VARIABLE_CHANNEL_MAP.get(v)
    return CHAN.get(chs[0], "Other") if chs else "Other"

CONCEPT_FRAMING = "The model identified these clinical concepts as most important for its prediction:"
CELL_FRAMING = "The model's attribution analysis identified these observations as most important for its prediction:"
CONDITIONS = ["A", "Aatt", "Atsr", "C", "D", "Dp",
              "i", "iatt", "itsr", "iia", "iiaatt", "iiatsr"]
COMPARISONS = [("D", "A"), ("D", "Aatt"), ("D", "Atsr"),
               ("Dp", "A"), ("Dp", "Aatt"), ("Dp", "Atsr"),
               ("C", "i"), ("C", "iatt"), ("C", "itsr"),
               ("C", "iia"), ("C", "iiaatt"), ("C", "iiatsr")]

BASELINE_SUFFIX = {"att": "attention", "tsr": "tsr"}

def split_cond(cond):
    for suf, m in BASELINE_SUFFIX.items():
        base = cond[:-len(suf)]
        if cond.endswith(suf) and base in ("A", "i", "iia"):
            return base, m
    return cond, "ig"
GROUP_SYS = ("You are a clinician. Group the model's flagged observations into clinical concepts "
             "(co-varying variables over a time window). Output plain text, one concept per line as: "
             "[Channel] var1, var2 — hours A-B — key values. No preamble.")


def _body(ep, framing, block):
    return (f"Patient prediction: {ep['risk_level']} mortality risk (probability: {ep['pred_prob']})\n\n"
            f"{framing}\n{block}\n\n"
            f"Clinical values over the 48-hour ICU stay:\n{tabulate_raw(ep['raw_values_all'])}")


def render_cells(cells, channel=False):
    out = []
    for c in cells:
        ch = f" [{var_chan(c['variable'])}]" if channel else ""
        v = f", value: {c['value_raw']}" if c.get("value_raw") is not None else ""
        out.append(f"  Hour {c['t']}: {c['variable']}{ch} (importance: {c['importance']:.4f}{v})")
    return "\n".join(out)


def render_concepts(concepts, label=False, char=None):
    """Render TI-SAE concepts. label: show clinical_label+desc.
    char: dict {k_c: characterization} -> add cross-episode line."""
    vind = {v: i for i, v in enumerate(VAR_NAMES)}
    lines = []
    for con in concepts:
        vars_str = ", ".join(f"{v[0]} (wt={v[1]})" for v in con["top_variables"][:3])
        lab = f' "{con["clinical_label"]}"' if (label and con.get("clinical_label")) else ""
        lines.append(f"  Concept {con['rank']+1} [{con['channel']}]{lab} "
                     f"(total weight={con['total_weight']:.1f}): {vars_str}")
        if label and con.get("clinical_description"):
            lines.append("    " + con["clinical_description"].split(". ")[0].rstrip(".") + ".")
        if char is not None:
            ch = char.get(f"{con.get('k_eff')}_{con.get('c_eff')}")
            if ch:
                pos = ch.get("mean_island_positions")
                pos_s = (f"; typical hours {', '.join(f'{p:.0f}' for p in pos)}"
                         if isinstance(pos, list) and pos else "")
                lines.append(f"    [across cohort: activates in {ch['activation_frequency']*100:.0f}% "
                             f"of patients; mortality odds ratio {ch['mortality_odds_ratio']:.2f}{pos_s}]")
        for isl in sorted([i for i in con["islands"] if i["weight"] >= 0.01], key=lambda x: x["t_start"]):
            lines.append(f"    Island h{isl['t_start']}-{isl['t_end']} (wt={isl['weight']:.2f}):")
            for vn, vals in isl.get("values", {}).items():
                if not vals:
                    continue
                vi = vind.get(vn)
                parts = []
                for v in vals[:6]:
                    parts.append(f"h{v['t']}={v['value_raw']}")
                lines.append(f"      {vn}: {', '.join(parts)}")
    return "\n".join(lines)


def render_channel_groups(cells):
    """Group IG cells by channel (no islands), keep per-cell importance."""
    by = defaultdict(list)
    for c in cells:
        by[var_chan(c["variable"])].append(c)
    groups = sorted(by.items(), key=lambda kv: -sum(abs(c["importance"]) for c in kv[1]))
    lines = []
    for i, (ch, cs) in enumerate(groups[:5]):
        tot = sum(abs(c["importance"]) for c in cs)
        vw = defaultdict(float)
        for c in cs:
            vw[c["variable"]] += abs(c["importance"])
        tv = ", ".join(v for v, _ in sorted(vw.items(), key=lambda x: -x[1])[:3])
        lines.append(f"  Concept {i+1} [{ch}] (total weight={tot:.2f}): {tv}")
        for c in sorted(cs, key=lambda c: -abs(c["importance"]))[:8]:
            lines.append(f"    Hour {c['t']}: {c['variable']} (importance: {c['importance']:.4f}, value: {c['value_raw']})")
    return "\n".join(lines)


def build_block(cond, ep, char, group_client=None, group_cache=None):
    """Return (framing, analysis_block) for a condition."""
    cond, method = split_cond(cond)
    ig = ep["attribution_cells"][method]
    con = ep["tisae_top5_latents"]
    if cond == "A":
        return CELL_FRAMING, render_cells(ig, channel=False)
    if cond == "C":
        return CONCEPT_FRAMING, render_concepts(con, label=True)
    if cond == "D":
        return CONCEPT_FRAMING, render_concepts(con, label=False)
    if cond == "Dp":
        return CONCEPT_FRAMING, render_concepts(con, label=False, char=char)
    if cond == "iia":
        return CONCEPT_FRAMING, render_channel_groups(ig)
    if cond == "i":
        key = str(ep["ep_idx"]) if method == "ig" else f"{ep['ep_idx']}_{method}"
        if group_cache is not None and key in group_cache:
            grouped = group_cache[key]
        else:
            prompt = f"Model-flagged observations:\n{render_cells(ig, channel=True)}"
            grouped, _ = group_client.label_one(prompt, GROUP_SYS, f"group_{key}")
            if group_cache is not None:
                group_cache[key] = grouped
        return CONCEPT_FRAMING, grouped.strip()
    raise ValueError(cond)


VAR_NAMES = None  # set in main


def main():
    global VAR_NAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="experiments/verbalization/episodes_data.json")
    ap.add_argument("--dictionary", default="experiments/discrete/mimic3_mortality/dictionary/latent_dictionary.json")
    ap.add_argument("--dataset", default="data/unified_mimic3_48h.pkl")
    ap.add_argument("--output_dir", default="experiments/ablation/label_ablation")
    ap.add_argument("--verb_provider", default="claude"); ap.add_argument("--verb_model", default="claude-sonnet-4-6")
    ap.add_argument("--judge_provider", default="openai"); ap.add_argument("--judge_model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    import pickle
    VAR_NAMES = pickle.load(open(args.dataset, "rb"))["metadata"]["variable_names"]
    eps = json.load(open(args.episodes))
    if args.n:
        eps = eps[:args.n]
    dic = json.load(open(args.dictionary))
    if isinstance(dic, list):
        char = {f"{e['latent_k']}_{e['channel_c']}": e for e in dic}
    else:
        char = dic
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    builder = PromptBuilder()
    DIMS = ["concept", "actionability"]  # faithfulness is judged in a separate, unblinded call
    rng = random.Random(42)

    if args.dry_run:
        ep = eps[0]
        for cond in CONDITIONS:
            gc = build_client(args.verb_provider, args.verb_model) if split_cond(cond)[0] == "i" else None
            fr, blk = build_block(cond, ep, char, group_client=gc, group_cache={})
            print(f"\n{'='*70}\n{cond}\n{'='*70}\n{fr}\n{blk[:1200]}")
        return

    # Phase A: verbalize all conditions (resumable) + i-grouping cache
    vpath = out / "verb_results.json"; verb = json.load(open(vpath)) if vpath.exists() else {}
    gpath = out / "group_cache.json"; gcache = json.load(open(gpath)) if gpath.exists() else {}
    vclient = build_client(args.verb_provider, args.verb_model, temperature=args.temperature)
    blocks = {}  # (ep_idx, cond) -> (framing, block) for faithfulness judge
    for ep in eps:
        for cond in CONDITIONS:
            fr, blk = build_block(cond, ep, char, group_client=vclient, group_cache=gcache)
            blocks[(ep["ep_idx"], cond)] = (fr, blk)
            json.dump(gcache, open(gpath, "w"))
            key = f"{ep['ep_idx']}_{cond}"
            if key in verb:
                continue
            resp, _ = vclient.label_one(_body(ep, fr, blk), VERBALIZER_SYSTEM, key)
            verb[key] = resp
            json.dump(verb, open(vpath, "w"), indent=2)
            print(f"  verbalized {key}: {len(resp)} chars")

    # Phase B: pairwise judge (resumable)
    ppath = out / "pairwise_results.json"; pw = json.load(open(ppath)) if ppath.exists() else []
    seen = {(r["ep_idx"], r["cmp"]) for r in pw}
    jclient = build_client(args.judge_provider, args.judge_model, temperature=args.temperature)
    for ep in eps:
        for X, Y in COMPARISONS:
            cmp = f"{X}_vs_{Y}"
            if (ep["ep_idx"], cmp) in seen:
                continue
            ex, ey = verb[f"{ep['ep_idx']}_{X}"], verb[f"{ep['ep_idx']}_{Y}"]
            ax, ay = blocks[(ep["ep_idx"], X)][1], blocks[(ep["ep_idx"], Y)][1]
            x_is_a = rng.random() < 0.5
            (ea, eb, aa, ab, an, bn) = (ex, ey, ax, ay, X, Y) if x_is_a else (ey, ex, ay, ax, Y, X)
            res = {"ep_idx": ep["ep_idx"], "cmp": cmp, "x_is_a": x_is_a}
            for dim in DIMS:
                p, s = builder.build_pairwise_single_dim_prompt(ep, ea, eb, dim)
                try:
                    r, _ = jclient.label_one(p, s, f"{cmp}_{ep['ep_idx']}_{dim}")
                    w = parse_llm_json(r).get("winner", "Tie")
                    res[dim] = an if w == "A" else bn if w == "B" else "Tie"
                except Exception:
                    res[dim] = "Tie"
            try:
                p, s = builder.build_pairwise_faithfulness_prompt(ep, ea, aa, eb, ab)
                r, _ = jclient.label_one(p, s, f"{cmp}_{ep['ep_idx']}_faith")
                w = parse_llm_json(r).get("winner", "Tie")
                res["faithfulness"] = an if w == "A" else bn if w == "B" else "Tie"
            except Exception:
                res["faithfulness"] = "Tie"
            pw.append(res); json.dump(pw, open(ppath, "w"), indent=2)
            print(f"  judged ep={ep['ep_idx']} {cmp}: {{d:res[d] for d in DIMS+['faithfulness']}}".replace("{d:res[d] for d in DIMS+['faithfulness']}", str({d: res[d] for d in DIMS+['faithfulness']})))

    # Phase C: aggregate
    dims = DIMS + ["faithfulness"]
    agg = {}
    for X, Y in COMPARISONS:
        cmp = f"{X}_vs_{Y}"; rows = [r for r in pw if r["cmp"] == cmp]
        agg[cmp] = {d: {X: sum(1 for r in rows if r.get(d)==X), Y: sum(1 for r in rows if r.get(d)==Y),
                        "Tie": sum(1 for r in rows if r.get(d)=="Tie")} for d in dims}
    json.dump(agg, open(out / "aggregated.json", "w"), indent=2)
    print("\n=== WIN COUNTS ===")
    for cmp, dd in agg.items():
        print(f"\n{cmp}:")
        for d, c in dd.items():
            print(f"  {d:14s} {c}")


if __name__ == "__main__":
    main()
