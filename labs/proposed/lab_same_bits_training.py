import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full", app_title="The Same-Bits Contract, Part 2 · MLSysBook")


# ZONE A · SETUP
@app.cell
async def _():
    import sys
    from pathlib import Path
    import marimo as mo
    if sys.platform == "emscripten":
        import micropip
        await micropip.install(["pydantic", "pint", "plotly", "pandas", "numpy"], keep_going=False)
        await micropip.install("../../wheels/mlsysim-0.3.1-py3-none-any.whl", keep_going=False)
        await micropip.install("../../wheels/mlsysbook_labs-0.1.0-py3-none-any.whl", keep_going=False)
    else:
        labs_dir = Path(__file__).resolve().parents[1]
        if str(labs_dir) not in sys.path:
            sys.path.insert(0, str(labs_dir))
        from bootstrap import native_bootstrap
        native_bootstrap(__file__)
    import plotly.graph_objects as go
    from mlsysim.hardware.registry import Hardware
    from mlsysim.labs.experiments.same_bits_training_experiments import (
        COLLECTIVES, WORLD_SIZES, collective_contrast, collective_snapshot,
    )
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysbook_labs import ACADEMIC_LAB_CSS
    from mlsysbook_labs.experiment_evidence import audit_evidence, capture_evidence
    return (
        ACADEMIC_LAB_CSS, COLLECTIVES, COLORS, Hardware, LAB_CSS, WORLD_SIZES,
        apply_plotly_theme, audit_evidence, capture_evidence, collective_contrast,
        collective_snapshot, go, mo,
    )


@app.cell
def _(mo):
    get_evidence, set_evidence = mo.state({})
    return get_evidence, set_evidence


@app.cell
def _(Hardware):
    # Prototype: one track.  The full lab would offer the book's four tracks.
    track_id = "cloud"
    device = Hardware.Cloud.H100
    stakeholder = "Training infrastructure lead"
    why_bits_matter = (
        "A regression appears after the job moves from 4 to 8 GPUs. To tell a real bug "
        "from arithmetic, the team reruns the old configuration and expects the old weights, "
        "bit for bit. Whether it gets them depends on which AllReduce schedule ran."
    )
    return device, stakeholder, track_id, why_bits_matter


# ZONE B · WIDGETS
@app.cell
def _(COLLECTIVES, WORLD_SIZES, mo):
    a_collective = mo.ui.dropdown(
        {name.capitalize(): name for name in COLLECTIVES},
        value="Ring", label="Collective to compare with the star (root adds in rank order)",
    )
    a_ranks = mo.ui.dropdown(
        {f"{r} ranks": r for r in WORLD_SIZES}, value="8 ranks", label="Data-parallel world size",
    )
    a_prediction = mo.ui.radio(
        {"One: every AllReduce computes the same sum": "one",
         "Two": "two",
         "Three: each collective gives its own bits": "three"},
        label="Eight ranks reduce the same 4,096 float32 gradients with ring, tree and star AllReduce. How many different results?",
    ).form(submit_button_label="Lock Part A prediction")
    return a_collective, a_prediction, a_ranks


@app.cell
def _(a_collective, a_ranks, collective_contrast, collective_snapshot):
    a_all = collective_contrast(ranks=a_ranks.value)
    a_base = collective_snapshot("star", ranks=a_ranks.value)
    a_result = collective_snapshot(a_collective.value, ranks=a_ranks.value)
    a_sweep = [collective_contrast(ranks=r) for r in (2, 4, 8, 16)]
    return a_all, a_base, a_result, a_sweep


@app.cell
def _(
    a_base, a_collective, a_prediction, a_ranks, a_result, capture_evidence,
    mo, set_evidence, track_id,
):
    upstream = {"scenario": track_id}
    a_capture = mo.ui.button(
        label="Capture collective contrast", kind="success",
        disabled=a_prediction.value is None or a_collective.value == "star",
        on_click=lambda _v: set_evidence(lambda current: {**current, "A": capture_evidence(
            track=track_id, part="A", prediction=a_prediction.value,
            inputs={"collective": a_collective.value, "ranks": a_ranks.value},
            baseline=a_base, result=a_result, upstream_inputs=upstream,
            model_key="same_bits_training_experiments.collective_snapshot",
        )}),
    )
    return a_capture, upstream


@app.cell
def _(ACADEMIC_LAB_CSS, LAB_CSS, device, mo, stakeholder):
    css = mo.Html("""
    <style>
    .pilot-head{background:linear-gradient(135deg,#101827,#1d4f78);color:white;border-radius:14px;padding:clamp(18px,4vw,32px);margin-bottom:14px}.pilot-top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font:700 .72rem ui-monospace;letter-spacing:.08em}.pilot-head h1{font-size:clamp(1.65rem,5vw,2.65rem);line-height:1.05;margin:16px 0 8px}.pilot-head p{color:#dbeafe;max-width:760px}.pilot-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:9px;margin-top:17px}.pilot-meta div{background:#ffffff14;border:1px solid #ffffff26;padding:9px 11px;border-radius:8px}.pilot-note{color:#475569;font-size:.9rem;line-height:1.5;margin:0;padding:0 2px}.stakeholder{border-left:4px solid #1d4f78;background:#f1f5f9;padding:10px 14px;border-radius:7px;line-height:1.5}.saved{border-left:4px solid #2ca02c;background:#f0fdf4;padding:9px 12px;border-radius:7px}@media(max-width:520px){.pilot-head{border-radius:9px}.pilot-meta{grid-template-columns:1fr}}
    </style>""")
    header = mo.Html(f"""<section class="pilot-head"><div class="pilot-top"><span>PROPOSED CO-LAB · DISTRIBUTED TRAINING · PART A PROTOTYPE</span><span>ABOUT 8 MIN</span></div><h1>The Same-Bits Contract, Part 2</h1><p>You run the same data-parallel job twice, with the same seed and the same data, and get different weights. Nothing random was sampled. Where did the difference come from, and what does it cost to remove it?</p><div class="pilot-meta"><div><b>Device</b><br>{device.name}</div><div><b>Stakeholder</b><br>{stakeholder}</div><div><b>Number format</b><br>float32 gradients</div><div><b>Deliverable</b><br>A reproducibility contract</div></div></section>""")
    mo.vstack([LAB_CSS, ACADEMIC_LAB_CSS, css, header, mo.Html('<p class="pilot-note">Every number is computed with real IEEE 754 float32 arithmetic, rounded after each addition, with each AllReduce schedule\'s order written out. The gradients are seeded synthetic values, so counts illustrate the mechanism rather than measure a production model.</p>')], gap=0.5)
    return


# ZONE C · TABS
@app.cell
def _(
    COLORS, a_all, a_base, a_capture, a_collective, a_prediction, a_ranks,
    a_result, a_sweep, apply_plotly_theme, audit_evidence, get_evidence, go,
    mo, stakeholder, track_id, upstream, why_bits_matter,
):
    _captures = get_evidence()
    audit = audit_evidence(
        _captures, track=track_id, required_parts=("A",),
        upstream_inputs=upstream, contrast_required_parts=("A",),
    )

    def table(rows):
        return mo.vstack([mo.ui.table(rows, pagination=False)]).style({"max-width": "100%", "overflow-x": "auto"})

    def saved(part):
        capture = _captures.get(part)
        if capture is None:
            return mo.callout(mo.md("No saved evidence for this part."), kind="warn")
        if part in audit.stale or (part, part) in audit.identical_pairs:
            return mo.callout(mo.md("**STALE OR NON-CONTRASTING EVIDENCE.** Review and recapture."), kind="danger")
        data = capture.to_dict()
        return mo.Html(f'<div class="saved"><b>Saved snapshot</b> · original prediction: {data["prediction"]}<br><small>Later control changes do not rewrite this record.</small></div>')

    def build_part_a():
        intro = mo.md("### A · One gradient, three collectives (8 min)\n**Driving question:** Do ring, tree and star AllReduce give the same bits for the same gradients?")
        if a_prediction.value is None:
            return mo.vstack([intro, a_prediction])
        rows = [
            {"Collective": r["collective"], "Bits (digest)": r["bits"],
             "Parameters differing from star": f'{r["params_differing_from_star"]:,} / {a_all["params"]:,}',
             "Max ULPs from star": r["max_ulps_from_star"]}
            for r in a_all["rows"]
        ]
        fig = go.Figure([go.Bar(
            x=[f'{s["inputs"]["ranks"]} ranks' for s in a_sweep],
            y=[s["params_where_collectives_disagree"] for s in a_sweep],
            marker_color=COLORS["RedLine"],
        )])
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=25, b=20), yaxis_title="Parameters where the collectives disagree")
        return mo.vstack([
            intro,
            mo.Html(f'<div class="stakeholder"><b>{stakeholder}:</b> {why_bits_matter}</div>'),
            a_ranks, a_prediction, table(rows),
            mo.callout(mo.md(
                f"**Your prediction:** {a_prediction.value}. **Measured at {a_ranks.value} ranks:** "
                f"**{a_all['distinct_results']}** distinct results from 3 collectives; they disagree on "
                f"**{a_all['params_where_collectives_disagree']:,} of {a_all['params']:,}** parameters. "
                f"Each collective repeats its own bits exactly when run again: {'yes' if a_all['repeatable'] else 'no'}."
            ), kind="info"),
            apply_plotly_theme(fig),
            a_collective,
            mo.md(f"Selected: **{a_result['inputs']['collective']}** differs from the star (bits {a_base['bits']}) on "
                  f"**{a_result['params_differing_from_star']:,}** parameters, by up to {a_result['max_ulps_from_star']} ULPs."),
            a_capture, saved("A"),
            mo.accordion({"Calculation Notes": mo.md(
                "Every rank holds float32 gradients for the same 4,096 parameters. The ring "
                "reduce-scatter sums chunk c starting at rank c + 1, so each chunk is added in "
                "its own rotation; the tree adds pairs of ranks, then pairs of pairs; the star "
                "adds ranks 0, 1, 2, ... at one root. Each rounds to float32 after every "
                "addition. With two ranks every schedule computes a + b, so they agree; from "
                "three ranks on, the orders differ."
            )}),
        ])

    tabs = mo.ui.tabs({
        "Part A": build_part_a(),
        "Parts B–E": mo.md("Parts B–E (world size, drift over training, fixed reduction trees, integer accumulation) and the synthesis follow once Part A has been reviewed."),
    })
    tabs
    return (audit,)


if __name__ == "__main__":
    app.run()
