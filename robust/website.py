"""
Write the project page for the personal website from the report's numbers.

Every number on the page comes from paper/numbers.tex, so the page and the
report cannot disagree.

Usage:
  python -m robust.website /Users/jessie/personal-website/projects/burn-severity
"""
import re
import sys
from pathlib import Path

from robust.common import ROOT


def macros() -> dict[str, str]:
    text = (ROOT / "paper" / "numbers.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text))


PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Burn Severity on Unseen Fires — Jessie Dong</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap">
    <link rel="stylesheet" href="../../style.css">
  </head>
  <body>
    <div class="masthead">
      <h1><a href="../../">Jessie Dong</a></h1>
      <nav>
        <a href="../../">Home</a>
      </nav>
    </div>

    <main class="project-page">
      <p class="back"><a href="../../">&larr; Projects</a></p>

      <h1 class="project-title">Burn Severity on Fires the Model Has Never Seen</h1>
      <p class="meta">Jessie Dong &middot; Stanford &middot; 2026</p>
      <p class="links"><a href="report.pdf">Report (PDF)</a><a href="https://github.com/jessiedong01/burn-severity-unseen-wildfires">Code (GitHub)</a></p>

      <p class="lede">After a wildfire, can a model map how badly each patch of land burned, on a fire it has never seen?</p>

      <p>Burn severity maps tell land managers where to put erosion control, which slopes may send debris flows down in the first winter rains, and where forests need replanting. In the U.S., the MTBS program makes them. An analyst compares Landsat images from before and after the fire, computes a burn index called dNBR, and picks the cutoffs between unburned, low, moderate, and high severity for that one fire.</p>

      <p>I asked how close an automatic method can get to the analyst's map, using the same two images, on a fire it was never trained on.</p>

      <div class="figure wide">
        <a href="maps.png"><img src="maps.png" alt="Severity maps for three held-out fires: post-fire image, MTBS ground truth, dNBR with generic thresholds, dNBR with learned thresholds, and U-Net" loading="lazy"></a>
        <p class="caption">Three fires, each predicted by a model that never saw it. Tap or click to enlarge. Generic thresholds call too much of the fire high severity. Learned thresholds and the U-Net land close to the MTBS map.</p>
      </div>

      <h2>What I did</h2>
      <ul>
        <li>Took {mFires} California and Oregon wildfires from {mYearMin}–{mYearMax}, {mAcresM} million acres in all, with MTBS's own severity maps as ground truth.</li>
        <li>Used the exact Landsat scenes MTBS used, on MTBS's own 30 m grid, so labels are never resampled.</li>
        <li>Held out whole fires. Every fire is tested by a model trained on other fires, and fires that overlap stay together.</li>
        <li>Compared a fixed-threshold index, thresholds learned from other fires, a per-pixel gradient-boosting model, and a U-Net. An "oracle" that uses each fire's own analyst thresholds shows the ceiling.</li>
      </ul>

      <h2>Results</h2>
      <table class="results">
        <thead><tr><th>Method</th><th>mIoU</th></tr></thead>
        <tbody>
          <tr><td>dNBR, generic thresholds</td><td>{mdnbrgenericMiou}</td></tr>
          <tr><td>Per-pixel gradient boosting</td><td>{mgbmMiou}</td></tr>
          <tr><td>dNBR, thresholds learned from other fires</td><td>{mdnbrlearnedMiou}</td></tr>
          <tr class="best"><td>U-Net</td><td>{mBestUnetMiou}</td></tr>
          <tr><td>Oracle: each fire's own analyst thresholds</td><td>{moracleMiou}</td></tr>
        </tbody>
      </table>
      <p class="footnote">Mean intersection-over-union across the {mFires} held-out fires; higher is better.</p>

      <h2>What I found</h2>
      <ul>
        <li><strong>Calibration is the big lever.</strong> Learning the cutoffs from other fires instead of using generic ones raised mIoU by {mDiffLearnGenAbs}, and it helped on {mWinsLearnGen} of {mFires} fires.</li>
        <li><strong>The U-Net ties a well-calibrated index.</strong> It scored {mBestUnetMiou} against {mdnbrlearnedMiou}, a difference of {mDiffLearned} with a 95% interval from {mDiffLearnedLo} to {mDiffLearnedHi}. Raw bands and spatial context added nothing I could measure.</li>
        <li><strong>The rest of the gap is fire-specific.</strong> The analyst's own cutoffs score {moracleMiou}. Their high-severity cutoff ranges from {mHighMin} to {mHighMax} across fires, so no single set fits every fire.</li>
      </ul>

      <div class="figure">
        <img src="thresholds.png" alt="Analyst dNBR thresholds for each fire against the generic values" loading="lazy">
        <p class="caption">The cutoffs MTBS analysts chose for each fire, against the generic values (dashed). Every analyst put the high-severity cutoff above the generic 440.</p>
      </div>

      <h2>What changed from the first version</h2>
      <p>My first version reported a U-Net mIoU of 0.756. When I rebuilt the project, I found that number didn't measure what I thought. Its labels came from dNBR, a formula of the model's own inputs, and the model was scored on fires it had trained on. Those labels also matched the official Camp Fire map on only {mCampOrigAgree}% of pixels. This version scores everything against the analysts' maps, on fires the model has never seen. The numbers are lower, and they are real.</p>

      <h2>What's next</h2>
      <p>The oracle points at the next idea: instead of predicting each pixel, predict each fire's cutoffs from its vegetation, terrain, and imagery. That targets exactly the gap no method here closes.</p>
    </main>
  </body>
</html>
"""


def main() -> None:
    out = Path(sys.argv[1])
    m = {k: re.sub(r"\\ensuremath\{(.*)\}", r"\1", v) for k, v in macros().items()}
    m["mDiffLearnGenAbs"] = m["mDiffLearnGen"].lstrip("+")
    html = PAGE
    for key in re.findall(r"\{(m\w+)\}", PAGE):
        html = html.replace("{" + key + "}", m[key])
    html = html.replace("–", "–")
    (out / "index.html").write_text(html)
    left = re.findall(r"\{m\w+\}", html)
    assert not left, left
    print(f"wrote {out / 'index.html'}")


if __name__ == "__main__":
    main()
