# README

Complete this README file before you hand in the pset.

# Collaboration

I did not collaborate with a partner.

# Creative Part

## Goal

My goal for this creative part is to let the user perform with one keyboard layout
while switching between multiple instrument-like timbres in real time.

The design is based on a simple timbre research model:
- Harmonic balance: each preset uses different overtone mixtures.
- Envelope shape: attack/decay profiles emulate articulation differences.
- Brightness filtering: a low-pass filter mimics resonant body coloration.

The harmonic presets were recalibrated to better match common acoustic
descriptions of woodwinds:
- Flute: fundamental-heavy with steep high-partial rolloff.
- Clarinet: odd-harmonic dominance with weaker even partials.
- Oboe: richer high-mid partials and brighter reed tone.
- Tenor sax: broad, rich harmonic spectrum with moderate brightness.
- Bassoon: darker spectrum with stronger low-mid emphasis.

## Instructions

Run the creative widget:

`python pset1.py 2`

Controls:
- Notes: `a s d f g h j k` (C major from C4 to C5)
- Record toggle: `z`
- Mix gain: `up` / `down`
- Stop most recent note: `x`

Timbre switching:
- `shift`: tenor sax-like timbre
- `tab`: flute-like timbre
- `caps lock`: clarinet-like timbre
- `backspace`: bassoon-like timbre
- `enter`: oboe-like timbre

Fallback timbre switches (more reliable across platforms):
- `1`: flute
- `2`: oboe
- `3`: clarinet
- `4`: tenor sax
- `5`: bassoon
- `c`: cycle through presets

Preset model details:
- Flute-like: fundamental-dominant spectrum, soft attack, long decay, dark filter.
- Oboe-like: stronger upper partials, reed-forward brightness, medium attack/decay.
- Clarinet-like: odd-harmonic dominant spectrum, warm filter, stable envelope.
- Tenor sax-like: dense harmonic content, slightly bright filter, quick response.
- Bassoon-like: low-mid weighted spectrum, darker filter, slower articulation.

## Demo video

Link to the demo video:
<!-- Add your URL here -->
