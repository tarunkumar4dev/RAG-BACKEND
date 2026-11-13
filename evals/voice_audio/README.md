# Real phone recordings for the audio eval

Put recordings here (any container the mic produces: `.webm`, `.m4a`/`.mp4`, `.ogg`, `.wav`) and list each one in
`manifest.yaml`:

```yaml
- {file: v01.webm, text: "Sharma ma'am ki is mahine ki attendance kya hai", role: admin, entities: ["Priya Sharma"]}
- {file: v02.m4a, text: "Attendance kholo", role: teacher, entities: []}
```

`text` is what was actually said, written the way the transcript should come back (Roman script for Hindi).
Recordings stay out of git: only this README and `manifest.yaml` are tracked (see `.gitignore` in this folder).
Get consent from whoever is recorded; use seed names (Priya Sharma, Rakesh Verma, 10A, 10B, 9A, Rahul Kumar), never
real students.

Run (PowerShell, keys in the same window):

```powershell
$env:AGENT_DB="embedded"; agent\.devdata\venv\Scripts\python evals\run_voice_audio_eval.py --provider sarvam --noise C:\path\corridor.wav
```

Noise mixing at 20 / 10 / 5 dB SNR needs `ffmpeg` on PATH. Without a key or without recordings the script prints
"audio eval NOT run" and exits 2.
