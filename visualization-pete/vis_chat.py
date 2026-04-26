import re
import time
from pathlib import Path
from datetime import datetime


def log_to_chat_html():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent

    while project_root.parent != project_root:
        if (project_root / "out" / "debug.log").exists():
            break
        project_root = project_root.parent

    log_file = project_root / "out" / "debug.log"
    output_dir = project_root / "visualization-pete" / "chats"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%d.%m_%H-%M-%S")
    output_file = output_dir / f"chat_log_{timestamp}.html"

    if not log_file.exists():
        print(f"Fehler: Log-Datei {log_file} nicht gefunden.")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    ROLE_MAP = {
        "nodeAgent": ("AI Researcher", "👨‍🔬", "researcher"),
        "treesearch": ("Tree Search", "🧠", "planner"),
        "interpreter": ("Python Sandbox", "💻", "sandbox"),
        "backend/openai": ("LLM Backend", "🤖", "llm"),
        "main": ("System", "⚙️", "system"),
    }

    pattern = re.compile(r'\[(?P<ts>.*?)\] \[(?P<lvl>.*?)\] isgsa\.(?P<logger>.*?): (?P<msg>.*)')

    parsed = []

    for line in lines:
        m = pattern.search(line)
        if not m:
            continue

        d = m.groupdict()

        role_key = next((k for k in ROLE_MAP if k in d["logger"]), "main")
        name, icon, cls = ROLE_MAP[role_key]

        parsed.append({
            "time": d["ts"],
            "level": d["lvl"],
            "role": cls,
            "name": name,
            "icon": icon,
            "msg": d["msg"],
        })

    # --- TIME DELTAS ---
    prev_time = None
    for p in parsed:
        current = datetime.strptime(p["time"], "%Y/%m/%d %H:%M:%S")
        if prev_time:
            delta = (current - prev_time).total_seconds()
            p["delta"] = f"+{delta:.1f}s"
        else:
            p["delta"] = ""
        prev_time = current

    # --- GROUPING ---
    grouped = []
    current_group = None

    for p in parsed:
        if not current_group or current_group["role"] != p["role"]:
            current_group = {
                "role": p["role"],
                "name": p["name"],
                "icon": p["icon"],
                "messages": [p]
            }
            grouped.append(current_group)
        else:
            current_group["messages"].append(p)

    # --- STATS ---
    total = len(parsed)
    errors = sum(1 for p in parsed if p["level"] == "ERROR")
    warns = sum(1 for p in parsed if p["level"] == "WARN")

    # --- HTML ---
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Log Viewer</title>

<style>
body {{
    font-family: sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    padding: 20px;
}}

.controls {{
    margin-bottom: 20px;
}}

.msg {{
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 12px;
}}

.researcher {{ background: #1e293b; border-left: 4px solid #3b82f6; }}
.planner {{ background: #312e81; border-left: 4px solid #8b5cf6; }}
.llm {{ background: #064e3b; border-left: 4px solid #10b981; }}
.sandbox {{ background: #000; font-family: monospace; }}
.system {{ background: #1f2937; }}

.time {{
    font-size: 0.8em;
    color: #94a3b8;
}}

.delta {{
    color: #22c55e;
}}

details {{
    margin-top: 8px;
}}

pre {{
    background: #020617;
    padding: 10px;
    border-radius: 6px;
    overflow-x: auto;
}}

button {{
    margin-bottom: 5px;
    cursor: pointer;
}}

.highlight-error {{ background: #450a0a; }}
.highlight-success {{ background: #022c22; }}

.stats {{
    margin-bottom: 20px;
    padding: 10px;
    background: #020617;
}}
</style>

<script>
function toggleLevel(level) {{
    document.querySelectorAll("." + level).forEach(e => {{
        e.style.display = e.style.display === "none" ? "block" : "none";
    }});
}}

function copyCode(btn) {{
    const code = btn.nextElementSibling.innerText;
    navigator.clipboard.writeText(code);
}}
</script>

</head>
<body>

<div class="stats">
<b>Events:</b> {total} | ❌ Errors: {errors} | ⚠️ Warnings: {warns}
</div>

<div class="controls">
<label><input type="checkbox" checked onclick="toggleLevel('DEBUG')"> DEBUG</label>
<label><input type="checkbox" checked onclick="toggleLevel('INFO')"> INFO</label>
<label><input type="checkbox" checked onclick="toggleLevel('ERROR')"> ERROR</label>
</div>
"""

    for group in grouped:
        html += f"""
<div class="msg {group['role']}">
<div><b>{group['icon']} {group['name']}</b> ({len(group['messages'])} events)</div>
<details>
<summary>Details anzeigen</summary>
"""

        for m in group["messages"]:
            msg = m["msg"]

            # highlights
            extra = ""
            if "error" in msg.lower():
                extra = "highlight-error"
            elif "initialized" in msg.lower():
                extra = "highlight-success"

            # code blocks
            if "```" in msg:
                msg = re.sub(r'```(.*?)```', r'<button onclick="copyCode(this)">Copy</button><pre>\1</pre>', msg, flags=re.DOTALL)

            html += f"""
<div class="{m['level']} {extra}">
<div class="time">{m['time']} <span class="delta">{m['delta']}</span></div>
<div>{msg}</div>
</div>
"""

        html += "</details></div>"

    html += "</body></html>"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Fertig: {output_file}")


if __name__ == "__main__":
    log_to_chat_html()