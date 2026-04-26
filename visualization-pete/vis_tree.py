"""
DIESES SKRIPT VISUALISIERT DEN SUCHBAUM DER TREESEARCH (AUS SAVE.PKL).
Optimiert für die Struktur: AutoRecLab-group5/visualization-pete/vis_tree.py
"""

import pickle
import os
import time
from graphviz import Digraph
from pathlib import Path

def visualize_ultimate():
    # 1. Dynamische Pfad-Ermittlung
    script_path = Path(__file__).resolve()

    # Wir gehen 2 Ebenen hoch:
    # 1. Ebene: visualization-pete/ -> Hauptverzeichnis
    # 2. Ebene: Falls vis_tree.py in einem Unterordner von visualization-pete liegt
    # Wir suchen den Ordner, der "out" als Unterordner hat:
    project_root = script_path.parent
    while not (project_root / "out").exists() and project_root.parent != project_root:
        project_root = project_root.parent

    pickle_path = project_root / "out" / "save.pkl"
    output_dir = project_root / "visualization-pete" / "trees"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Zeitstempel für den Dateinamen
    timestamp = time.strftime("%d.%m_%H:%M")
    filename = f"tree_search_{timestamp}"
    full_output_path = output_dir / filename

    print(f"Projekt-Root gefunden: {project_root}")
    print(f"Suche save.pkl in: {pickle_path}")

    if not pickle_path.exists():
        print(f"Fehler: '{pickle_path}' wurde nicht gefunden!")
        return

    # 2. Daten laden
    with open(pickle_path, "rb") as f:
        try:
            # Wir wechseln kurz das Arbeitsverzeichnis für Pickle
            old_cwd = os.getcwd()
            os.chdir(project_root)
            roots = pickle.load(f)
            os.chdir(old_cwd)
        except Exception as e:
            print(f"Fehler beim Laden der Pickle-Datei: {e}")
            return

    # 3. Graph-Initialisierung
    dot = Digraph(comment='AutoRecLab Tree Visualization')
    dot.attr(rankdir='LR', splines='curved', nodesep='0.6', ranksep='1.0', bgcolor='#fcfcfc')
    dot.attr('node', shape='none', fontname='Helvetica,Arial,sans-serif')

    def add_node(node):
        if node.is_buggy:
            bg, border, text, subtext = "#ffeded", "#ff5c5c", "#a80000", "#d65a5a"
            icon = "🛠️ "
        elif node.score.score >= 20:
            bg, border, text, subtext = "#e6fffa", "#38b2ac", "#234e52", "#319795"
            icon = "🚀 "
        else:
            bg, border, text, subtext = "#f0fff4", "#68d391", "#22543d", "#38a169"
            icon = "📈 "

        html_label = f'''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="10" BGCOLOR="{bg}" COLOR="{border}" STYLE="ROUNDED">
            <TR><TD BORDER="0" ALIGN="LEFT"><FONT POINT-SIZE="11" COLOR="{text}"><B>{icon} {node.stage_name.upper()}</B></FONT></TD></TR>
            <TR><TD BORDER="0" ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="{subtext}">ID: {node.id[:6]}</FONT></TD></TR>
            <TR><TD BORDER="0" ALIGN="CENTER"><FONT POINT-SIZE="18" COLOR="{text}"><B>{node.score.score}%</B></FONT></TD></TR>
        </TABLE>>'''

        dot.node(node.id, label=html_label)

        if node._parent:
            dot.edge(node._parent.id, node.id, color="#cbd5e0", penwidth="1.5", arrowhead="vee")

        for child in node.children:
            add_node(child)

    for root in roots:
        add_node(root)

    # 5. Rendern
    try:
        render_result = dot.render(str(full_output_path), format='png', cleanup=True)
        print("-" * 50)
        print(f"VISUALISIERUNG ERFOLGREICH")
        print(f"Gespeichert in: {render_result}")
        print("-" * 50)
    except Exception as e:
        print(f"Fehler beim Rendern: {e}")

if __name__ == "__main__":
    visualize_ultimate()