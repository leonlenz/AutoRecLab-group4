#Tracks the differents and statistics from node to node

from treesearch.node import Node
from utils.log import _ROOT_LOGGER
import os
import ast
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = _ROOT_LOGGER.getChild("statistics")


class StatisticNode:

    def __init__(self):
        self.out_dir : str = None
        self.checkpoint_dir : str = None
        self.last_node : Node = None

        self.id : str = ""
        self.position : int = 0

        self.score : float = 0.0
        self.is_buggy : bool = False
        self.is_satisfactory : bool = False
        self.exec_time : float = 0.0

        self.loc = 0
        self.empty_lines = 0
        self.comment_lines = 0
        self.total_characters = 0
        self.libraries_imported = []
        self.sum_libraries_imported = 0
        self.sum_functions = 0
        self.sum_classes = 0
        self.avg_ags_per_function = 0.0
        self.variable_assignments = 0
        self.sum_loops = 0
        self.sum_conditions = 0
        self.insertions = 0
        self.deletions = 0

    def analyze_code(self):
        code_file = os.path.join(self.checkpoint_dir, self.id, "code.py")
        try:
            logger.debug("Analyzing code for statistic from node " + self.id + " at file: " + code_file)

            with open(code_file, "r", encoding="utf-8") as f:
                code_content = f.read()

            lines = code_content.splitlines()
            self.loc = len(lines)
            self.total_characters = len(code_content)
            self.empty_lines = sum(1 for l in lines if not l.strip())
            self.comment_lines = sum(1 for l in lines if l.strip().startswith("#"))

            ast_tree = ast.parse(code_content)
            imports = set()
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
            self.libraries_imported = list(imports)
            self.sum_libraries_imported = len(self.libraries_imported)

            sum_functions   = [n for n in ast.walk(ast_tree) if isinstance(n, ast.FunctionDef)]
            sum_classes     = [n for n in ast.walk(ast_tree) if isinstance(n, ast.ClassDef)]
            self.avg_ags_per_function    = (sum(len(f.args.args) for f in sum_functions) / len(sum_functions) if sum_functions else 0)
            self.variable_assignments = sum(1 for n in ast.walk(ast_tree) if isinstance(n, ast.Assign))
            self.sum_loops       = sum(1 for n in ast.walk(ast_tree) if isinstance(n, (ast.For, ast.While)))
            self.sum_conditions  = sum(1 for n in ast.walk(ast_tree) if isinstance(n, ast.If))
            self.sum_functions   = len(sum_functions)
            self.sum_classes     = len(sum_classes)

            self.insertions = self.loc
            self.deletions = 0
            if self.last_node is not None:
                code_file_last = os.path.join(self.checkpoint_dir, self.last_node.id, "code.py")
                self.insertions, self.deletions = self.git_diff_stat(Path(code_file_last), Path(code_file))

        except Exception as e:
            logger.warning("Could not analyze code for statistic from node " + self.id + ": " + str(e))
            return

    def normalize_file(self, path: Path):  
        text = path.read_text().rstrip("\n")
        path.write_text(text + "\n")
        
    def git_diff_stat(self, file1: Path, file2: Path):
        self.normalize_file(file1)
        self.normalize_file(file2)

        proc = subprocess.run(
            ["git", "diff", "--no-index", "--numstat", file1, file2],
            capture_output=True,
            text=True,
        )

        s = proc.stdout.split("\t")
        if len(s) < 2:
            return 0, 0
        insertions = int(s[0])
        deletions = int(s[1])
        return insertions, deletions

    def save_to_file(self):
        if self.out_dir is None:
            return
        file_path = os.path.join(self.out_dir, str(self.position) + "_" + self.id + ".txt")
        with open(file_path, "w") as f:
            f.write("ID: " + self.id + "\n")
            f.write("Position: " + str(self.position) + "\n")
            f.write("Score: " + str(self.score) + "\n")
            f.write("Is Buggy: " + str(self.is_buggy) + "\n")
            f.write("Is Satisfactory: " + str(self.is_satisfactory) + "\n")
            f.write("Execution Time: " + str(self.exec_time) + "\n")
            f.write("Lines of Code: " + str(self.loc) + "\n")
            f.write("Empty Lines: " + str(self.empty_lines) + "\n")
            f.write("Comment Lines: " + str(self.comment_lines) + "\n")
            f.write("Total Characters: " + str(self.total_characters) + "\n")
            f.write("Libraries Imported: " + ", ".join(self.libraries_imported) + "\n")
            f.write("Count of Libraries Imported: " + str(self.sum_libraries_imported) + "\n")
            f.write("Functions: " + str(self.sum_functions) + "\n")
            f.write("Classes: " + str(self.sum_classes) + "\n")
            f.write("Average Arguments per Function: " + str(self.avg_ags_per_function) + "\n")
            f.write("Variable Assignments: " + str(self.variable_assignments) + "\n")
            f.write("Loops: " + str(self.sum_loops) + "\n")
            f.write("Conditions: " + str(self.sum_conditions) + "\n")
            f.write("Insertions: " + str(self.insertions) + "\n")
            f.write("Deletions: " + str(self.deletions) + "\n")

        logger.debug("Saved statistic for node " + self.id + " to file: " + file_path)



class StatisticsTracker:

    def __init__(self):
        self.out_dir = None
        self.checkpoint_dir = None
        self.nodes_ordered = []

    def reset(self):
        """Reset the tracker state so it can be reused for a fresh run."""
        self.out_dir = None
        self.checkpoint_dir = None
        self.nodes_ordered = []

    def set_out_dir(self, out_dir):
        self.checkpoint_dir = os.path.join(out_dir, "checkpoint")

        stats_folder = os.path.join(out_dir, "statistics")
        os.makedirs(stats_folder, exist_ok=True)
        self.out_dir = stats_folder


    def add_node(self, node: Node):
        new_node = StatisticNode()
        new_node.out_dir = self.out_dir
        new_node.checkpoint_dir = self.checkpoint_dir
        new_node.id = node.id
        new_node.score = node.score.score
        new_node.is_buggy = node.is_buggy
        new_node.is_satisfactory = node.score.is_satisfactory
        new_node.exec_time = node.exec_time
        new_node.position = len(self.nodes_ordered)

        if(len(self.nodes_ordered) > 0):
            new_node.last_node = self.nodes_ordered[-1]
        
        self.nodes_ordered.append(new_node)

        new_node.analyze_code()
        new_node.save_to_file()

    def get_avgs(self):
        avgs = {
            "score": 0, "exec_time": 0, "loc": 0, "empty_lines": 0,
            "comment_lines": 0, "total_characters": 0, "sum_libraries_imported": 0,
            "sum_functions": 0, "sum_classes": 0, "avg_ags_per_function": 0,
            "variable_assignments": 0, "sum_loops": 0, "sum_conditions": 0,
            "insertions": 0, "deletions": 0
        }

        for n in self.nodes_ordered:
            avgs["score"] += n.score
            avgs["exec_time"] += n.exec_time
            avgs["loc"] += n.loc
            avgs["empty_lines"] += n.empty_lines
            avgs["comment_lines"] += n.comment_lines
            avgs["total_characters"] += n.total_characters
            avgs["sum_libraries_imported"] += n.sum_libraries_imported
            avgs["sum_functions"] += n.sum_functions
            avgs["sum_classes"] += n.sum_classes
            avgs["avg_ags_per_function"] += n.avg_ags_per_function
            avgs["variable_assignments"] += n.variable_assignments
            avgs["sum_loops"] += n.sum_loops
            avgs["sum_conditions"] += n.sum_conditions
            avgs["insertions"] += n.insertions
            avgs["deletions"] += n.deletions

        for k in avgs.keys():
            avgs[k] /= len(self.nodes_ordered)

        return avgs

    def generate_plot(self, label, get_value):
        values = [get_value(n) for n in self.nodes_ordered]
        x = list(range(len(self.nodes_ordered)))
        checkpoint_labels = [f"{n.position}_{n.id[:6]}" for n in self.nodes_ordered]
        avg_val = sum(values) / len(values) if values else 0
        key = label.lower().replace(" ", "_")

        fig, ax = plt.subplots(figsize=(max(8, len(values) * 0.5), 5))

        ax.plot(x, values, color="#4C72B0", linewidth=2, zorder=2)
        ax.scatter(x, values, color="#4C72B0", s=60, zorder=3)

        ax.axhline(avg_val, color="#DD4444", linewidth=1.5, linestyle="--", label=f"Avg: {avg_val:.1f}")

        for xi, val in zip(x, values):
            ax.text(
                xi,
                val + (max(values) * 0.03 if max(values) > 0 else 0.03),
                f"{val:.1f}" if isinstance(val, float) else str(val),
                ha="center", va="bottom", fontsize=8, color="#222222"
            )

        ax.set_xticks(x)
        ax.set_xticklabels(checkpoint_labels, fontsize=8, rotation=45, ha="right")
        ax.set_xlabel("Checkpoint", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        png_path = os.path.join(self.out_dir, f"{key}.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.debug(f"Statistic plot saved for {label} → {png_path}")


    def summarize_statistics(self):
        if len(self.nodes_ordered) == 0:
            logger.warning("No nodes to summarize statistics for.")
            return
        
        avgs = self.get_avgs()

        #Save summary statistics to file
        summary_file = os.path.join(self.out_dir, "summary_overall_nodes.txt")
        with open(summary_file, "w") as f:
            f.write("Total Nodes: " + str(len(self.nodes_ordered)) + "\n")
            f.write("Average Score: " + str(avgs["score"]) + "\n")
            f.write("Average Execution Time: " + str(avgs["exec_time"]) + "\n")
            f.write("Average Lines of Code: " + str(avgs["loc"]) + "\n")
            f.write("Average Empty Lines: " + str(avgs["empty_lines"]) + "\n")
            f.write("Average Comment Lines: " + str(avgs["comment_lines"]) + "\n")
            f.write("Average Total Characters: " + str(avgs["total_characters"]) + "\n")
            f.write("Average Count of Libraries Imported: " + str(avgs["sum_libraries_imported"]) + "\n")
            f.write("Average Functions: " + str(avgs["sum_functions"]) + "\n")
            f.write("Average Classes: " + str(avgs["sum_classes"]) + "\n")
            f.write("Average Arguments per Function: " + str(avgs["avg_ags_per_function"]) + "\n")
            f.write("Average Variable Assignments: " + str(avgs["variable_assignments"]) + "\n")
            f.write("Average Loops: " + str(avgs["sum_loops"]) + "\n")
            f.write("Average Conditions: " + str(avgs["sum_conditions"]) + "\n")
            f.write("Average Insertions: " + str(avgs["insertions"]) + "\n")
            f.write("Average Deletions: " + str(avgs["deletions"]) + "\n")

        #Generate plots
        self.generate_plot("Execution_Time", lambda n: n.exec_time)
        self.generate_plot("Lines_of_Code", lambda n: n.loc)
        self.generate_plot("Empty_Lines", lambda n: n.empty_lines)
        self.generate_plot("Comment_Lines", lambda n: n.comment_lines)
        self.generate_plot("Total_Characters", lambda n: n.total_characters)
        self.generate_plot("Libraries_Imported", lambda n: n.sum_libraries_imported)
        self.generate_plot("Functions", lambda n: n.sum_functions)
        self.generate_plot("Classes", lambda n: n.sum_classes)
        self.generate_plot("Arguments_per_Function", lambda n: n.avg_ags_per_function)
        self.generate_plot("Variable_Assignments", lambda n: n.variable_assignments)
        self.generate_plot("Loops", lambda n: n.sum_loops)
        self.generate_plot("Conditions", lambda n: n.sum_conditions)
        self.generate_plot("Insertions", lambda n: n.insertions)
        self.generate_plot("Deletions", lambda n: n.deletions)




_STATISTICS_TRACKER = StatisticsTracker()
def get_statistics_tracker():
    return _STATISTICS_TRACKER