import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.log import _ROOT_LOGGER

logger = _ROOT_LOGGER.getChild("type_checker")


@dataclass
class TypeCheckResult:
    has_errors: bool
    error_count: int
    errors: list[dict]
    raw_output: str
    full_output: str = ""

    def format_errors_for_llm(self) -> str:
        """Return the full rustc-like ty output for the LLM"""
        if not self.has_errors:
            return "No type checking errors found."

        return (
            f"Found {self.error_count} type checking error(s).\n"
            f"Full ty output:\n\n{self.full_output}"
        )


class TypeChecker:
    """Wrapper for ty type checker to validate Python code"""

    def __init__(self, working_dir: Path | str):
        """
        Initialize the type checker

        Args:
            working_dir: The directory where code files will be type-checked
        """
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), (
            f"Working directory {self.working_dir} does not exist"
        )

    def check_code(self, code: str, file_name: str = "runfile.py") -> TypeCheckResult:
        """
        Run ty type checker on the provided code

        Args:
            code: The Python code to type check
            file_name: The name of the file (this is just for error reporting)

        Returns:
            TypeCheckResult: Object containing type checking results
        """
        file_path = self.working_dir / file_name
        logger.debug(f"Type checking code in: {file_path}")

        with open(file_path, "w") as f:
            f.write(code)

        concise_cmd = ["uv", "run", "ty", "check", "--no-progress", "--output-format", "concise", str(file_path)]
        full_cmd = ["uv", "run", "ty", "check", "--no-progress", str(file_path)]

        try:
            result = subprocess.run(
                concise_cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            raw_output = result.stdout + result.stderr
            logger.debug(f"ty output: {raw_output}")

            # Run full format for rich LLM context
            full_result = subprocess.run(
                full_cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            full_output = full_result.stdout + full_result.stderr
            logger.debug(f"ty full output: {full_output}")

            # Parse text output
            errors = []
            error_count = 0

            try:
                for line in result.stdout.strip().split("\n"):
                    if line.strip() and ("error[" in line.lower() or "warning[" in line.lower()):
                        # Try to parse the line: file:line:col: error[rule] message
                        parts = line.split(":", 3)
                        if len(parts) >= 4:
                            file = parts[0].strip() if parts[0].strip() else file_name
                            try:
                                line_num = int(parts[1].strip())
                            except ValueError:
                                line_num = 0
                            try:
                                col_num = int(parts[2].strip())
                            except ValueError:
                                col_num = 0
                            
                            # The rest is "error[rule] message" or "warning[rule] message"
                            rest = parts[3].strip()
                            
                            severity = "error"
                            rule = ""
                            message = rest
                            
                            # Parse "error[rule-name]" or "warning[rule-name]"
                            if rest.lower().startswith("error["):
                                severity = "error"
                                error_count += 1
                                # Extract rule name and message
                                rule_end = rest.find("]")
                                if rule_end > 0:
                                    rule = rest[6:rule_end]  # Extract text between "error[" and "]"
                                    message = rest[rule_end + 1:].strip()
                            elif rest.lower().startswith("warning["):
                                severity = "warning"
                                rule_end = rest.find("]")
                                if rule_end > 0:
                                    rule = rest[8:rule_end]  # Extract text between "warning[" and "]"
                                    message = rest[rule_end + 1:].strip()
                            
                            errors.append(
                                {
                                    "file": file,
                                    "line": line_num,
                                    "column": col_num,
                                    "message": message,
                                    "rule": rule,
                                    "severity": severity,
                                }
                            )
            except Exception as e:
                logger.warning(f"Failed to parse ty output: {e}")
                # Fallback: check return code and stderr
                if result.returncode != 0:
                    error_count = max(1, result.stderr.count("error") + result.stdout.count("error:"))
                    if not errors:  # Only add fallback error if we didnt parse any
                        errors = [
                            {
                                "file": file_name,
                                "line": 0,
                                "column": 0,
                                "message": raw_output,
                                "rule": "parse-error",
                                "severity": "error",
                            }
                        ]

            has_errors = error_count > 0 or result.returncode != 0

            return TypeCheckResult(
                has_errors=has_errors,
                error_count=error_count,
                errors=errors,
                raw_output=raw_output,
                full_output=full_output,
            )

        except subprocess.TimeoutExpired:
            logger.error("Type checking timed out")
            return TypeCheckResult(
                has_errors=True,
                error_count=1,
                errors=[
                    {
                        "file": file_name,
                        "line": 0,
                        "column": 0,
                        "message": "Type checking timed out after 60 seconds",
                        "rule": "timeout",
                        "severity": "error",
                    }
                ],
                raw_output="Timeout",
            )
        except Exception as e:
            logger.error(f"Type checking failed with exception: {e}")
            return TypeCheckResult(
                has_errors=True,
                error_count=1,
                errors=[
                    {
                        "file": file_name,
                        "line": 0,
                        "column": 0,
                        "message": f"Type checking failed: {str(e)}",
                        "rule": "exception",
                        "severity": "error",
                    }
                ],
                raw_output=str(e),
            )
