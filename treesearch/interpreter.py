"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import os
import subprocess
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from time import time

import humanize
from dataclasses_json import DataClassJsonMixin

from utils.log import _ROOT_LOGGER

logger = _ROOT_LOGGER.getChild("interpreter")


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    has_exception: bool


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,  # Currently unused, left compat
        agent_file_name: str = "runfile.py",
        env_vars: dict[str, str] = {},
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
            env_vars (dict[str, str], optional): Environment variables to set in the child process. Defaults to {}.
        """
        # this really needs to be a path, otherwise causes issues that don't raise exc
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), (
            f"Working directory {self.working_dir} does not exist"
        )
        self.timeout = timeout
        self.agent_file_name = agent_file_name

        # TODO: ???
        self.process: Process = None  # type: ignore

        self.env = os.environ.copy()
        self.env.update(env_vars)

        self.agent_file_path = self.working_dir / agent_file_name

    def cleanup_session(self):
        # TODO: Do some cleanup here if necessary
        pass

    def run(
        self,
        code: str,
        reset_session=True,  # Currently unused, left compat
    ) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.

        """

        logger.debug(f"Writing code to agent file: {self.agent_file_path}")
        with open(self.agent_file_path, "w") as agent_file:
            agent_file.write(code)
            logger.debug("Done.")

        cmd = [
            "uv",
            "run",
            self.agent_file_path,
        ]
        cmd = [str(x) for x in cmd]

        exec_time = None
        output: list[str] = []
        has_exception = True
        start = time()
        try:
            res = subprocess.run(
                cmd,
                cwd=self.working_dir,
                env=self.env,
                timeout=self.timeout,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
            end = time()
            exec_time = end - start
            output = res.stdout.splitlines(keepends=True)
        except subprocess.TimeoutExpired as e:
            if e.stdout:
                # The subprocess is started with text=True, so stdout is already
                # a str. Guard against bytes too, just in case, and never let
                # decoding crash the timeout handler.
                stdout = e.stdout
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                output = stdout.splitlines(keepends=True)
            exec_time = self.timeout
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        except subprocess.CalledProcessError as e:
            end = time()
            exec_time = end - start
            output = e.stdout.splitlines(keepends=True)
            output.append(
                f"Program crashed with exception (see above) after {humanize.naturaldelta(exec_time)}!"
            )
        else:
            has_exception = False
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} (time limit is {humanize.naturaldelta(self.timeout)})."
            )

        return ExecutionResult(output, exec_time, has_exception)
