from pathlib import Path

import pytest

from treesearch.interpreter import Interpreter

IMPATIENT_TIMEOUT = 1


@pytest.fixture
def interpreter(tmp_path: Path):
    i = Interpreter(tmp_path)
    yield i
    i.cleanup_session()


@pytest.fixture
def impatient_interpreter(tmp_path: Path):
    i = Interpreter(tmp_path, timeout=IMPATIENT_TIMEOUT)
    yield i
    i.cleanup_session()


def test_error(interpreter: Interpreter):
    res = interpreter.run("1/0")
    term_out = res.term_out

    assert term_out[0].startswith("Traceback")
    assert term_out[-1].startswith("Program crashed")

    # Check if line ending are kept in lines
    for line in term_out[:-1]:
        assert line.endswith("\n")

    assert res.has_exception


def test_simple_run(interpreter: Interpreter):
    res = interpreter.run("print(42)")

    assert res.term_out[0] == "42\n"
    assert res.term_out[1].startswith("Execution time:")

    assert not res.has_exception


def test_timeout(impatient_interpreter: Interpreter):
    res = impatient_interpreter.run("from time import sleep\nsleep(20)")

    assert res.term_out[0].startswith("TimeoutError")

    assert res.exec_time == IMPATIENT_TIMEOUT
    assert res.has_exception


def test_multiprocessing(interpreter: Interpreter):
    code = """from concurrent.futures import ProcessPoolExecutor
from random import randint
from time import sleep


def work(x):
    return x * x


def main():
    with ProcessPoolExecutor() as pool:
        print(list(pool.map(work, range(5))))


if __name__ == "__main__":
    main()
"""

    res = interpreter.run(code)

    assert res.term_out[0] == "[0, 1, 4, 9, 16]\n"
    assert res.term_out[1].startswith("Execution time:")

    assert not res.has_exception
