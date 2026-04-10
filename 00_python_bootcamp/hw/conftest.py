import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--module-suffix",
        action="store",
        default="",
        help="Suffix to append to module names when importing (e.g. '_solution')",
    )


@pytest.fixture
def module_suffix(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--module-suffix")
