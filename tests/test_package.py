import importlib


def test_application_package_imports() -> None:
    application = importlib.import_module("app")
    assert application.__name__ == "app"
