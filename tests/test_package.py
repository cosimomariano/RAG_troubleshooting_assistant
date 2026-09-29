import importlib


def testApplicationPackageImports() -> None:
    application = importlib.import_module("app")
    assert application.__name__ == "app"
