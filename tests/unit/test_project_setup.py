def test_python_runtime_is_supported():
    import sys

    assert sys.version_info.major == 3
    assert sys.version_info.minor == 12


def test_industrial_ai_package_imports():
    import industrial_ai

    assert industrial_ai.__file__ is not None
