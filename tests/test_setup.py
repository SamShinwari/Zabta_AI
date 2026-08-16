from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


def test_project_root_exists():

    assert ROOT_DIR.exists()


def test_config_directory_exists():

    assert (ROOT_DIR / "config").exists()


def test_data_directory_exists():

    assert (ROOT_DIR / "data").exists()


def test_fbr_directory_exists():

    assert (
        ROOT_DIR /
        "data" /
        "raw" /
        "fbr"
    ).exists()


def test_tax_rules_directory_exists():

    assert (
        ROOT_DIR /
        "data" /
        "tax_rules"
    ).exists()


def test_vector_database_directory_exists():

    assert (
        ROOT_DIR /
        "vector_db"
    ).exists()


def test_src_directory_exists():

    assert (
        ROOT_DIR /
        "src"
    ).exists()