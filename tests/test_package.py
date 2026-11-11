def test_ris_package_importable():
    import ris
    
    assert ris.__name__ == "ris"