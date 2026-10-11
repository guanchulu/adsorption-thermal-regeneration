from latex_text import tex_text


def test_source_checksum_survives_tex_conversion() -> None:
    checksum = "46ee97a410cd1b92455714df144ca2641f108254d3de0145c96af250197ac85f"
    converted = tex_text("Original archive SHA256: " + checksum)
    assert checksum in converted


def test_inverse_molar_heat_capacity_uses_native_exponents() -> None:
    converted = tex_text("J mol^-1 K^-1")
    assert r"\mathrm{J\,mol^{-1}\,K^{-1}}" in converted


def test_ascii_inequality_uses_math_relation() -> None:
    converted = tex_text("E>=0")
    assert r"\geq" in converted
