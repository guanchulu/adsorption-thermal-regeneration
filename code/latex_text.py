# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# Imported by build_latex.py; protects mathematical tokens before text escaping.
import re
from typing import Final

from latex_inline import INLINE_MATH

SYMBOLS: Final = {
    "θ": r"\theta",
    "δ": r"\delta",
    "μ": r"\mu",
    "γ": r"\gamma",
    "σ": r"\sigma",
    "Δ": r"\Delta",
    "β": r"\beta",
    "τ": r"\tau",
    "ℓ": r"\ell",
    "∞": r"\infty",
    "≤": r"\leq",
    "≥": r"\geq",
    "≈": r"\approx",
    "≠": r"\neq",
    "∈": r"\in",
    "∫": r"\int",
    "√": r"\sqrt{\vphantom{x}}",
    "±": r"\pm",
    "×": r"\times",
    "−": "-",
    "′": r"\prime",
    "²": "{}^2",
    "³": "{}^3",
    "→": r"\to",
}
TERMS: Final = {
    "delta": r"\delta",
    "Nj": "N_j",
    "U_ads": r"U_{\mathrm{ads}}",
    "mu_g": r"\mu_g",
    "mu": r"\mu",
    "tau": r"\tau",
    "h_ens": r"h_{\mathrm{ens}}",
    "h_hom": r"h_{\mathrm{hom}}",
    "b_m": "b_m",
    "h_m": "h_m",
    "E_min,m": r"E_{\min,m}",
    "k_allow": r"k_{\mathrm{allow}}",
    "b_ref": r"b_{\mathrm{ref}}",
    "M_k": "M_k",
    "M_0": "M_0",
    "M_1": "M_1",
    "M_2": "M_2",
    "Q_i": "Q_i",
    "theta_i": r"\theta_i",
    "theta_v": r"\theta_v",
    "d_0": "d_0",
    "v_hat": r"\hat v",
    "v_l": "v_l",
    "E′": r"E^{\prime}",
    "Qg_m": "Qg_m",
    "Qu_m": "Qu_m",
    "QE": "QE",
    "AE": "AE",
    "Ct": "Ct",
    "Ck_m": "Ck_m",
    "k_mT_a": "k_mT_a",
    "Qc_max": r"Qc_{\max}",
    "atanh": r"\operatorname{atanh}",
    "tanh": r"\tanh",
    "logit": r"\operatorname{logit}",
    "ln": r"\ln",
    "T_d,ref": r"T_{d,\mathrm{ref}}",
    "T_ref": r"T_{\mathrm{ref}}",
    "Tref": r"T_{\mathrm{ref}}",
    "T_cap": r"T_{\mathrm{cap}}",
    "Tcap": r"T_{\mathrm{cap}}",
    "T_a": "T_a",
    "Ta": "T_a",
    "T_d": "T_d",
    "Td": "T_d",
    "T_l": "T_l",
    "Tl": "T_l",
    "T_u": "T_u",
    "Tu": "T_u",
    "f_a": "f_a",
    "fa": "f_a",
    "f_d": "f_d",
    "fd": "f_d",
    "f_0": "f_0",
    "f0": "f_0",
    "H_path": r"H_{\mathrm{path}}",
    "Hpath": r"H_{\mathrm{path}}",
    "H_red": r"H_{\mathrm{red}}",
    "Hred": r"H_{\mathrm{red}}",
    "C_dry": r"C_{\mathrm{dry}}",
    "C_plus": "C_+",
    "c_p,ads": r"c_{p,\mathrm{ads}}",
    "cp,ads": r"c_{p,\mathrm{ads}}",
    "c_p,g": "c_{p,g}",
    "cp,g": "c_{p,g}",
    "c_max": r"c_{\max}",
    "cmax": r"c_{\max}",
    "u_ads": r"u_{\mathrm{ads}}",
    "uads": r"u_{\mathrm{ads}}",
    "h_g": "h_g",
    "hg": "h_g",
    "theta_a0": r"\theta_{a0}",
    "theta_a": r"\theta_a",
    "theta_d": r"\theta_d",
    "theta": r"\theta",
    "θa": r"\theta_a",
    "θd": r"\theta_d",
    "θa0": r"\theta_{a0}",
    "v0": "v_0",
    "v_0": "v_0",
    "d_bar": r"\bar d",
    "d̄": r"\bar d",
    "g_m": "g_m",
    "gm": "g_m",
    "F_m": "F_m",
    "Fm": "F_m",
    "k_m": "k_m",
    "km": "k_m",
    "ell_m": r"\ell_m",
    "u_m": "u_m",
    "q_f": "q_f",
    "M0": "M_0",
    "M1": "M_1",
    "M2": "M_2",
    "dupper": r"d_{\mathrm{upper}}",
    "dfeasible": r"d_{\mathrm{feasible}}",
    "CO2": r"\mathrm{CO_2}",
    "N2": r"\mathrm{N_2}",
    "Q": "Q",
    "C": "C",
    "B": "B",
    "E": "E",
    "R": "R",
    "L": "L",
    "J": "J",
    "T": "T",
    "Qd̄": r"Q\bar d",
    "Qd": "Qd",
    "s": "s",
    "m": "m",
    "d": "d",
    "v": "v",
    "w": "w",
    "k": "k",
}
TERM_RE: Final = re.compile(
    r"(?<![A-Za-z0-9])("
    + "|".join(re.escape(k) for k in sorted(TERMS, key=len, reverse=True))
    + r")(?![A-Za-z0-9])"
)


def tex_text(text: str, *, math_tokens: bool = True) -> str:
    saved: list[str] = []

    def protect(raw: str) -> str:
        saved.append(raw)
        return f"ZZTOKEN{len(saved) - 1}ZZ"

    def math(raw: str) -> str:
        return protect(r"\(" + raw + r"\)")

    text = re.sub(
        r"\[@([^\]]+)\]",
        lambda match: protect(r"\cite{" + ",".join(x.strip() for x in match[1].split(";")) + "}"),
        text,
    )
    text = re.sub(r"\\\((.*?)\\\)", lambda match: protect(match[0]), text)
    text = re.sub(r"https?://[^\s]+", lambda match: protect(r"\url{" + match[0] + "}"), text)
    text = re.sub(
        r"\b[a-f0-9]{40,64}\b", lambda match: protect(r"\texttt{\seqsplit{" + match[0] + "}}"), text
    )
    text = re.sub(
        r"\b(?:code|results|qa|data)/[A-Za-z0-9_./-]+",
        lambda match: protect(r"\path{" + match[0] + "}"),
        text,
    )
    for source, target in sorted(INLINE_MATH.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(source, math(target))
    text = text.replace("<=", "≤").replace(">=", "≥")
    for source, target in (
        ("kJ kg−1 K−1", r"\mathrm{kJ\,kg^{-1}\,K^{-1}}"),
        ("kJ mol−1 K−1", r"\mathrm{kJ\,mol^{-1}\,K^{-1}}"),
        ("kJ kg^-1 K^-1", r"\mathrm{kJ\,kg^{-1}\,K^{-1}}"),
        ("kJ mol^-1 K^-1", r"\mathrm{kJ\,mol^{-1}\,K^{-1}}"),
        ("J mol^-1 K^-1", r"\mathrm{J\,mol^{-1}\,K^{-1}}"),
        ("kJ kg−1", r"\mathrm{kJ\,kg^{-1}}"),
        ("kJ mol−1", r"\mathrm{kJ\,mol^{-1}}"),
        ("kJ kg^-1", r"\mathrm{kJ\,kg^{-1}}"),
        ("kJ mol^-1", r"\mathrm{kJ\,mol^{-1}}"),
        ("mol kg−1", r"\mathrm{mol\,kg^{-1}}"),
        ("mol kg^-1", r"\mathrm{mol\,kg^{-1}}"),
        ("mmol g−1", r"\mathrm{mmol\,g^{-1}}"),
        ("mmol g^-1", r"\mathrm{mmol\,g^{-1}}"),
        ("cm3 mmol−1", r"\mathrm{cm^3\,mmol^{-1}}"),
    ):
        text = text.replace(source, math(target))
    text = re.sub(
        r"(\d+(?:\.\d+)?)×10[−-](\d+)", lambda match: math(match[1] + r"\times10^{-" + match[2] + "}"), text
    )
    text = re.sub(r"10(?:\^-|−)(\d+)", lambda match: math("10^{-" + match[1] + "}"), text)
    text = text.replace("Tref/Ta=√[1+B/(CTa)]", math(r"T_{\mathrm{ref}}/T_a=\sqrt{1+B/(CT_a)}"))
    if math_tokens:
        text = TERM_RE.sub(lambda match: math(TERMS[match[0]]), text)
    for char, latex in SYMBOLS.items():
        text = text.replace(char, math(latex))
    text = text.replace("–", "--").replace("—", "---").replace("’", "'").replace("“", "``").replace("”", "''")
    for char, escaped in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
    ):
        text = text.replace(char, escaped)
    for index, raw in enumerate(saved):
        text = text.replace(f"ZZTOKEN{index}ZZ", raw)
    return text
