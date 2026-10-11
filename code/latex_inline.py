from typing import Final

INLINE_MATH: Final = {
    "t=T_a sqrt(k_m/C)": r"t=T_a\sqrt{k_m/C}",
    "T_a[k_m+2sqrt(Ck_m)]": r"T_a[k_m+2\sqrt{Ck_m}]",
    "T_a[1+sqrt(k_m(d)/C)]": r"T_a[1+\sqrt{k_m(d)/C}]",
    "H_C(T)=integral C(tau)d tau": r"H_C(T)=\int_{T_a}^{T}C(\tau)\,d\tau",
    "dh_g/dT=c_p,g(T)>0": r"dh_g/dT=c_{p,g}(T)>0",
    "F-T(partial F/partial T)": r"F-T(\partial F/\partial T)",
    "U_ads=Q integral theta[h_g(T)-E]d mu": r"U_{\mathrm{ads}}=Q\int\theta[h_g(T)-E]\,d\mu",
    "dn_out=-Q integral dtheta d mu": r"dn_{\mathrm{out}}=-Q\int d\theta\,d\mu",
    "-QE dtheta/dT": r"-QE\,d\theta/dT",
    "-QEtheta": r"-QE\theta",
    "dE/dT=c_p,g-c_p,ads": r"dE/dT=c_{p,g}-c_{p,\mathrm{ads}}",
    "ds/dT=(c_p,g-c_p,ads)/T": r"ds/dT=(c_{p,g}-c_{p,\mathrm{ads}})/T",
    "w'(a)=r/[r-(r-1)a]^2>0": r"w'(a)=r/[r-(r-1)a]^2>0",
    "w''(a)=2r(r-1)/[r-(r-1)a]^3>=0": r"w''(a)=2r(r-1)/[r-(r-1)a]^3\geq0",
    "q=Q sigmoid[ln p+ln b_ref+(E/R)(1/T−1/Tref)]": r"q=Q\sigma[\ln p+\ln b_{\mathrm{ref}}+(E/R)(1/T-1/T_{\mathrm{ref}})]",
    "1/sqrt(Nj M)": r"1/\sqrt{N_jM}",
    "DeltaH=cov(U,N)/var(N)−RT": r"\Delta H=\operatorname{cov}(U,N)/\operatorname{var}(N)-RT",
    "mu_g^0": r"\mu_g^0",
    "BT_a": r"BT_a",
    "H_C": r"H_C",
}
