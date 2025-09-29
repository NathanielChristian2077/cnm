import sympy as sp
import math
from typing import Optional

# =========================
# Parsing & utils
# =========================
x = sp.symbols('x')
ALLOWED = {
    'x': x,
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
    'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
    'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
    'exp': sp.exp, 'log': sp.log, 'ln': sp.log,
    'sqrt': sp.sqrt, 'pi': sp.pi, 'E': sp.E
}

def parse_function(expr_str: str, var=x) -> sp.Expr:
    expr_str = expr_str.replace('^', '**')
    return sp.sympify(expr_str, locals=ALLOWED)

def derivative(expr: sp.Expr, var=x, order: int = 1) -> sp.Expr:
    return sp.diff(expr, var, order)

def evaluate(expr: sp.Expr, x_value: float, var=x):
    return sp.N(expr.subs(var, x_value))

def make_numeric(expr: sp.Expr, var=x, backend='math'):
    return sp.lambdify(var, expr, backend)

# =========================
# Helpers
# =========================

def _num_deriv(f, x, h: float = 1e-6):
    try:
        return (f(x+h) - f(x-h)) / (2*h)
    except Exception:
        return float('inf')

def _safe_eval(fun, x):
    val = fun(x)
    if not math.isfinite(float(val)):
        raise ValueError("Avaliação não finita.")
    return float(val)

# =========================
# Bisseção (com validações de finitude e prints limpos)
# =========================

def bissecao(expr_str: str, a: float, b: float,
             tol_intervalo: float = 1e-6,
             tol_func: float = 1e-12,
             max_iter: int = 100,
             var_name: str = 'x',
             imprime: bool = True):
    var = sp.symbols(var_name)
    expr = parse_function(expr_str, var)
    f = make_numeric(expr, var)

    fa = f(a); fb = f(b)
    for v in (fa, fb):
        if not math.isfinite(float(v)):
            raise ValueError("f(a) ou f(b) não finitos.")

    if fa == 0:
        if imprime:
            print("f(a) = 0. Raiz encontrada em a.")
        return a, 0, [{'k': 0, 'a': a, 'b': b, 'm': a, 'fa': fa, 'fb': fb, 'fm': 0.0, 'err': 0.0}]
    if fb == 0:
        if imprime:
            print("f(b) = 0. Raiz encontrada em b.")
        return b, 0, [{'k': 0, 'a': a, 'b': b, 'm': b, 'fa': fa, 'fb': fb, 'fm': 0.0, 'err': 0.0}]

    if fa * fb > 0:
        raise ValueError("Intervalo inválido: f(a) e f(b) têm o mesmo sinal.")

    header = f"{'k':>3} | {'a':>12} {'b':>12} {'m':>12} | {'f(a)':>12} {'f(b)':>12} {'f(m)':>12} | {'|b-a|/2':>12}"
    sep = "-" * len(header)
    historico = []

    if imprime:
        print(header); print(sep)

    for k in range(1, max_iter + 1):
        m = (a + b) / 2.0
        fm = f(m)
        err = abs(b - a) / 2.0

        historico.append({'k': k, 'a': a, 'b': b, 'm': m, 'fa': fa, 'fb': fb, 'fm': fm, 'err': err})
        if imprime:
            print(f"{k:>3} | {a:>12.6f} {b:>12.6f} {m:>12.6f} | {fa:>12.6g} {fb:>12.6g} {fm:>12.6g} | {err:>12.6g}")

        if abs(fm) <= tol_func or err <= tol_intervalo:
            return m, k, historico

        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    if imprime:
        print("Atingiu max_iter. Pegue o último m como aproximação.")
    return m, max_iter, historico

# =========================
# Newton-Raphson (dupla parada, damping e cerca opcional)
# =========================

def newton_raphson(expr_str: str, x0: float, var_name: str = 'x',
                   max_iter: int = 100, tol_decimals: int = 4, imprime: bool = True,
                   a: Optional[float] = None, b: Optional[float] = None):
    var = sp.symbols(var_name)
    f_expr = parse_function(expr_str, var)
    df_expr = sp.diff(f_expr, var)

    f  = make_numeric(f_expr, var)
    df = make_numeric(df_expr, var)

    eps = 0.5 * 10**(-tol_decimals)
    if imprime:
        header = f"{'k':>2} | {'x_k':>12} {'f(x_k)':>14} {'f\'(x_k)':>14} | {'x_{k+1}':>12} {'|Δx|':>12} {'|f|':>12}"
        print(header); print('-' * len(header))

    hist = []
    xk = float(x0)
    for k in range(1, max_iter + 1):
        fx  = f(xk); dfx = df(xk)
        if dfx == 0:
            raise ZeroDivisionError("Derivada zerada. Escolha outro x0.")
        xnext = xk - fx/dfx

        if a is not None and b is not None:
            lo, hi = (a,b) if a < b else (b,a)
            xnext = min(max(xnext, lo), hi)

        # damping se piorou
        try:
            if abs(f(xnext)) > abs(fx):
                xnext = xk - 0.5*fx/dfx
        except Exception:
            pass

        err = abs(xnext - xk)
        try:
            fval = abs(f(xnext))
        except Exception:
            fval = float('inf')

        hist.append({'k': k, 'xk': xk, 'fx': fx, 'dfx': dfx, 'xnext': xnext, 'err': err, 'f': fval})
        if imprime:
            print(f"{k:>2} | {xk:>12.9f} {fx:>14.9g} {dfx:>14.9g} | {xnext:>12.9f} {err:>12.3e} {fval:>12.3e}")
        if err < eps and fval < 5*eps:
            return xnext, k, hist, f_expr, df_expr
        xk = xnext

    return xk, max_iter, hist, f_expr, df_expr

# =========================
# Secante (dupla parada e nudge)
# =========================

def secante(expr_str: str, x0: float, x1: float, var_name: str = 'x',
            max_iter: int = 100, tol_decimals: int = 5, imprime: bool = True):
    var = sp.symbols(var_name)
    f_expr = parse_function(expr_str, var)
    f = make_numeric(f_expr, var)

    eps = 0.5 * 10**(-tol_decimals)

    fx0 = f(x0)
    fx1 = f(x1)
    if fx0 == 0:
        if imprime:
            print("f(x0)=0. Raiz encontrada em x0.")
        return x0, 0, [{'k': 0, 'xk_1': None, 'xk': x0, 'fxk_1': None, 'fxk': fx0, 'xnext': x0, 'err': 0.0, 'f': 0.0}], f_expr
    if fx1 == 0:
        if imprime:
            print("f(x1)=0. Raiz encontrada em x1.")
        return x1, 0, [{'k': 0, 'xk_1': x0, 'xk': x1, 'fxk_1': fx0, 'fxk': fx1, 'xnext': x1, 'err': 0.0, 'f': 0.0}], f_expr

    if imprime:
        header = f"{'k':>2} | {'x_{k-1}':>12} {'x_k':>12} | {'f(x_{k-1})':>14} {'f(x_k)':>14} | {'x_{k+1}':>12} {'|Δx|':>12} {'|f|':>12}"
        print(header); print('-' * len(header))

    hist = []
    for k in range(1, max_iter + 1):
        fx0 = f(x0); fx1 = f(x1)
        denom = (fx1 - fx0)
        if denom == 0:
            raise ZeroDivisionError("Divisão por zero na secante. Tente outros chutes.")
        x2 = x1 - fx1*(x1 - x0)/denom
        if x2 == x1:
            x2 = x1 + 1e-12
        err = abs(x2 - x1); 
        try:
            fval = abs(f(x2))
        except Exception:
            fval = float('inf')

        hist.append({'k': k, 'xk_1': x0, 'xk': x1, 'fxk_1': fx0, 'fxk': fx1, 'xnext': x2, 'err': err, 'f': fval})
        if imprime:
            print(f"{k:>2} | {x0:>12.9f} {x1:>12.9f} | {fx0:>14.9g} {fx1:>14.9g} | {x2:>12.9f} {err:>12.3e} {fval:>12.3e}")
        if err < eps and fval < 5*eps:
            return x2, k, hist, f_expr
        x0, x1 = x1, x2

    return x1, max_iter, hist, f_expr

# =========================
# Ponto Fixo Universal (g pronto ou f->g com λ adaptativo, Aitken, projeção)
# =========================

def ponto_fixo_universal(
    *,
    g_expr_str: Optional[str] = None,        # se você já tem g(x)
    f_expr_str: Optional[str] = None,        # ou só f(x)=0
    x0: float,
    a: Optional[float] = None,               # intervalo opcional
    b: Optional[float] = None,
    var_name: str = 'x',
    max_iter: int = 300,
    tol_x: float = 1e-8,
    tol_f: float = 1e-10,
    imprime: bool = True,
    usar_aitken: bool = True,
    lambda_inicial: Optional[float] = None,
):
    """
    Resolve ponto fixo robusto:
      - se g não for dado, constrói g = x - λ f com λ adaptativo
      - garante contração local reduzindo λ quando |g'| ~ 1
      - para por ||Δx|| < tol_x e |f| < tol_f (se f existir)
      - projeta em [a,b] se intervalo for dado
      - aceleração de Aitken opcional
    Retorna: (raiz, iters, historico, g_expr, f_expr_ou_None, lambda_final)
    """
    var = sp.symbols(var_name)

    if g_expr_str is None and f_expr_str is None:
        raise ValueError("Passe g_expr_str ou f_expr_str.")

    g_expr = None
    f_expr = None
    f = None
    g = None

    if f_expr_str is not None:
        f_expr = parse_function(f_expr_str, var)
        f = make_numeric(f_expr, var)

    if g_expr_str is not None:
        g_expr = parse_function(g_expr_str, var)
        g = make_numeric(g_expr, var)

    # Se não foi dado g, construímos g = x - λ f
    lam = 1.0
    if g is None:
        lam = 1.0 if lambda_inicial is None else float(lambda_inicial)
        try:
            df_expr = sp.diff(f_expr, var)
            df = make_numeric(df_expr, var)
            d0 = abs(_safe_eval(df, x0))
            if math.isfinite(d0) and d0 > 0:
                lam = 1.0 / d0
            elif lambda_inicial is not None:
                lam = float(lambda_inicial)
            else:
                d0n = abs(_num_deriv(f, x0))
                lam = 1.0 / d0n if math.isfinite(d0n) and d0n > 0 else 1e-1
        except Exception:
            if lambda_inicial is not None:
                lam = float(lambda_inicial)
            else:
                d0n = abs(_num_deriv(f, x0))
                lam = 1.0 / d0n if math.isfinite(d0n) and d0n > 0 else 1e-1

        g_expr = sp.simplify(var - lam * f_expr)
        g = make_numeric(g_expr, var)

    def _project(z):
        if a is not None and b is not None:
            aa, bb = (a, b) if a <= b else (b, a)
            if z < aa: return aa
            if z > bb: return bb
        return z

    # impressão
    if imprime:
        if f_expr is None:
            header = f"{'k':>3} | {'x_k':>14} {'g(x_k)':>14} {'|Δx|':>12}"
        else:
            header = f"{'k':>3} | {'x_k':>14} {'g(x_k)':>14} {'|Δx|':>12} {'|f|':>12} {'λ':>10}"
        print(header)
        print('-' * len(header))

    hist = []
    xkm2 = None
    xkm1 = None
    xk = float(x0)

    for k in range(1, max_iter + 1):
        gx = _safe_eval(g, xk)
        xnext = _project(gx)
        err = abs(xnext - xk)
        fval = None
        if f is not None:
            try:
                fval = abs(_safe_eval(f, xnext))
            except Exception:
                fval = float('inf')

        rec = {'k': k, 'xk': xk, 'gx': gx, 'xnext': xnext, 'err': err, 'f': fval, 'lambda': lam}
        hist.append(rec)

        if imprime:
            if f is None:
                print(f"{k:>3} | {xk:>14.9f} {gx:>14.9f} {err:>12.3e}")
            else:
                print(f"{k:>3} | {xk:>14.9f} {gx:>14.9f} {err:>12.3e} {fval:>12.3e} {lam:>10.3e}")

        if err <= tol_x and (fval is None or fval <= tol_f):
            return xnext, k, hist, g_expr, f_expr, lam

        # checagem de contração local e ajuste de λ
        try:
            gp = abs(_num_deriv(g, xk))
        except Exception:
            gp = float('inf')

        if f_expr is not None and gp >= 0.95:
            lam *= 0.5
            g_expr = sp.simplify(var - lam * f_expr)
            g = make_numeric(g_expr, var)
            xkm2 = None
            xkm1 = None
            continue  # reavalia com mesmo xk

        # Aitken Δ²
        if usar_aitken:
            if xkm2 is not None and xkm1 is not None:
                dx1 = xkm1 - xkm2
                dx2 = xnext - xkm1
                denom = dx2 - dx1
                if denom != 0 and math.isfinite(denom):
                    x_aitken = xkm2 - dx1*dx1/denom
                    if math.isfinite(x_aitken):
                        if abs(x_aitken - xnext) < 10*max(1e-12, err):
                            xnext = _project(x_aitken)
                            err = abs(xnext - xk)
                            hist[-1]['x_aitken'] = xnext
                            hist[-1]['err_after_aitken'] = err

        # ciclo curto (período 2)
        if xkm1 is not None and abs(xnext - xkm1) < 1e-14 and err > 1e-6:
            if f_expr is not None:
                lam *= 0.5
                g_expr = sp.simplify(var - lam * f_expr)
                g = make_numeric(g_expr, var)

        xkm2, xkm1, xk = xkm1, xk, xnext

    return xk, max_iter, hist, g_expr, f_expr, lam

# =========================
# Exemplo rápido de uso
# =========================
if __name__ == '__main__':
    raiz_bi, it_bi, hist_bi = bissecao("(x**3) - (2*x**2) - 5", a=2, b=3, tol_intervalo=1e-5, tol_func=1e-8)
    print("\n[Bisseção] Aproximação da raiz:", f"{raiz_bi:.9f}")
    print("[Bisseção] Iterações:", it_bi)

    print("\n[Newton-Raphson] f(x) = ln(x+2) - 1, x0 = 1\n")
    raiz_nr, it_nr, hist_nr, f_expr_nr, df_expr_nr = newton_raphson("ln(x+2) - 1", x0=1, tol_decimals=5, imprime=True)
    print(f"\n[Newton] Aproximação (5 casas): {raiz_nr:.5f}")
    print(f"[Newton] Iterações: {it_nr}")

    print("\n[Secante] f(x)=exp(x)-4 em [1,2], 5 casas\n")
    raiz_s3, it_s3, hist_s3, fexpr_s3 = secante("exp(x) - 4", x0=1.0, x1=2.0, tol_decimals=5, imprime=True)
    print(f"\n[Secante] Aproximação (5 casas): {raiz_s3:.5f}")
    print(f"[Secante] Iterações: {it_s3}")

    print("\n[Ponto Fixo] f(x)=x^2-3, x0=1, universal\n")
    raiz_pf, it_pf, hist_pf, gexpr_pf, fexpr_pf, lam_pf = ponto_fixo_universal(
        f_expr_str="x**2 - 3",
        x0=1.0,
        a=0.0, b=3.0,
        tol_x=1e-8, tol_f=1e-10, imprime=True
    )
    print(f"\n[Ponto Fixo] Aproximação: {raiz_pf:.9f} | iterações: {it_pf} | λ_final: {lam_pf:.3e}")
