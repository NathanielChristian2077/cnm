import sympy as sp
import math

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
# Bisseção
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

    fa = f(a)
    fb = f(b)

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

    header = f"{'k':>3} | {'a':>12} {'b':>12} {'m':>12} | {'f(a)':>12} {'f(b)':>12} {'f(m)':>12} | {'err':>12}"
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
# Newton-Raphson
# =========================
def newton_raphson(expr_str: str, x0: float, var_name: str = 'x',
                   max_iter: int = 100, tol_decimals: int = 4, imprime: bool = True):
    var = sp.symbols(var_name)
    f_expr = parse_function(expr_str, var)
    df_expr = sp.diff(f_expr, var)

    f  = make_numeric(f_expr, var)
    df = make_numeric(df_expr, var)

    eps = 0.5 * 10**(-tol_decimals)

    if imprime:
        header = f"{'k':>2} | {'x_k':>12} {'f(x_k)':>14} {'f\'(x_k)':>14} | {'x_{k+1}':>12} {'|Δx|':>12}"
        print(header); print('-' * len(header))

    hist = []
    xk = float(x0)

    for k in range(1, max_iter + 1):
        fx  = f(xk)
        dfx = df(xk)
        if dfx == 0:
            raise ZeroDivisionError("Derivada zerada. Escolha outro x0.")

        xnext = xk - fx/dfx
        err = abs(xnext - xk)

        hist.append({'k': k, 'xk': xk, 'fx': fx, 'dfx': dfx, 'xnext': xnext, 'err': err})

        if imprime:
            print(f"{k:>2} | {xk:>12.9f} {fx:>14.9g} {dfx:>14.9g} | {xnext:>12.9f} {err:>12.9g}")

        if err < eps:
            return xnext, k, hist, f_expr, df_expr

        xk = xnext

    return xk, max_iter, hist, f_expr, df_expr

# =========================
# Secante
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
        return x0, 0, [{'k': 0, 'xk_1': None, 'xk': x0, 'fxk_1': None, 'fxk': fx0, 'xnext': x0, 'err': 0.0}], f_expr
    if fx1 == 0:
        if imprime:
            print("f(x1)=0. Raiz encontrada em x1.")
        return x1, 0, [{'k': 0, 'xk_1': x0, 'xk': x1, 'fxk_1': fx0, 'fxk': fx1, 'xnext': x1, 'err': 0.0}], f_expr

    if imprime:
        header = f"{'k':>2} | {'x_{k-1}':>12} {'x_k':>12} | {'f(x_{k-1})':>14} {'f(x_k)':>14} | {'x_{k+1}':>12} {'|Δx|':>12}"
        print(header); print('-' * len(header))

    hist = []
    for k in range(1, max_iter + 1):
        fx0 = f(x0); fx1 = f(x1)
        denom = (fx1 - fx0)
        if denom == 0:
            raise ZeroDivisionError("Divisão por zero na secante. Tente outros chutes.")
        x2 = x1 - fx1*(x1 - x0)/denom
        err = abs(x2 - x1)

        hist.append({'k': k, 'xk_1': x0, 'xk': x1, 'fxk_1': fx0, 'fxk': fx1, 'xnext': x2, 'err': err})
        if imprime:
            print(f"{k:>2} | {x0:>12.9f} {x1:>12.9f} | {fx0:>14.9g} {fx1:>14.9g} | {x2:>12.9f} {err:>12.9g}")

        if err < eps:
            return x2, k, hist, f_expr

        x0, x1 = x1, x2

    return x1, max_iter, hist, f_expr

# =========================
# Ponto Fixo
# =========================
def ponto_fixo(g_expr_str: str, x0: float, var_name: str = 'x',
               max_iter: int = 200, tol_decimals: int = 5, imprime: bool = True):
    var = sp.symbols(var_name)
    g_expr = parse_function(g_expr_str, var)
    g = make_numeric(g_expr, var)

    eps = 0.5 * 10**(-tol_decimals)

    if imprime:
        header = f"{'k':>2} | {'x_k':>14} {'g(x_k)':>14} {'|Δx|':>12}"
        print(header); print('-' * len(header))

    hist = []
    xk = float(x0)
    for k in range(1, max_iter + 1):
        gx = g(xk)
        err = abs(gx - xk)
        hist.append({'k': k, 'xk': xk, 'gx': gx, 'err': err})
        if imprime:
            print(f"{k:>2} | {xk:>14.9f} {gx:>14.9f} {err:>12.9g}")
        if err < eps:
            return gx, k, hist, g_expr
        xk = gx

    return xk, max_iter, hist, g_expr

raiz_bi, it_bi, hist_bi = bissecao("(x**3) - (2*x**2) - 5", a=2, b=3, tol_intervalo=1e-5, tol_func=1e-5)
print("\n[Bisseção] Aproximação da raiz:", raiz_bi)
print("[Bisseção] Iterações:", it_bi)

print("\nNewton-Raphson para f(x) = ln(x+2) - 1, x0 = 1, alvo: 4 casas decimais\n")
raiz_nr, it_nr, hist_nr, f_expr_nr, df_expr_nr = newton_raphson("ln(x+2) - 1", x0=1, tol_decimals=4, imprime=True)
print(f"\n[Newton] Aproximação (4 casas): {raiz_nr:.4f}")
print(f"[Newton] Iterações: {it_nr}")
print("[Newton] f(x) simbólica:", f_expr_nr)
print("[Newton] f'(x) simbólica:", df_expr_nr)

print("\n=== Secante para f(x)=exp(x)-4 em [1,2], 5 casas ===\n")
raiz_s3, it_s3, hist_s3, fexpr_s3 = secante("exp(x) - 4", x0=1.0, x1=2.0, tol_decimals=5, imprime=True)
print(f"\n[Secante - Caso 3] Aproximação (5 casas): {raiz_s3:.5f}")
print(f"[Secante - Caso 3] Iterações: {it_s3}")

print("\n=== Ponto Fixo para f(x)=x^2-3 em [1,2], x0=1 ===\n")
raiz_pf4, it_pf4, hist_pf4, gexpr4 = ponto_fixo("(x + 3/x)/2", x0=1.0, tol_decimals=5, imprime=True)
print(f"\n[Ponto Fixo - Caso 4] Aproximação (5 casas): {raiz_pf4:.5f}")
print(f"[Ponto Fixo - Caso 4] Iterações: {it_pf4}")

print("\n=== Secante para f(x)=cos(x)-exp(-x) em [0,1], tol=1e-5 ===\n")
raiz_s5, it_s5, hist_s5, fexpr_s5 = secante("cos(x) - exp(-x)", x0=0.0, x1=1.0, tol_decimals=5, imprime=True)
print(f"\n[Secante - Caso 5] Aproximação (5 casas): {raiz_s5:.5f}")
print(f"[Secante - Caso 5] Iterações: {it_s5}")
