import math


def func_apply(x):
    return math.sin(x.a**2) + math.sin(x.b**2)


def func_applymap(x):
    return math.sin(x**2) - math.cos(x**2)


def func_groupby_apply(df):
    dum = 0
    for item in df.b:
        dum += math.log10(math.sqrt(math.exp(item**2)))

    return dum / len(df.b)


def func_groupby_rolling_apply(x):
    return x.iloc[0] + x.iloc[1] ** 2 + x.iloc[2] ** 3 + x.iloc[3] ** 4
