#finitedif_class_scratch.py
from numpy import sin, sqrt, exp,cos,pi 

def f(x):
    return exp(x)/sqrt(sin(x)**3 + cos(x)**3)
def df(x,h):
    return (f(x+h) - f(x))/h
def df_cent(x,h):
    return (f(x+h) - f(x-h))/(2*h)
def df_exact(x):
    return exp(x)/sqrt(sin(x)**3 + cos(x)**3) + (-1/2)*exp(x)*(sin(x)**3 + cos
exact_df_value = df_exact(1.5)
print("The exact value using symbolic differentiation is:")
print(exact_df_value, "<----- exact value")
h = 10**-15
print(df(1.5,h))
print("error:", df(1.5, h)-exact_df_value)