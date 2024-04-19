import streamlit as st
import numpy as np
import math
from scipy.stats import norm

# Modèle Black-Scholes
def black_scholes(option_type, S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type doit être 'call' ou 'put'.")

    return option_price

# Modèle Ornstein-Uhlenbeck
def ornstein_uhlenbeck(S0, theta, mu, sigma, T, n):
    dt = T / n
    dW = np.random.normal(0, np.sqrt(dt), n)
    t = np.linspace(0, T, n+1)
    S = np.zeros(n+1)
    S[0] = S0

    for i in range(1, n+1):
        S[i] = S[i-1] + theta * (mu - S[i-1]) * dt + sigma * dW[i-1]

    return t, S

# Interface utilisateur
def main():
    st.title("Modèles Black-Scholes et Ornstein-Uhlenbeck")
    st.sidebar.title("Paramètres")

    option_type = st.sidebar.radio("Type d'option", ["Call", "Put"])
    S0 = st.sidebar.number_input("Prix actuel du sous-jacent (S0)", value=100.0)
    K = st.sidebar.number_input("Prix d'exercice (K)", value=105.0)
    T = st.sidebar.number_input("Temps de maturité (en années)", value=0.5)
    r = st.sidebar.number_input("Taux d'intérêt sans risque (r)", value=0.05)
    sigma = st.sidebar.number_input("Volatilité du sous-jacent (σ)", value=0.2)

    # Calcul du prix de l'option avec le modèle Black-Scholes
    call_price = black_scholes("call", S0, K, T, r, sigma)
    put_price = black_scholes("put", S0, K, T, r, sigma)

    st.write(f"Prix de l'option d'achat : **{call_price:.2f}**")
    st.write(f"Prix de l'option de vente : **{put_price:.2f}**")

    # Calcul du processus Ornstein-Uhlenbeck
    theta = st.sidebar.number_input("Paramètre theta", value=0.1)
    mu = st.sidebar.number_input("Paramètre mu", value=100.0)
    sigma_ou = st.sidebar.number_input("Volatilité du processus Ornstein-Uhlenbeck", value=0.2)
    n = st.sidebar.number_input("Nombre de pas de temps", value=100)

    t, S_ou = ornstein_uhlenbeck(S0, theta, mu, sigma_ou, T, n)
    st.line_chart(S_ou, use_container_width=True)

if __name__ == "__main__":
    main()
