'''import numpy as np
import sounddevice as sd
import time
import random

DURACAO_SINAL = 1800.0  # Duração do sinal em segundos
FREQUENCIAS = [10000]  # Frequências das senoides em Hz
TAXA_AMOSTRAGEM = 44100  # Taxa de amostragem em Hz
TEMPO_TOTAL = 90.0  # Duração total em segundos (1,5 minutos)

def gera_senoide(frequencia, duracao=DURACAO_SINAL, taxa_amostragem=TAXA_AMOSTRAGEM):
    t = np.linspace(0, duracao, int(taxa_amostragem * duracao), endpoint=False)
    return np.sin(2 * np.pi * frequencia * t)

def toca_aleatoriamente():
    frequencias_tocadas = []
    inicio = time.time()
    while time.time() - inicio < TEMPO_TOTAL:
        freq = random.choice(FREQUENCIAS)
        frequencias_tocadas.append(freq)
        sinal = gera_senoide(freq)
        sd.play(sinal, samplerate=TAXA_AMOSTRAGEM)
        time.sleep(DURACAO_SINAL)
    return frequencias_tocadas

if __name__ == "__main__":
    frequencias = toca_aleatoriamente()
    print("Frequências tocadas em ordem:")
    for freq in frequencias:
        print(f"{freq} Hz")'''

import numpy as np
import sounddevice as sd
import time
import random

DURACAO_SINAL = 15.0  # Duração do sinal em segundos
TAXA_AMOSTRAGEM = 44100  # Taxa de amostragem em Hz
TEMPO_TOTAL = 60.0  # Duração total em segundos (1 minuto)

def gera_senoide(frequencia, duracao=DURACAO_SINAL, taxa_amostragem=TAXA_AMOSTRAGEM):
    t = np.linspace(0, duracao, int(taxa_amostragem * duracao), endpoint=False)
    return np.sin(2 * np.pi * frequencia * t)

def toca_aleatoriamente():
    frequencias_tocadas = []
    inicio = time.time()
    while time.time() - inicio < TEMPO_TOTAL:
        tempo_corrente = time.time() - inicio
        if tempo_corrente < 15:
            freq = 10000  # 10 kHz
        elif tempo_corrente < 30:
            freq = 1000  # 1 kHz
        elif tempo_corrente < 45:
            freq = 5000  # 5 kHz
        else:
            break  # Encerra a reprodução após 45 segundos

        frequencias_tocadas.append(freq)
        sinal = gera_senoide(freq)
        sd.play(sinal, samplerate=TAXA_AMOSTRAGEM)
        time.sleep(DURACAO_SINAL)
    return frequencias_tocadas

if __name__ == "__main__":
    frequencias = toca_aleatoriamente()
    print("Frequências tocadas em ordem:")
    for freq in frequencias:
        print(f"{freq} Hz")
