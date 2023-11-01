import os
import librosa
import soundfile as sf
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import tensorflow as tf
import tkinter as tk
from tkinter import Button, Label
import tkinter as tk
from tkinter import Button, Label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import shutil

DURACAO_GRAVACAO = 60  # Duração da gravação em segundos

def gravar_audio(duracao=DURACAO_GRAVACAO, taxa_amostragem=44100):
    print("Gravando áudio...")
    audio_gravado = sd.rec(int(duracao * taxa_amostragem), samplerate=taxa_amostragem, channels=2, dtype='float32')
    sd.wait()
    print("Gravação concluída!")

    '''# Plot da forma de onda do áudio
    time = np.linspace(0, duracao, len(audio_gravado))
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio_gravado)
    plt.title("Forma de Onda do Áudio Captado")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show(block=False)'''

    # Plot do espectrograma do áudio
    plt.figure(figsize=(10, 4))
    plt.specgram(audio_gravado[:, 0], Fs=taxa_amostragem)
    plt.title("Espectrograma do Áudio Captado")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Frequência (Hz)")
    plt.colorbar()
    plt.show(block=False)

    '''# Cria uma figura com dois eixos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot da forma de onda do áudio no primeiro eixo
    time = np.linspace(0, duracao, len(audio_gravado))
    ax1.plot(time, audio_gravado)
    ax1.set_title("Forma de Onda do Áudio Captado")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid()

    # Plot do espectrograma do áudio no segundo eixo
    ax2.specgram(audio_gravado[:, 0], Fs=taxa_amostragem)
    ax2.set_title("Espectrograma do Áudio Captado")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Frequência (Hz)")

    # Ajuste de layout para evitar sobreposição
    plt.tight_layout()

    # Exibe a figura com os dois eixos em modo não-bloqueante
    plt.show(block=False)'''
    
    return audio_gravado, taxa_amostragem

def dividir_audio_em_trechos(y, sr, duracao_trecho=5.0, diretorio_saida="trechos"):
    amostras_por_trecho = int(duracao_trecho * sr)
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
    trechos = []
    for i, inicio in enumerate(range(0, len(y), amostras_por_trecho)):
        fim = inicio + amostras_por_trecho
        trecho = y[inicio:fim]
        nome_trecho = os.path.join(diretorio_saida, f"trecho_{i}.wav")
        sf.write(nome_trecho, trecho, sr)
        trechos.append(nome_trecho)
    return trechos

def extract_spectrograms(file_path, output_dir):
    y, sr = librosa.load(file_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Espectrograma de {os.path.basename(file_path)}')
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, f"extracaoEspec{os.path.splitext(os.path.basename(file_path))[0]}.png")
    plt.savefig(output_filename)
    plt.close()

def carregar_transformada_fourier_direto(diretorio):
    transformadas = []
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith('.png'):
            imagem_path = os.path.join(diretorio, arquivo)
            try:
                imagem = tf.keras.preprocessing.image.load_img(imagem_path, target_size=(200, 500))
                imagem = tf.keras.preprocessing.image.img_to_array(imagem) / 255.0  # Normalização
                transformadas.append(imagem)
            except Exception as e:
                print(f"Erro ao carregar o arquivo {arquivo}: {e}")
    return np.array(transformadas)

def main(output_segments_dir, output_spectrograms_dir):
    # Gravação do áudio
    y, sr = gravar_audio()
    segmentos = dividir_audio_em_trechos(y, sr, diretorio_saida=output_segments_dir)
    for segmento in segmentos:
        extract_spectrograms(segmento, output_spectrograms_dir)

def limpar_diretorios(diretorio):
    # Verifique se o diretório existe
    if os.path.exists(diretorio):
        # Use a função shutil.rmtree() para remover o diretório e seu conteúdo
        shutil.rmtree(diretorio)
        print(f"Conteúdo do diretório {diretorio} removido com sucesso.")
    else:
        print(f"O diretório {diretorio} não existe.")

def iniciar_gravacao():
    def callback():
        # Altere o texto do label para "Gravação em andamento"
        gravar_botao.config(text="Gravação Finalizada")

        main(output_segments_dir, output_spectrograms_dir)
        espectrogramas_capturados = carregar_transformada_fourier_direto(output_spectrograms_dir)
        if espectrogramas_capturados.size == 0:
            resultado_label.config(text="Nenhum espectrograma foi carregado. Verifique o diretório e os arquivos.")
            gravar_botao.after(0, reset_gravar_botao)  # Agende a função para redefinir o botão após 2 segundos
            return
        rotulos_preditos_capturados = np.argmax(model.predict(espectrogramas_capturados), axis=-1)
        frequencias = ["10kHz", "1kHz", "5kHz"]
        frequencias_preditas_capturados = [frequencias[idx] for idx in rotulos_preditos_capturados]

        # Altere o texto do label para "Análise de Frequências"
        status_label.config(text="Análise das Frequências Obtidas")
        resultado_label.config(text="Frequências detectadas nos espectrogramas capturados: " + ", ".join(frequencias_preditas_capturados))
        gravar_botao.after(2000, reset_gravar_botao)  # Agende a função para redefinir o botão após 2 segundos

    def reset_gravar_botao():
        gravar_botao.config(text="Iniciar Gravação")

    # Crie a janela principal
    janela = tk.Tk()
    janela.title("Gravação de Áudio e Análise")

    # Defina as dimensões da janela principal
    largura_janela = 800  # Largura desejada
    altura_janela = 400   # Altura desejada
    janela.geometry(f"{largura_janela}x{altura_janela}")

    # Crie um frame para centralizar o botão e os labels
    frame = tk.Frame(janela)
    frame.pack(expand=True)

    # Crie um botão para iniciar a gravação
    gravar_botao = Button(frame, text="Iniciar Gravação", command=callback)
    gravar_botao.pack(side="top", pady=20)  # pady para adicionar espaço na parte superior

    # Crie um label para mostrar o status
    status_label = Label(frame, text="", font=("Helvetica", 12))
    status_label.pack()

    # Crie uma label para exibir os resultados
    resultado_label = Label(janela, text="")
    resultado_label.pack(fill="both", expand=True, padx=20, pady=10)  # Use fill e expand


    janela.mainloop()

if __name__ == '__main__':
    # Carregar o modelo CNN
    model = tf.keras.models.load_model('C:\\Users\\ldnaj\\OneDrive\\Documents\\Faculdade\\Projeto 2 Meses\\EmissorReceptor\\meu_modelo.h5')

    output_segments_dir = 'C:\\Users\\ldnaj\\OneDrive\\Documents\\Faculdade\\Projeto 2 Meses\\EmissorReceptor\\trechos'
    output_spectrograms_dir = 'C:\\Users\\ldnaj\\OneDrive\\Documents\\Faculdade\\Projeto 2 Meses\\EmissorReceptor\\espectogramas'

    # Limpe os diretórios antes de iniciar a gravação
    limpar_diretorios(output_segments_dir)
    limpar_diretorios(output_spectrograms_dir)

    iniciar_gravacao()
