import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_mel_spectrogram(audio_path, output_image_path, n_mels=128, hop_length=512, sr=22050):
    """
    Extrai o Mel-spectrograma de uma música e salva como imagem, sem bordas brancas.
    
    :param audio_path: Caminho para o arquivo de áudio.
    :param output_image_path: Caminho para salvar o Mel-spectrograma como imagem.
    :param n_mels: Número de bandas Mel (padrão: 128).
    :param hop_length: Tamanho do salto entre janelas de STFT.
    :param sr: Taxa de amostragem (padrão: 22050 Hz).
    """
    try:
        # Carregar o áudio
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Calcular o Mel-spectrograma
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        
        # Converter para escala logarítmica (dB)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Configurar o gráfico sem bordas
        plt.figure(figsize=(10, 4), dpi=300)  # Aumentar o DPI para maior qualidade
        plt.axis('off')  # Remover os eixos
        librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=hop_length, cmap='viridis')  # Escolha um cmap agradável
        
        # Salvar o gráfico diretamente sem bordas
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        #print(f"Mel-spectrograma salvo em: {output_image_path}")
    
    except Exception as e:
        print(f"Erro ao processar {audio_path}: {e}")


def process_folder(input_folder, output_folder, n_mels=128, hop_length=512, sr=22050):
    """
    Processa todas as músicas de uma pasta, extraindo Mel-espectrogramas.
    
    :param input_folder: Pasta contendo os arquivos de áudio.
    :param output_folder: Pasta para salvar os Mel-espectrogramas.
    :param n_mels: Número de bandas Mel (padrão: 128).
    :param hop_length: Tamanho do salto entre janelas de STFT.
    :param sr: Taxa de amostragem (padrão: 22050 Hz).
    """
    if not os.path.exists(input_folder):
        print(f"Pasta de entrada não encontrada: {input_folder}")
        return
    
    # Criar a pasta de saída se não existir
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterar sobre todos os arquivos na pasta de entrada
    for index, file_name in enumerate(os.listdir(input_folder)):
        # Verificar se é um arquivo de áudio
        if file_name.endswith(('.mp3', '.wav', '.flac', '.ogg')):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
            
            # Verificar se o arquivo já foi processado
            if os.path.exists(output_path):
                print(f"Já processado: {output_path}")
                continue  # Ignorar este arquivo
            
        if(index%10==0):
            print(f"Processando Espectrogamas {index}/{len(os.listdir(input_folder))}")
            extract_mel_spectrogram(input_path, output_path, n_mels=n_mels, hop_length=hop_length, sr=sr)

# Exemplo de uso
input_folder = "musicas"  # Substitua pelo caminho da pasta com músicas
output_folder = "espectrogramas"  # Substitua pelo caminho da pasta para salvar os espectrogramas
process_folder(input_folder, output_folder)