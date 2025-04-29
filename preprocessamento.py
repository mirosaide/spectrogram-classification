import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import librosa
import librosa.display
import tempfile

def carregar_audio(caminho_arquivo, taxa_amostragem=22050):
    """
    Carrega um arquivo de áudio e converte para formato numpy.
    """
    # Verifica se o arquivo existe
    if not os.path.exists(caminho_arquivo):
        print(f"Erro: O arquivo {caminho_arquivo} não existe.")
        return None, None
    
    print(f"Carregando: {os.path.basename(caminho_arquivo)}")
    
    # Arquivo temporário para conversão
    temp_wav = None
    
    try:
        # Carrega com pydub (funciona para AAC e MP3)
        audio = AudioSegment.from_file(caminho_arquivo)
        
        # Converte para mono se for estéreo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Cria arquivo WAV temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_wav = tmp.name
            audio.export(temp_wav, format="wav")
        
        # Carrega com librosa na taxa de amostragem desejada
        y, sr = librosa.load(temp_wav, sr=taxa_amostragem)
        
    except Exception as e:
        print(f"Erro ao processar o áudio: {e}")
        return None, None
        
    finally:
        # Remove o arquivo temporário
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
    
    return y, sr

def criar_espectrograma(y, sr, n_mels=128):
    """
    Cria um espectrograma de Mel a partir do sinal de áudio.
    """
    if y is None:
        return None
    
    # Gera o espectrograma de Mel
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    
    # Converte para dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def padronizar_audio(y, sr, duracao=3.0):
    """
    Ajusta o áudio para uma duração específica.
    """
    if y is None:
        return None
    
    # Calcula número de amostras para a duração desejada
    amostras_desejadas = int(sr * duracao)
    
    # Corta ou preenche o áudio
    if len(y) > amostras_desejadas:
        y_padronizado = y[:amostras_desejadas]  # Corta
    else:
        # Preenche com zeros
        pad_width = amostras_desejadas - len(y)
        y_padronizado = np.pad(y, (0, pad_width), 'constant')
    
    return y_padronizado

def visualizar_audio(y, sr, mel_spec, titulo="Análise de Áudio"):
    """
    Visualiza a forma de onda e o espectrograma.
    """
    if y is None or mel_spec is None:
        print("Dados inválidos para visualização.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Forma de onda
    plt.subplot(2, 1, 1)
    plt.title("Forma de Onda")
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    
    # Espectrograma de Mel
    plt.subplot(2, 1, 2)
    plt.title("Espectrograma de Mel")
    librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.suptitle(titulo, fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    plt.show()

def processar_pasta(pasta, duracao=3.0, visualizar=False, salvar=False, pasta_saida=None):
    """
    Processa todos os arquivos de áudio em uma pasta.
    """
    # Verifica se a pasta existe
    if not os.path.exists(pasta):
        print(f"Erro: A pasta {pasta} não existe.")
        return None
    
    # Lista arquivos de áudio
    arquivos = [f for f in os.listdir(pasta) 
               if f.lower().endswith(('.aac', '.mp3', '.wav', '.m4a'))]
    
    if not arquivos:
        print(f"Nenhum arquivo de áudio encontrado na pasta {pasta}.")
        return None
    
    print(f"Encontrados {len(arquivos)} arquivos de áudio.")
    
    # Cria pasta de saída se necessário
    if salvar and pasta_saida and not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
    
    # Armazena os espectrogramas
    espectrogramas = []
    nomes_arquivos = []
    
    # Processa cada arquivo
    for arquivo in arquivos:
        caminho_completo = os.path.join(pasta, arquivo)
        nome_base = os.path.splitext(arquivo)[0]
        
        # Carrega e processa o áudio
        y, sr = carregar_audio(caminho_completo)
        if y is None:
            continue
        
        # Padroniza a duração
        y = padronizar_audio(y, sr, duracao)
        
        # Gera o espectrograma
        mel_spec = criar_espectrograma(y, sr)
        
        # Visualiza se solicitado
        if visualizar:
            visualizar_audio(y, sr, mel_spec, titulo=f"Análise: {arquivo}")
        
        # Salva o espectrograma se solicitado
        if salvar and pasta_saida:
            # Salva como arquivo numpy
            np.save(os.path.join(pasta_saida, f"{nome_base}_melspec.npy"), mel_spec)
            
            # Salva como imagem para visualização
            plt.figure(figsize=(10, 4))
            plt.title(f"Espectrograma: {arquivo}")
            librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(os.path.join(pasta_saida, f"{nome_base}_melspec.png"), dpi=200)
            plt.close()
        
        # Adiciona aos resultados
        espectrogramas.append(mel_spec)
        nomes_arquivos.append(arquivo)
        
        print(f"Processado: {arquivo}")
    
    print(f"Concluído: {len(espectrogramas)} arquivos processados com sucesso.")
    
    return {
        'espectrogramas': espectrogramas,
        'nomes_arquivos': nomes_arquivos
    }

def processar_classes(pasta_base, classes=['background', 'edjo', 'sao_mag_cenas'], 
                    duracao=3.0, salvar=True):
    """
    Processa os áudios de cada classe e salva os espectrogramas.
    """
    resultados = {}
    
    for classe in classes:
        pasta_classe = os.path.join(pasta_base, classe)
        
        if not os.path.exists(pasta_classe):
            print(f"Aviso: Pasta para a classe '{classe}' não encontrada ({pasta_classe}).")
            continue
        
        print(f"\n=== Processando classe: {classe} ===")
        
        # Cria pasta de saída para os espectrogramas
        pasta_saida = os.path.join(pasta_base, f"espectrogramas_{classe}")
        
        # Processa a pasta da classe
        resultado = processar_pasta(
            pasta_classe, 
            duracao=duracao, 
            visualizar=False, 
            salvar=salvar, 
            pasta_saida=pasta_saida
        )
        
        if resultado:
            resultados[classe] = resultado
            print(f"Classe '{classe}': {len(resultado['espectrogramas'])} amostras processadas.")
        else:
            print(f"Nenhum resultado para a classe '{classe}'.")
    
    return resultados

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Processador de áudio simples para 3 classes")
    parser.add_argument("pasta", help="Pasta base contendo as subpastas das classes")
    parser.add_argument("--duracao", type=float, default=3.0, help="Duração padrão em segundos")
    parser.add_argument("--visualizar", action="store_true", help="Visualizar alguns exemplos")
    
    args = parser.parse_args()
    
    if args.visualizar:
        # Visualiza alguns exemplos de cada classe
        classes = ['background', 'edjo', 'sao_mag_cenas']
        
        for classe in classes:
            pasta_classe = os.path.join(args.pasta, classe)
            
            if not os.path.exists(pasta_classe):
                continue
            
            # Lista arquivos
            arquivos = [f for f in os.listdir(pasta_classe) 
                       if f.lower().endswith(('.aac', '.mp3', '.wav', '.m4a'))]
            
            if not arquivos:
                continue
            
            # Seleciona um arquivo aleatório para visualizar
            import random
            arquivo = random.choice(arquivos)
            caminho = os.path.join(pasta_classe, arquivo)
            
            print(f"\nVisualizando exemplo da classe '{classe}': {arquivo}")
            y, sr = carregar_audio(caminho)
            if y is not None:
                y = padronizar_audio(y, sr, args.duracao)
                mel_spec = criar_espectrograma(y, sr)
                visualizar_audio(y, sr, mel_spec, titulo=f"Classe: {classe}")
    else:
        # Processa todas as classes
        resultados = processar_classes(args.pasta, duracao=args.duracao)
        print("\nProcessamento concluído!")