# Visão Geral do Modelo

Este é um modelo CNN simples para classificação de áudio baseado nos espectrogramas Mel gerados pelo script de preprocessamento.py, o modelo foi projectado para classificar áudios em três classes: background, edjo e sao_mag_cenas.

## Características Principais do Modelo:

1. **Arquitetura**:
   * 3 camadas convolucionais com normalização em batch e max pooling
   * Camadas totalmente conectadas com dropout para regularização
   * Tratamento adaptativo do tamanho de entrada para diferentes dimensões de espectrogramas

2. **Manipulação de Dados**:
   * Dataset personalizado em PyTorch para carregamento e agrupamento de espectrogramas
   * Divisões treino/validação/teste para avaliação adequada
   * Técnicas de aumento de dados podem ser adicionadas se necessário

3. **Processo de Treinamento**:
   * Utiliza optimizador Adam e função de perda cross-entropy
   * Implementa early_stop salvando o melhor modelo
   * Inclui visualização de curvas de aprendizado

4. **Avaliação**:
   * Gera matriz de confusão para visualizar o desempenho específico por classe
   * Produz relatório de classificação com precisão, recall e pontuação F1

## Como Utilizar o Modelo

1. Primeiro, pré-processe seus arquivos de áudio utilizando o script `preprocessamento.py`, que gerará espectrogramas Mel e os salvará como arquivos `.npy`.

2. O script principal espera a seguinte estrutura de diretórios:

```
./audio_data/
├── espectrogramas_background/
│   └── [arquivos de espectrograma com sufixo _melspec.npy]
├── espectrogramas_edjo/
│   └── [arquivos de espectrograma com sufixo _melspec.npy]
└── espectrogramas_sao_mag_cenas/
    └── [arquivos de espectrograma com sufixo _melspec.npy]
```

3. Modifique o parâmetro `base_folder` na função `main()` para direcionar para o seu diretório de dados.

4. Execute o script para treinar e avaliar o modelo:

```bash
python model_cnn.py
```

## Possíveis Melhorias

Para um melhor desempenho, você pode considerar:

1. **Técnicas de aumento de dados** como deslocamento temporal, alteração de tom ou adição de ruído
2. **Arquiteturas mais profundas** ou modelos pré-treinados se você tiver um conjunto de dados maior
3. **Agendamento da taxa de aprendizado** para melhorar a convergência
4. **Validação cruzada** para uma avaliação mais robusta
5. **Ajuste de hiperparâmetros** para optimizar o desempenho do modelo

