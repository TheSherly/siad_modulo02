import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# =========================================================
# FUNÇÕES DE DISTÂNCIA (Implementação da Letra B)
# =========================================================
def distancia_euclidiana(subseq1, subseq2):
    if len(subseq1) != len(subseq2):
        raise ValueError("Erro: As subsequências devem ter o mesmo tamanho.")
    soma_quadrados = 0
    for i in range(len(subseq1)):
        soma_quadrados += (subseq1[i] - subseq2[i]) ** 2
    return soma_quadrados ** 0.5

def distancia_manhattan(subseq1, subseq2):
    if len(subseq1) != len(subseq2):
        raise ValueError("Erro: As subsequências devem ter o mesmo tamanho.")
    soma_absoluta = 0
    for i in range(len(subseq1)):
        soma_absoluta += abs(subseq1[i] - subseq2[i])
    return soma_absoluta

# =========================================================
# FUNÇÃO DE BUSCA DE SUBSEQUÊNCIAS (Letra C - Versão 1)
# =========================================================
def busca_subsequencias_versao1(serie_completa):
    # 1. Divide a série em 11 partes iguais
    n_partes = 11
    # Pega a parte inteira da divisão para saber quantos dias tem cada bloco
    tamanho_parte = len(serie_completa) // n_partes 
    
    partes = []
    for i in range(n_partes):
        inicio = i * tamanho_parte
        fim = inicio + tamanho_parte
        # Recorta a série e adiciona na nossa lista de partes
        partes.append(serie_completa[inicio:fim])
        
    # 2. Separa a última parte (11ª subsequência) e as 10 anteriores
    subsequencia_atual = partes[10] # No Python, o índice 10 é a 11ª posição
    subsequencias_anteriores = partes[:10]
    
    # Listas para guardar os resultados (índice da parte, distância)
    resultados_euclidiana = []
    resultados_manhattan = []
    
    # 3. Compara a 11ª parte com cada uma das 10 anteriores
    for i in range(len(subsequencias_anteriores)):
        parte_passada = subsequencias_anteriores[i]
        
        dist_euc = distancia_euclidiana(subsequencia_atual, parte_passada)
        dist_man = distancia_manhattan(subsequencia_atual, parte_passada)
        
        # Guarda o número da parte (1 a 10) e a distância calculada
        resultados_euclidiana.append((i + 1, dist_euc))
        resultados_manhattan.append((i + 1, dist_man))
        
    # 4. Ordena os resultados da menor distância para a maior (busca pelos mínimos)
    # A função lambda diz para o Python ordenar pelo segundo item da tupla (a distância)
    resultados_euclidiana.sort(key=lambda x: x[1])
    resultados_manhattan.sort(key=lambda x: x[1])
    
    # 5. Retorna os 3 primeiros (mais próximos)
    return resultados_euclidiana[:3], resultados_manhattan[:3]

# =========================================================
# FUNÇÃO DE BUSCA DE SUBSEQUÊNCIAS (Letra C - Versão 2)
# =========================================================
def busca_subsequencias_versao2(serie_pandas):
    # 1. Divide a série em 11 partes iguais
    n_partes = 11
    tamanho_parte = len(serie_pandas) // n_partes 
    
    partes = []
    for i in range(n_partes):
        inicio = i * tamanho_parte
        fim = inicio + tamanho_parte
        
        # Pega a parte, extrai apenas os valores e converte para uma nova Series.
        # Isso remove as datas originais, resetando o índice. É essencial para
        # que a função .corr() do Pandas compare os valores lado a lado corretamente.
        parte_limpa = pd.Series(serie_pandas.iloc[inicio:fim].values)
        partes.append(parte_limpa)
        
    # 2. Separa a última parte (11ª subsequência) e as 10 anteriores
    subsequencia_atual = partes[10] 
    subsequencias_anteriores = partes[:10]
    
    resultados_pearson = []
    resultados_spearman = []
    
    # 3. Compara a 11ª parte com as 10 anteriores
    for i in range(len(subsequencias_anteriores)):
        parte_passada = subsequencias_anteriores[i]
        
        # Calcula as correlações usando a função nativa do Pandas
        corr_pearson = subsequencia_atual.corr(parte_passada, method='pearson')
        corr_spearman = subsequencia_atual.corr(parte_passada, method='spearman')
        
        resultados_pearson.append((i + 1, corr_pearson))
        resultados_spearman.append((i + 1, corr_spearman))
        
    # 4. Ordena os resultados do MAIOR para o MENOR (reverse=True)
    # Na correlação, quanto mais perto de 1 (ou 100%), mais idênticas são.
    resultados_pearson.sort(key=lambda x: x[1], reverse=True)
    resultados_spearman.sort(key=lambda x: x[1], reverse=True)
    
    # 5. Retorna os 3 primeiros
    return resultados_pearson[:3], resultados_spearman[:3]

# =========================================================
# FUNÇÕES DE PREDIÇÃO (Letra D)
# =========================================================
def predicao_metodo_1_media_proximos(serie_completa, top3_pearson):
    """
    Método 1: Predição com a média dos próximos pontos.
    """
    # Descobre o tamanho de cada parte (total de dados dividido por 11 partes)
    tamanho_parte = len(serie_completa) // 11
    proximos_pontos = []
    
    for parte_num, correlacao in top3_pearson:
        # A parte_num vai de 1 a 10. 
        # Se a parte é a 1 (índices 0 a 9), o "próximo ponto" é o índice 10.
        # Logo, o índice do próximo ponto é sempre (parte_num * tamanho_parte)
        indice_proximo_ponto = parte_num * tamanho_parte
        
        # Pega o valor real que ocorreu logo após essa subsequência no passado
        valor_proximo = serie_completa[indice_proximo_ponto]
        proximos_pontos.append(valor_proximo)
        
    # Calcula a média aritmética dos 3 pontos encontrados
    predicao = sum(proximos_pontos) / len(proximos_pontos)
    
    return predicao, proximos_pontos

# =========================================================
# FUNÇÃO DE PREDIÇÃO (Letra D - Método 2)
# =========================================================
def predicao_metodo_2_media_distancia(serie_completa, top3_pearson):
    """
    Método 2: Predição somando a média das distâncias ao último ponto atual.
    """
    tamanho_parte = len(serie_completa) // 11
    distancias = []
    
    for parte_num, correlacao in top3_pearson:
        # Índice do dia seguinte à subsequência
        indice_proximo_ponto = parte_num * tamanho_parte
        # Índice do último dia da subsequência
        indice_ultimo_ponto = indice_proximo_ponto - 1
        
        valor_proximo = serie_completa[indice_proximo_ponto]
        valor_ultimo = serie_completa[indice_ultimo_ponto]
        
        # Calcula o "salto" (a distância) que ocorreu na época
        distancia = valor_proximo - valor_ultimo
        distancias.append(distancia)
        
    # Tira a média desses 3 saltos
    media_distancias = sum(distancias) / len(distancias)
    
    # Pega o último ponto da nossa série atual (o "hoje")
    ultimo_ponto_atual = serie_completa[-1]
    
    # A predição é o nosso valor atual somado à média dos saltos históricos
    predicao = ultimo_ponto_atual + media_distancias
    
    return predicao, distancias

# =========================================================
# FUNÇÃO DE PREDIÇÃO (Letra D - Método 3)
# =========================================================
def predicao_metodo_3_media_distancia_relativa(serie_completa, top3_pearson):
    """
    Método 3: Predição aplicando a média da distância relativa (percentual de mudança).
    """
    tamanho_parte = len(serie_completa) // 11
    distancias_relativas = []
    
    for parte_num, correlacao in top3_pearson:
        indice_proximo_ponto = parte_num * tamanho_parte
        indice_ultimo_ponto = indice_proximo_ponto - 1
        
        valor_proximo = serie_completa[indice_proximo_ponto]
        valor_ultimo = serie_completa[indice_ultimo_ponto]
        
        # Calcula a distância relativa (a taxa de crescimento/queda)
        # Ex: se era 10 e foi pra 11, o salto foi 1, e a relativa é 1/10 = 0.1 (10%)
        distancia_relativa = (valor_proximo - valor_ultimo) / valor_ultimo
        distancias_relativas.append(distancia_relativa)
        
    # Tira a média dessas taxas
    media_dist_relativa = sum(distancias_relativas) / len(distancias_relativas)
    
    ultimo_ponto_atual = serie_completa[-1]
    
    # Aplica a taxa de crescimento ao ponto atual (multiplica por 1 + a taxa)
    predicao = ultimo_ponto_atual * (1 + media_dist_relativa)
    
    return predicao, distancias_relativas

# =========================================================
# CARREGAMENTO DOS DADOS (Comum para Letra A e B)
# =========================================================

# Carregar o dataset
df = pd.read_csv('temperaturas.csv')


# Prepara a data para a Letra A
# Preparação dos dados: converter para datetime e colocar como índice
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# =========================================================
# RESOLUÇÃO DA LETRA A (Decomposição)
# =========================================================
print("Gerando o gráfico da Letra (a)...")
print("ATENÇÃO: Feche a janela do gráfico para o código continuar e exibir a Letra (b)!")

# Realizar a decomposição da série temporal
# Como são dados diários, usamos period=7 para testar uma sazonalidade semanal
decomposicao = seasonal_decompose(df['meantemp'], model='additive', period=7)

# Plotar os componentes (Tendência, Sazonalidade e Ruído)
fig = decomposicao.plot()
fig.set_size_inches(10, 8)

# Ajuste visual
plt.suptitle('Decomposição da Série Temporal de Temperaturas', fontsize=14)
plt.tight_layout()

# O código vai "pausar" nesta linha até você fechar a janela do gráfico
plt.show()

# =========================================================
# RESOLUÇÃO DA LETRA B (Cálculo com Subsequências)
# =========================================================
# Extrai a lista de temperaturas do dataframe
serie_completa = df['meantemp'].tolist()

# Define o tamanho e a posição das janelas (ajuste conforme o que o professor pedir)
tamanho_janela = 7 
inicio_janela_A = 0
inicio_janela_B = 14

subsequencia_A = serie_completa[inicio_janela_A : inicio_janela_A + tamanho_janela]
subsequencia_B = serie_completa[inicio_janela_B : inicio_janela_B + tamanho_janela]

dist_euc = distancia_euclidiana(subsequencia_A, subsequencia_B)
dist_man = distancia_manhattan(subsequencia_A, subsequencia_B)

print("\n" + "="*50)
print("RESULTADOS DA LETRA (b)")
print("="*50)
print(f"Subsequência A: {subsequencia_A}")
print(f"Subsequência B: {subsequencia_B}")
print("-" * 50)
print(f"Distância Euclidiana: {dist_euc:.4f}")
print(f"Distância de Manhattan: {dist_man:.4f}")

# =========================================================
# EXECUÇÃO DA LETRA C (Versão 1 - Subsequências)
# =========================================================
# Extrai a lista de temperaturas do dataframe
serie_completa = df['meantemp'].tolist()

# Chama a nossa função que faz todo o trabalho pesado
top3_euc, top3_man = busca_subsequencias_versao1(serie_completa)

print("\n" + "="*50)
print("RESULTADOS DA LETRA (c) - Versão 1")
print("="*50)

print("\nTop 3 Subsequências Mais Próximas (Distância EUCLIDIANA):")
for posicao, (parte_num, distancia) in enumerate(top3_euc, start=1):
    print(f"{posicao}º lugar: Parte {parte_num:02d} com distância de {distancia:.4f}")

print("\nTop 3 Subsequências Mais Próximas (Distância de MANHATTAN):")
for posicao, (parte_num, distancia) in enumerate(top3_man, start=1):
    print(f"{posicao}º lugar: Parte {parte_num:02d} com distância de {distancia:.4f}")
print("="*50)

# =========================================================
# EXECUÇÃO DA LETRA C (Versão 2 - Correlação Pandas)
# =========================================================
# Na versão 2, passamos a coluna original do dataframe (que é uma Pandas Series)
serie_pandas = df['meantemp']

# Chama a função e recebe os rankings
top3_pearson, top3_spearman = busca_subsequencias_versao2(serie_pandas)

print("\n" + "="*50)
print("RESULTADOS DA LETRA (c) - Versão 2")
print("="*50)

print("\nTop 3 Subsequências Mais Próximas (Correlação de PEARSON):")
for posicao, (parte_num, correlacao) in enumerate(top3_pearson, start=1):
    print(f"{posicao}º lugar: Parte {parte_num:02d} com correlação de {correlacao:.4f}")

print("\nTop 3 Subsequências Mais Próximas (Correlação de SPEARMAN):")
for posicao, (parte_num, correlacao) in enumerate(top3_spearman, start=1):
    print(f"{posicao}º lugar: Parte {parte_num:02d} com correlação de {correlacao:.4f}")
print("="*50)

# =========================================================
# EXECUÇÃO DA LETRA D (Método 1)
# =========================================================
# Executa a função usando a série completa e o top 3 do Pearson que já calculamos
predicao_m1, pontos_m1 = predicao_metodo_1_media_proximos(serie_completa, top3_pearson)

print("\n" + "="*50)
print("RESULTADOS DA LETRA (d) - PREDIÇÕES")
print("="*50)

print("\nMÉTODO 1: Média dos Próximos Pontos")
print(f"Os próximos pontos após as 3 melhores subsequências foram: {[round(x, 2) for x in pontos_m1]}")
print(f"-> VALOR PREDITO (Média): {predicao_m1:.4f}")

# =========================================================
# EXECUÇÃO DA LETRA D (Método 2)
# =========================================================
predicao_m2, distancias_m2 = predicao_metodo_2_media_distancia(serie_completa, top3_pearson)

print("\nMÉTODO 2: Média da Distância")
print(f"As distâncias (saltos) encontrados no passado foram: {[round(x, 2) for x in distancias_m2]}")
print(f"Último ponto atual da série: {serie_completa[-1]:.4f}")
print(f"-> VALOR PREDITO (Último Ponto + Média das Distâncias): {predicao_m2:.4f}")

# =========================================================
# EXECUÇÃO DA LETRA D (Método 3)
# =========================================================
predicao_m3, distancias_m3 = predicao_metodo_3_media_distancia_relativa(serie_completa, top3_pearson)

print("\nMÉTODO 3: Média da Distância Relativa")
print(f"As distâncias relativas (taxas) no passado foram: {[round(x, 2) for x in distancias_m3]}")
print(f"Último ponto atual da série: {serie_completa[-1]:.4f}")
print(f"-> VALOR PREDITO (Último Ponto aplicado à Taxa Média): {predicao_m3:.4f}")
print("="*50 + "\n")