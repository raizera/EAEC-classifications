# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from google.colab import files
import pickle
import os
import time

# Baixar recursos do NLTK para processamento de texto em português
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('stemmers/rslp')
except LookupError:
    print("Baixando recursos do NLTK para processamento de texto em português...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('rslp')

# Função para pré-processar o texto
def preprocessar_texto(titulo, resumo):
    """Pré-processa o texto: combina título e resumo, remove caracteres especiais,
    converte para minúsculas, remove stopwords e aplica stemming."""
    # Garantir que ambos são strings
    titulo = "" if not isinstance(titulo, str) else titulo
    resumo = "" if not isinstance(resumo, str) else resumo

    # Combinar título e resumo
    texto_completo = f"{titulo} {resumo}"

    # Converter para minúsculas e remover caracteres especiais
    texto = re.sub(r'[^\w\s]', '', texto_completo.lower())

    # Tokenizar
    tokens = word_tokenize(texto, language='portuguese')

    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]

    # Aplicar stemming
    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return ' '.join(tokens)

# Função para dividir classes separadas por ';' em listas
def dividir_classes(valor):
    if not isinstance(valor, str):
        return []
    # Remove espaços extras e divide pelo delimitador ';'
    return [item.strip() for item in valor.split(';') if item.strip()]

# === PARTE 1: TREINAMENTO DO MODELO ===
def treinar_modelos():
    print("\n=== INICIANDO TREINAMENTO DE MODELOS ===")

    # Carregando o arquivo de treinamento
    print("Por favor, faça o upload do arquivo de dados de treinamento (CSV ou Excel):")
    uploaded = files.upload()

    if not uploaded:
        print("Nenhum arquivo foi carregado.")
        return None, None, None

    # Obter o nome do arquivo carregado
    nome_arquivo = list(uploaded.keys())[0]

    # Carregar o arquivo com base na extensão
    try:
        if nome_arquivo.endswith('.csv'):
            df = pd.read_csv(nome_arquivo)
        elif nome_arquivo.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(nome_arquivo)
        else:
            print("Formato de arquivo não suportado. Use CSV ou Excel.")
            return None, None, None
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return None, None, None

    # Visualizar as primeiras linhas para confirmar a estrutura
    print("\nEstrutura dos dados:")
    print(df.head())

    # Verificar se as colunas necessárias existem
    print("\nColunas disponíveis:", df.columns.tolist())

    # Solicitar nomes das colunas
    coluna_titulo = 'título'
    coluna_resumo = 'resumo'
    coluna_area = 'Área de Ciências'
    coluna_modalidade = 'Modalidade Educacional'
    coluna_nivel = 'Nível Escolar'
    coluna_foco = 'Foco'

    # Verificar se as colunas informadas existem
    colunas_necessarias = [coluna_titulo, coluna_resumo, coluna_area, coluna_modalidade, coluna_nivel, coluna_foco]
    colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]

    if colunas_faltantes:
        print(f"\nATENÇÃO: As seguintes colunas informadas não existem: {', '.join(colunas_faltantes)}")
        return None, None, None

    # Pré-processar o texto combinando título e resumo
    print("\nPré-processando os textos...")
    df['texto_processado'] = df.apply(lambda x: preprocessar_texto(x[coluna_titulo], x[coluna_resumo]), axis=1)

    # Preparar os dados para treinamento
    X = df['texto_processado']

    # Aplicar função de divisão e criar listas para cada categoria
    df['area_lista'] = df[coluna_area].apply(dividir_classes)
    df['modalidade_lista'] = df[coluna_modalidade].apply(dividir_classes)
    df['nivel_lista'] = df[coluna_nivel].apply(dividir_classes)
    df['foco_lista'] = df[coluna_foco].apply(dividir_classes)

    # Verificar os valores únicos em cada categoria
    categorias_unicas = {
        'Área de Ciências': set(),
        'Modalidade Educacional': set(),
        'Nível Escolar': set(),
        'Foco': set()
    }

    for lista in df['area_lista']:
        categorias_unicas['Área de Ciências'].update(lista)

    for lista in df['modalidade_lista']:
        categorias_unicas['Modalidade Educacional'].update(lista)

    for lista in df['nivel_lista']:
        categorias_unicas['Nível Escolar'].update(lista)

    for lista in df['foco_lista']:
        categorias_unicas['Foco'].update(lista)

    # Mostrar valores únicos encontrados
    print("\nValores únicos encontrados em cada categoria:")
    for categoria, valores in categorias_unicas.items():
        print(f"{categoria}: {sorted(valores)}")

    # Criar um binarizador para cada categoria
    mlb_area = MultiLabelBinarizer()
    mlb_modalidade = MultiLabelBinarizer()
    mlb_nivel = MultiLabelBinarizer()
    mlb_foco = MultiLabelBinarizer()

    # Transformar as listas em matrizes binárias
    y_area = mlb_area.fit_transform(df['area_lista'])
    y_modalidade = mlb_modalidade.fit_transform(df['modalidade_lista'])
    y_nivel = mlb_nivel.fit_transform(df['nivel_lista'])
    y_foco = mlb_foco.fit_transform(df['foco_lista'])

    # Guardar as classes para cada categoria
    classes = {
        'Área de Ciências': mlb_area.classes_,
        'Modalidade Educacional': mlb_modalidade.classes_,
        'Nível Escolar': mlb_nivel.classes_,
        'Foco': mlb_foco.classes_
    }

    # Mostrar número de classes por categoria
    print("\nNúmero de classes por categoria:")
    for categoria, cls in classes.items():
        print(f"{categoria}: {len(cls)} classes")
        print(f"Classes: {cls}")

    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_area_train, y_area_test, y_nivel_train, y_nivel_test, \
        y_foco_train, y_foco_test, y_modalidade_train, y_modalidade_test = train_test_split(
            X, y_area, y_nivel, y_foco, y_modalidade, test_size=0.2, random_state=42)

    # Vetorizar o texto usando TF-IDF
    print("\nVetorizando o texto...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Treinar um modelo para cada categoria
    print("\nTreinando modelos de classificação multilabel...")
    classifiers = {}
    binarizers = {
        'Área de Ciências': mlb_area,
        'Modalidade Educacional': mlb_modalidade,
        'Nível Escolar': mlb_nivel,
        'Foco': mlb_foco
    }

    # Dados de treino e teste para cada categoria
    y_train_data = {
        'Área de Ciências': y_area_train,
        'Modalidade Educacional': y_modalidade_train,
        'Nível Escolar': y_nivel_train,
        'Foco': y_foco_train
    }

    y_test_data = {
        'Área de Ciências': y_area_test,
        'Modalidade Educacional': y_modalidade_test,
        'Nível Escolar': y_nivel_test,
        'Foco': y_foco_test
    }

    # Treinar um modelo para cada categoria com tratamento de erro para classes únicas
    for nome_categoria in binarizers.keys():
        print(f"\nTreinando modelo para {nome_categoria}...")

        # Obter os dados de treino para esta categoria
        y_train_cat = y_train_data[nome_categoria]
        y_test_cat = y_test_data[nome_categoria]

        # Verificar se há pelo menos duas classes nos dados de treinamento
        num_classes = y_train_cat.shape[1]
        classes_presentes = np.sum(np.sum(y_train_cat, axis=0) > 0)

        print(f"Número de classes em {nome_categoria}: {num_classes}")
        print(f"Classes presentes no conjunto de treinamento: {classes_presentes}")

        if classes_presentes < 2:
            print(f"AVISO: {nome_categoria} possui menos de duas classes com amostras nos dados de treinamento.")
            print("Usando classificador alternativo (DummyClassifier).")

            # Usar um classificador dummy que sempre prevê a classe mais comum
            clf = MultiOutputClassifier(DummyClassifier(strategy='most_frequent'))
            clf.fit(X_train_vec, y_train_cat)
            classifiers[nome_categoria] = clf
        else:
            # Criar e treinar o modelo normalmente
            try:
                base_clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=10.0)
                clf = MultiOutputClassifier(base_clf)
                clf.fit(X_train_vec, y_train_cat)
                classifiers[nome_categoria] = clf
            except ValueError as e:
                print(f"Erro ao treinar modelo para {nome_categoria}: {e}")
                print("Usando classificador alternativo (DummyClassifier)...")

                # Usar um algoritmo que funciona com classes únicas
                clf = MultiOutputClassifier(DummyClassifier(strategy='most_frequent'))
                clf.fit(X_train_vec, y_train_cat)
                classifiers[nome_categoria] = clf

        # Avaliar modelo
        y_pred = clf.predict(X_test_vec)

        # Calcular métricas com tratamento de erro
        try:
            # Calcular micro F1-score (mais adequado para multilabel)
            micro_f1 = f1_score(y_test_cat, y_pred, average='micro', zero_division=0)
            macro_f1 = f1_score(y_test_cat, y_pred, average='macro', zero_division=0)

            print(f"Micro F1-Score para {nome_categoria}: {micro_f1:.4f}")
            print(f"Macro F1-Score para {nome_categoria}: {macro_f1:.4f}")
        except Exception as e:
            print(f"Erro ao calcular métricas para {nome_categoria}: {e}")

    # Salvar os modelos para uso posterior
    print("\nSalvando modelos, vetorizador e binarizadores...")
    with open('classificador_resumos_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    for nome_categoria, clf in classifiers.items():
        with open(f'classificador_resumos_{nome_categoria.replace(" ", "_").lower()}.pkl', 'wb') as f:
            pickle.dump(clf, f)

    for nome_categoria, binarizer in binarizers.items():
        with open(f'binarizer_{nome_categoria.replace(" ", "_").lower()}.pkl', 'wb') as f:
            pickle.dump(binarizer, f)

    print("\nModelos treinados e salvos com sucesso!")

    return vectorizer, classifiers, binarizers

# === PARTE 2: PROCESSAR ARQUIVO EM LOTE ===
def processar_lote_excel(vectorizer=None, classifiers=None, binarizers=None):
    print("\n=== INICIANDO PROCESSAMENTO EM LOTE ===")

    # Se não recebemos os modelos como parâmetros, precisamos carregá-los
    if vectorizer is None or classifiers is None or binarizers is None:
        try:
            print("Carregando modelos salvos durante o treinamento...")
            # Carregar o vetorizador
            with open('classificador_resumos_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)

            # Categorias que precisamos carregar
            categorias = ['Área de Ciências', 'Modalidade Educacional', 'Nível Escolar', 'Foco']
            classifiers = {}
            binarizers = {}

            # Carregar classificadores e binarizadores para cada categoria
            for categoria in categorias:
                nome_arquivo = categoria.replace(" ", "_").lower()

                # Carregar o classificador se existir
                if os.path.exists(f'classificador_resumos_{nome_arquivo}.pkl'):
                    with open(f'classificador_resumos_{nome_arquivo}.pkl', 'rb') as f:
                        classifiers[categoria] = pickle.load(f)
                else:
                    print(f"AVISO: Classificador para {categoria} não encontrado.")

                # Carregar o binarizador se existir
                if os.path.exists(f'binarizer_{nome_arquivo}.pkl'):
                    with open(f'binarizer_{nome_arquivo}.pkl', 'rb') as f:
                        binarizers[categoria] = pickle.load(f)
                else:
                    print(f"AVISO: Binarizador para {categoria} não encontrado.")
        except Exception as e:
            print(f"Erro ao carregar modelos: {e}")
            return

    # Solicitar o upload do arquivo Excel com os resumos a serem classificados
    print("\nPor favor, faça o upload do arquivo Excel contendo os títulos e resumos:")
    uploaded = files.upload()

    if not uploaded:
        print("Nenhum arquivo foi carregado.")
        return

    nome_arquivo = list(uploaded.keys())[0]
    if not nome_arquivo.endswith(('.xlsx', '.xls')):
        print("O arquivo deve estar no formato Excel (.xlsx ou .xls)")
        return

    # Carregar o arquivo Excel
    try:
        df_input = pd.read_excel(nome_arquivo)
        print(f"\nArquivo carregado com sucesso. Formato do arquivo: {df_input.shape[0]} linhas, {df_input.shape[1]} colunas.")
        print("\nColunas disponíveis:", df_input.columns.tolist())
    except Exception as e:
        print(f"Erro ao ler o arquivo Excel: {e}")
        return

    # Solicitar os nomes das colunas que contêm os títulos e resumos
    coluna_titulo = input("\nDigite o nome da coluna que contém os títulos dos trabalhos: ")
    coluna_resumo = input("Digite o nome da coluna que contém os resumos dos trabalhos: ")

    # Verificar se as colunas existem
    if coluna_titulo not in df_input.columns:
        print(f"ERRO: Coluna '{coluna_titulo}' não encontrada no arquivo.")
        return

    if coluna_resumo not in df_input.columns:
        print(f"ERRO: Coluna '{coluna_resumo}' não encontrada no arquivo.")
        return

    # Função para prever categorias de um texto
    def prever_categorias(titulo, resumo):
        # Pré-processar o texto
        texto_processado = preprocessar_texto(titulo, resumo)

        # Vetorizar o texto
        texto_vec = vectorizer.transform([texto_processado])

        # Fazer previsões para cada categoria
        previsoes = {}

        for nome_categoria, clf in classifiers.items():
            # Verificar se o binarizador correspondente existe
            if nome_categoria not in binarizers:
                previsoes[nome_categoria] = ["Modelo não disponível"]
                continue

            binarizer = binarizers[nome_categoria]

            # Obter previsões binárias
            try:
                y_pred_bin = clf.predict(texto_vec)[0]

                # Converter de volta para nomes de classes
                indices = np.where(y_pred_bin == 1)[0]

                if len(indices) == 0:
                    # Se nenhuma classe for prevista, escolha a mais provável
                    try:
                        y_proba = clf.predict_proba(texto_vec)
                        probabilidades = np.concatenate([prob[:, 1].reshape(-1, 1) for prob in y_proba], axis=1)
                        top_idx = np.argmax(probabilidades)
                        previsao = [binarizer.classes_[top_idx]]
                    except:
                        # Se não for possível obter probabilidades, use a classe mais comum
                        previsao = [binarizer.classes_[0]] if len(binarizer.classes_) > 0 else ["Desconhecido"]
                else:
                    previsao = [binarizer.classes_[idx] for idx in indices]

                previsoes[nome_categoria] = previsao
            except Exception as e:
                print(f"Erro ao fazer previsão para {nome_categoria}: {e}")
                previsoes[nome_categoria] = ["Erro na previsão"]

        return previsoes

    # Processar cada linha do arquivo
    print(f"\nProcessando {df_input.shape[0]} registros...")

    # Criar colunas para os resultados
    categorias = list(classifiers.keys())
    for categoria in categorias:
        df_input[f'Previsto: {categoria}'] = None

    # Iniciar cronômetro
    inicio = time.time()
    registros_processados = 0

    # Processar cada linha
    for idx, row in df_input.iterrows():
        try:
            titulo = row[coluna_titulo]
            resumo = row[coluna_resumo]

            # Fazer previsões
            previsoes = prever_categorias(titulo, resumo)

            # Adicionar resultados ao DataFrame
            for categoria, valores in previsoes.items():
                df_input.at[idx, f'Previsto: {categoria}'] = '; '.join(valores)

            # Atualizar contador
            registros_processados += 1

            # Mostrar progresso a cada 10 registros ou no final
            if registros_processados % 10 == 0 or registros_processados == df_input.shape[0]:
                tempo_decorrido = time.time() - inicio
                registros_por_segundo = registros_processados / tempo_decorrido if tempo_decorrido > 0 else 0
                tempo_restante = (df_input.shape[0] - registros_processados) / registros_por_segundo if registros_por_segundo > 0 else 0

                print(f"Processados {registros_processados}/{df_input.shape[0]} registros "
                      f"({registros_processados/df_input.shape[0]*100:.1f}%) - "
                      f"Velocidade: {registros_por_segundo:.1f} reg/s - "
                      f"Tempo restante estimado: {tempo_restante/60:.1f} minutos")

        except Exception as e:
            print(f"Erro ao processar o registro {idx}: {e}")
            for categoria in categorias:
                df_input.at[idx, f'Previsto: {categoria}'] = "Erro no processamento"

    # Salvar o resultado em um novo arquivo Excel
    nome_arquivo_saida = 'resultados_classificacao.xlsx'
    df_input.to_excel(nome_arquivo_saida, index=False)

    # Disponibilizar para download
    print(f"\nClassificação concluída! Salvando resultados em '{nome_arquivo_saida}'...")
    files.download(nome_arquivo_saida)

    # Resumo dos resultados
    print("\nResumo dos resultados:")
    for categoria in categorias:
        if f'Previsto: {categoria}' in df_input.columns:
            # Contar ocorrências de cada classe
            todas_classes = []
            for valores in df_input[f'Previsto: {categoria}'].dropna():
                todas_classes.extend([v.strip() for v in valores.split(';')])

            contagem = pd.Series(todas_classes).value_counts()

            print(f"\n{categoria}:")
            for classe, count in contagem.items():
                print(f"  {classe}: {count} ocorrências")

# Função principal que integra as duas etapas
def executar_pipeline_completo():
    print("=== SISTEMA DE CLASSIFICAÇÃO AUTOMÁTICA DE RESUMOS ACADÊMICOS ===")
    print("Este script irá treinar modelos e depois processar um arquivo em lote.")

    # Etapa 1: Treinar modelos
    vectorizer, classifiers, binarizers = treinar_modelos()

    if vectorizer is None or classifiers is None or binarizers is None:
        print("\nO treinamento de modelos não foi concluído com sucesso.")
        resposta = input("Deseja tentar carregar modelos existentes e continuar? (s/n): ")
        if resposta.lower() != 's':
            return

    # Etapa 2: Processar arquivo em lote
    processar_lote_excel(vectorizer, classifiers, binarizers)

    print("\n=== PROCESSAMENTO COMPLETO ===")

# Executar o pipeline completo
if __name__ == "__main__":
    executar_pipeline_completo()
