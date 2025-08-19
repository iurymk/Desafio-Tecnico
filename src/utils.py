"""
Módulo de utilitários para análise de dados e visualização.

Este módulo contém funções para:
- Geração de gráficos de análise exploratória
- Utilitários de visualização
- Funções auxiliares para processamento de dados
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Optional
import os
import logging


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analysis_plots(data, features, histplot=True, barplot=False, mean=None, text_y=0.5,
                             outliers=False, boxplot=False, boxplot_x=None, kde=False, hue=None,
                             nominal=False, color='#023047', figsize=(24, 12), palette=None,
                             save_separate_files=False, save_path='.'):
    """
    Gera e exibe ou salva múltiplos plots de análise a partir de um DataFrame.

    Args:
        data (pd.DataFrame): O DataFrame com os dados.
        features (list): Lista de colunas (features) a serem plotadas.
        ... (demais parâmetros de plotagem)
        save_separate_files (bool): Se True, salva cada subplot como um arquivo de imagem separado.
                                    Se False, exibe todos os subplots em uma única janela.
        save_path (str): O caminho do diretório onde as figuras serão salvas.
    """
    # --- FLUXO PARA SALVAR CADA PLOT EM UM ARQUIVO SEPARADO ---
    if save_separate_files:
        # Primeiro salva os arquivos individuais
        os.makedirs(save_path, exist_ok=True)
        
        # Determina o tipo de plot para nomear o arquivo
        if barplot: plot_type = 'barplot'
        elif outliers: plot_type = 'outliers_boxplot'
        elif boxplot: plot_type = 'distribution_boxplot'
        else: plot_type = 'histplot'

        for feature in features:
            # Cria uma figura e eixo para cada feature
            fig, ax = plt.subplots(figsize=(8, 6))

            # Lógica de plotagem - mesma lógica dos subplots
            if barplot:
                if mean:
                    data_grouped = data.groupby([feature])[[mean]].mean().reset_index()
                    data_grouped[mean] = round(data_grouped[mean], 2)
                    ax.barh(y=data_grouped[feature].astype(str), width=data_grouped[mean], color=color)
                    for index, value in enumerate(data_grouped[mean]):
                        ax.text(value + text_y, index, f'{value:.1f}', va='center', fontsize=12)
                
                else: # Barplot de contagem/percentual com hue
                     if hue:
                        data_grouped = data.groupby([feature, hue]).size().reset_index(name='count')
                        data_grouped['pct'] = (data_grouped['count'] / data_grouped.groupby(feature)['count'].transform('sum') * 100)
                        pivot_data = data_grouped.pivot(index=feature, columns=hue, values='pct').fillna(0)
                        pivot_data.plot(kind='barh', stacked=True, ax=ax, color=palette)
                        ax.legend(title=hue)
                        for container in ax.containers:
                            labels = [f'{w:.1f}%' if (w := v.get_width()) > 0 else '' for v in container]
                            ax.bar_label(container, labels=labels, label_type='center', fontsize=10)
                     else:
                        data_grouped = data[feature].value_counts(normalize=True).mul(100).reset_index()
                        data_grouped.columns = [feature, 'pct']
                        ax.barh(y=data_grouped[feature].astype(str), width=data_grouped['pct'], color=color)
                        for index, value in enumerate(data_grouped['pct']):
                            ax.text(value + text_y, index, f'{value:.1f}%', va='center', fontsize=12)

                ax.get_xaxis().set_visible(False)
                for spine in ['top', 'right', 'bottom', 'left']:
                    ax.spines[spine].set_visible(False)

            elif outliers:
                sns.boxplot(data=data, x=feature, ax=ax, color=color)
            elif boxplot:
                sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax, palette=palette)
            else: # histplot
                sns.histplot(data=data, x=feature, kde=kde, ax=ax, color=color, stat='proportion', hue=hue)

            ax.set_title(feature, fontsize=14, weight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(axis='both', which='major', labelsize=12)
            fig.tight_layout()

            # Salva a figura
            filename = f"{feature.replace(' ', '_')}_{plot_type}.png"
            full_path = os.path.join(save_path, filename)
            
            try:
                fig.savefig(full_path, dpi=150, bbox_inches='tight')
            except Exception as e:
                print(f"Falha ao salvar '{feature}': {e}")

            plt.close(fig)
        


    # --- FLUXO ORIGINAL: MOSTRAR TODOS OS PLOTS JUNTOS ---
    try:
        num_features = len(features)
        num_cols = 3
        num_rows = num_features // num_cols + (num_features % num_cols > 0)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]

            # A lógica de plotagem aqui é a mesma, mas com rótulos internos para visualização direta
            if barplot:
                if mean:
                    data_grouped = data.groupby([feature])[[mean]].mean().reset_index()
                    data_grouped[mean] = round(data_grouped[mean], 2)
                    ax.barh(y=data_grouped[feature].astype(str), width=data_grouped[mean], color=color)
                    for index, value in enumerate(data_grouped[mean]):
                        ax.text(value + text_y, index, f'{value:.1f}', va='center', fontsize=12)
                
                else: # Barplot de contagem/percentual com hue
                     if hue:
                        data_grouped = data.groupby([feature, hue]).size().reset_index(name='count')
                        data_grouped['pct'] = (data_grouped['count'] / data_grouped.groupby(feature)['count'].transform('sum') * 100)
                        pivot_data = data_grouped.pivot(index=feature, columns=hue, values='pct').fillna(0)
                        pivot_data.plot(kind='barh', stacked=True, ax=ax, color=palette)
                        ax.legend(title=hue)
                        for container in ax.containers:
                            labels = [f'{w:.1f}%' if (w := v.get_width()) > 0 else '' for v in container]
                            ax.bar_label(container, labels=labels, label_type='center', fontsize=10)
                     else:
                        data_grouped = data[feature].value_counts(normalize=True).mul(100).reset_index()
                        data_grouped.columns = [feature, 'pct']
                        ax.barh(y=data_grouped[feature].astype(str), width=data_grouped['pct'], color=color)
                        for index, value in enumerate(data_grouped['pct']):
                            ax.text(value + text_y, index, f'{value:.1f}%', va='center', fontsize=12)

                ax.get_xaxis().set_visible(False)
                for spine in ['top', 'right', 'bottom', 'left']:
                    ax.spines[spine].set_visible(False)

            elif outliers:
                sns.boxplot(data=data, x=feature, ax=ax, color=color)
            elif boxplot:
                sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax, palette=palette)
            else: # histplot
                sns.histplot(data=data, x=feature, kde=kde, ax=ax, color=color, stat='proportion', hue=hue)

            ax.set_title(feature)
            ax.set_xlabel('')
            ax.set_ylabel('')

        # Esconde os eixos não utilizados
        for j in range(num_features, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Ocorreu um erro ao gerar os gráficos: {e}")

# Se a função não estiver funcionando, criar uma versão inline
def plot_histogram_outliers_inline(variable, data):
    """
    Versão inline da função plot_histogram_outliers
    """
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))
    
    # Gera o histograma com a linha de densidade (KDE)
    sns.histplot(x=variable, data=data, ax=axes[0], kde=True)
    
    # Gera o boxplot
    sns.boxplot(x=variable, data=data, ax=axes[1])
    
    return fig, axes

def check_outliers(data: pd.DataFrame, features: List[str]) -> Tuple[Dict[str, List[int]], Dict[str, int], int]:
    """
    Verifica outliers nas features especificadas do dataset.

    Esta função calcula e identifica outliers nas features especificadas
    usando o método do Intervalo Interquartil (IQR).

    Args:
        data (DataFrame): O DataFrame contendo os dados para verificar outliers.
        features (list): Uma lista de nomes de features para verificar outliers.

    Returns:
        tuple: Uma tupla contendo três elementos:
            - outlier_indexes (dict): Um dicionário mapeando nomes de features para listas de índices de outliers.
            - outlier_counts (dict): Um dicionário mapeando nomes de features para a contagem de outliers.
            - total_outliers (int): A contagem total de outliers no dataset.

    Raises:
        Exception: Se ocorrer um erro ao verificar outliers.
    """
    try:
        outlier_counts = {}
        outlier_indexes = {}
        total_outliers = 0

        for feature in features:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            feature_outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
            outlier_indexes[feature] = feature_outliers.index.tolist()
            outlier_count = len(feature_outliers)
            outlier_counts[feature] = outlier_count
            total_outliers += outlier_count

        logger.info(f'Existem {total_outliers} outliers no dataset.')
        logger.info('Número (percentual) de outliers por feature:')
        
        for feature, count in outlier_counts.items():
            percentage = round(count/len(data)*100, 2)
            logger.info(f'{feature}: {count} ({percentage}%)')

        return outlier_indexes, outlier_counts, total_outliers

    except Exception as e:
        logger.error(f"Erro ao verificar outliers: {e}")
        raise


def outliers_search(data: pd.DataFrame, variable: str, save_plot: bool = False, path: str = '') -> None:
    """
    Gera um histograma com KDE e um boxplot para visualizar a distribuição e os
    outliers de uma variável, com a opção de salvar o gráfico como PNG.

    Args:
        data (pd.DataFrame): O DataFrame contendo os dados.
        variable (str): O nome da coluna (variável) a ser plotada.
        save_plot (bool, optional): Se True, salva o gráfico em um arquivo PNG.
                                    Default é False.
        path (str, optional): O caminho da pasta onde o arquivo será salvo.
                              Default é o diretório atual.
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(17, 5))

        # Gera o histograma com a linha de KDE
        sns.histplot(x=variable, data=data, ax=axes[0], kde=True)

        # Gera o boxplot
        sns.boxplot(x=variable, data=data, ax=axes[1])

        # Lógica para salvar o gráfico
        if save_plot:
            # Garante que o diretório de destino exista
            if path and not os.path.exists(path):
                os.makedirs(path)
                logger.info(f"Diretório criado: {path}")

            # Cria o nome do arquivo e o salva
            filename = os.path.join(path, f'analise_outliers_{variable}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em: {filename}")

        # Mostra o gráfico e depois fecha a figura para liberar memória
        plt.show()
        plt.close(fig)

    except Exception as e:
        logger.error(f"Erro na função outliers_search: {e}")
        raise


def proportion_z_test(dfx, lista, z_base):
    """
    Realiza teste Z de proporção para 2 amostras.
    
    Args:
        dfx (pd.DataFrame): DataFrame com dados das duas amostras
        lista (list): Lista com nomes das amostras
        z_base (float): Valor crítico do Z para comparação
    """
    # Calculating information of Sample 1:
    churn_1 = dfx['Yes'][0]
    n_1 = dfx.sum(axis=1)[0]
    proportion_1 = churn_1 / n_1

    # Calculating information of Sample 2:
    churn_2 = dfx['Yes'][1]
    n_2 = dfx.sum(axis=1)[1]
    proportion_2 = churn_2 / n_2

    # Creating a table with stats of samples
    df_stats = pd.DataFrame([[churn_1,n_1,proportion_1],
                           [churn_2,n_2,proportion_2]],
                           index = lista,
                           columns = ['X','n','proportion'])
    print(df_stats)
    print('\n')

    # Cálculo correto da proporção ponderada
    p_bar = (churn_1 + churn_2) / (n_1 + n_2)

    z = (proportion_1 - proportion_2)/np.sqrt(p_bar*(1-p_bar)*(1/n_1+1/n_2))

    if np.absolute(z) > z_base:
        print('Since p-value < alpha, H0 is can be rejected!')
    elif np.absolute(z) < z_base:
        print("Since p-value > alpha, we can't reject H0.")
    
    return z


def mean_z_test(dfx, lista):
    """
    Realiza teste Z de média para 2 amostras.
    
    Args:
        dfx (pd.DataFrame): DataFrame com dados das duas amostras
        lista (list): Lista com nomes das amostras
    """
    # Normalizing and calculating information of Sample 1:
    churn_1 = dfx['Yes'][0] / dfx.sum(axis=1)[0]
    n_1 = dfx.sum(axis=1)[0]
    sigma_1 = np.sqrt(churn_1*(1-churn_1))

    # Calculating information of Sample 2:
    churn_2 = dfx['Yes'][1] / dfx.sum(axis=1)[1]
    n_2 = dfx.sum(axis=1)[1]
    sigma_2 = np.sqrt(churn_2*(1-churn_2))

    # Creating a table with stats of samples
    df_stats = pd.DataFrame([[churn_1,n_1,sigma_1],
                             [churn_2,n_2,sigma_2]],
                             index = lista,
                             columns = ['X_bar','n','sigma'])
    print(df_stats)
    print('\n')

    z = (churn_1-churn_2)/np.sqrt(((sigma_1**2)/n_1)+((sigma_2**2)/n_2))
    print('The z-score found is {};'.format(z))
    
    return z
