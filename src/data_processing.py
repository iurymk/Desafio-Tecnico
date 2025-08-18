"""
Módulo para processamento de dados do projeto de análise de churn.

Este módulo contém funções para:
- Carregamento de dados
- Limpeza e tratamento de dados
- Merge de datasets
- Feature engineering
- Preparação para modelagem
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(data_path: str = "../data/01_raw/") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega os dados brutos dos três arquivos CSV.
    
    Args:
        data_path: Caminho para o diretório com os dados brutos
        
    Returns:
        Tuple com os três DataFrames carregados
    """
    try:
        customer_original = pd.read_csv(f"{data_path}/customer_original.csv")
        customer_social = pd.read_csv(f"{data_path}/customer_social.csv")
        customer_nps = pd.read_csv(f"{data_path}/customer_nps.csv")
        
        logger.info(f"Dados carregados com sucesso:")
        logger.info(f"- customer_original: {customer_original.shape}")
        logger.info(f"- customer_social: {customer_social.shape}")
        logger.info(f"- customer_nps: {customer_nps.shape}")
        
        return customer_original, customer_social, customer_nps
    
    except FileNotFoundError as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado ao carregar dados: {e}")
        raise


def clean_customer_original(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e trata o dataset customer_original.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame limpo
    """
    df_clean = df.copy()
    
    # Verificar e tratar valores duplicados
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Removendo {duplicates} registros duplicados")
        df_clean = df_clean.drop_duplicates()
    
    # Verificar valores nulos
    null_counts = df_clean.isnull().sum()
    if null_counts.sum() > 0:
        logger.info("Valores nulos encontrados:")
        logger.info(null_counts[null_counts > 0])
    
    # Tratar tipos de dados
    # TODO: Implementar tratamento específico baseado na análise exploratória
    
    return df_clean


def clean_customer_social(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e trata o dataset customer_social.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame limpo
    """
    df_clean = df.copy()
    
    # Verificar e tratar valores duplicados
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Removendo {duplicates} registros duplicados")
        df_clean = df_clean.drop_duplicates()
    
    # Verificar valores nulos
    null_counts = df_clean.isnull().sum()
    if null_counts.sum() > 0:
        logger.info("Valores nulos encontrados:")
        logger.info(null_counts[null_counts > 0])
    
    return df_clean


def clean_customer_nps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e trata o dataset customer_nps.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame limpo
    """
    df_clean = df.copy()
    
    # Verificar e tratar valores duplicados
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Removendo {duplicates} registros duplicados")
        df_clean = df_clean.drop_duplicates()
    
    # Verificar valores nulos
    null_counts = df_clean.isnull().sum()
    if null_counts.sum() > 0:
        logger.info("Valores nulos encontrados:")
        logger.info(null_counts[null_counts > 0])
    
    return df_clean


def merge_datasets(customer_original: pd.DataFrame, 
                  customer_social: pd.DataFrame, 
                  customer_nps: pd.DataFrame,
                  merge_strategy: str = "inner") -> pd.DataFrame:
    """
    Faz o merge dos três datasets.
    
    Args:
        customer_original: DataFrame com dados originais
        customer_social: DataFrame com dados sociais
        customer_nps: DataFrame com dados NPS
        merge_strategy: Estratégia de merge ('inner', 'left', 'right', 'outer')
        
    Returns:
        DataFrame com dados mergeados
    """
    try:
        # Assumindo que há uma coluna comum para fazer o merge
        # TODO: Identificar a coluna chave baseado na análise exploratória
        
        # Exemplo de merge (ajustar conforme necessário)
        merged_df = customer_original.merge(
            customer_social, 
            how=merge_strategy,
            on='customer_id'  # Ajustar nome da coluna
        )
        
        merged_df = merged_df.merge(
            customer_nps,
            how=merge_strategy,
            on='customer_id'  # Ajustar nome da coluna
        )
        
        logger.info(f"Merge concluído. Shape final: {merged_df.shape}")
        
        return merged_df
    
    except Exception as e:
        logger.error(f"Erro no merge dos datasets: {e}")
        raise


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas para modelagem.
    
    Args:
        df: DataFrame base
        
    Returns:
        DataFrame com features adicionais
    """
    df_features = df.copy()
    
    # TODO: Implementar criação de features baseada na análise exploratória
    # Exemplos de features que podem ser criadas:
    # - Idade do cliente
    # - Tempo de relacionamento
    # - Valor médio de transações
    # - Frequência de uso
    # - Features de engajamento social
    
    logger.info("Features criadas com sucesso")
    
    return df_features


def prepare_for_modeling(df: pd.DataFrame, target_column: str = "churn") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara dados para modelagem, separando features e target.
    
    Args:
        df: DataFrame com todos os dados
        target_column: Nome da coluna target
        
    Returns:
        Tuple com features (X) e target (y)
    """
    try:
        # Verificar se a coluna target existe
        if target_column not in df.columns:
            raise ValueError(f"Coluna target '{target_column}' não encontrada")
        
        # Separar features e target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Preparação concluída. Features: {X.shape}, Target: {y.shape}")
        
        return X, y
    
    except Exception as e:
        logger.error(f"Erro na preparação para modelagem: {e}")
        raise


def save_processed_data(df: pd.DataFrame, output_path: str = "../data/02_processed/merged_customer_data.csv"):
    """
    Salva dados processados.
    
    Args:
        df: DataFrame processado
        output_path: Caminho para salvar o arquivo
    """
    try:
        # Criar diretório se não existir
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar dados
        df.to_csv(output_path, index=False)
        logger.info(f"Dados salvos em: {output_path}")
    
    except Exception as e:
        logger.error(f"Erro ao salvar dados: {e}")
        raise


def main():
    """
    Função principal para executar todo o pipeline de processamento.
    """
    try:
        logger.info("Iniciando pipeline de processamento de dados...")
        
        # 1. Carregar dados
        customer_original, customer_social, customer_nps = load_raw_data()
        
        # 2. Limpar dados
        customer_original_clean = clean_customer_original(customer_original)
        customer_social_clean = clean_customer_social(customer_social)
        customer_nps_clean = clean_customer_nps(customer_nps)
        
        # 3. Fazer merge
        merged_data = merge_datasets(customer_original_clean, customer_social_clean, customer_nps_clean)
        
        # 4. Criar features
        data_with_features = create_features(merged_data)
        
        # 5. Salvar dados processados
        save_processed_data(data_with_features)
        
        logger.info("Pipeline de processamento concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro no pipeline de processamento: {e}")
        raise


if __name__ == "__main__":
    main()
