"""
Módulo para treinamento de modelos de machine learning.

Este módulo contém funções para:
- Preparação de dados para modelagem
- Treinamento de diferentes algoritmos
- Avaliação de performance
- Otimização de hiperparâmetros
- Salvamento de modelos
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import pickle
import json

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Classe para treinamento e avaliação de modelos de machine learning.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa o ModelTrainer.
        
        Args:
            random_state: Seed para reprodutibilidade
        """
        self.random_state = random_state
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.feature_selector = None
        self.best_model = None
        self.best_score = 0
        
        # Configurar visualização
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = "churn", 
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepara dados para modelagem.
        
        Args:
            df: DataFrame com todos os dados
            target_column: Nome da coluna target
            test_size: Proporção do conjunto de teste
            
        Returns:
            Tuple com X_train, X_test, y_train, y_test
        """
        try:
            # Verificar se a coluna target existe
            if target_column not in df.columns:
                raise ValueError(f"Coluna target '{target_column}' não encontrada")
            
            # Separar features e target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Identificar colunas numéricas e categóricas
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            logger.info(f"Features numéricas: {len(numeric_columns)}")
            logger.info(f"Features categóricas: {len(categorical_columns)}")
            
            # Tratar variáveis categóricas
            X_encoded = X.copy()
            for col in categorical_columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                self.label_encoders[col] = le
            
            # Split estratificado
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=test_size, random_state=self.random_state, 
                stratify=y
            )
            
            # Normalizar features numéricas
            self.scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[numeric_columns] = self.scaler.fit_transform(X_train[numeric_columns])
            X_test_scaled[numeric_columns] = self.scaler.transform(X_test[numeric_columns])
            
            logger.info(f"Preparação concluída:")
            logger.info(f"- X_train: {X_train_scaled.shape}")
            logger.info(f"- X_test: {X_test_scaled.shape}")
            logger.info(f"- y_train: {y_train.shape}")
            logger.info(f"- y_test: {y_test.shape}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Erro na preparação dos dados: {e}")
            raise
    
    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
                       method: str = "kbest", k: int = 20) -> pd.DataFrame:
        """
        Seleciona as melhores features para modelagem.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            method: Método de seleção ('kbest', 'rfe')
            k: Número de features a selecionar
            
        Returns:
            DataFrame com features selecionadas
        """
        try:
            if method == "kbest":
                self.feature_selector = SelectKBest(score_func=f_classif, k=k)
                X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
                
                # Obter nomes das features selecionadas
                selected_features = X_train.columns[self.feature_selector.get_support()]
                X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
                
            elif method == "rfe":
                # Usar Random Forest para RFE
                rf = RandomForestClassifier(random_state=self.random_state)
                self.feature_selector = RFE(estimator=rf, n_features_to_select=k)
                X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
                
                # Obter nomes das features selecionadas
                selected_features = X_train.columns[self.feature_selector.get_support()]
                X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
            
            logger.info(f"Seleção de features concluída. Features selecionadas: {len(selected_features)}")
            
            return X_train_selected
            
        except Exception as e:
            logger.error(f"Erro na seleção de features: {e}")
            raise
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Treina múltiplos modelos.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            
        Returns:
            Dicionário com modelos treinados
        """
        try:
            # Definir modelos
            models_config = {
                'Logistic Regression': {
                    'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                    'params': {
                        'C': [0.1, 1, 10],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear']
                    }
                },
                'Random Forest': {
                    'model': RandomForestClassifier(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5]
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingClassifier(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5]
                    }
                },
                'XGBoost': {
                    'model': xgb.XGBClassifier(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 5],
                        'learning_rate': [0.01, 0.1]
                    }
                },
                'LightGBM': {
                    'model': lgb.LGBMClassifier(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 5],
                        'learning_rate': [0.01, 0.1]
                    }
                }
            }
            
            results = {}
            
            for name, config in models_config.items():
                logger.info(f"Treinando {name}...")
                
                # Grid Search para otimização de hiperparâmetros
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                results[name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_
                }
                
                logger.info(f"{name} - Melhor score: {grid_search.best_score_:.4f}")
                
                # Atualizar melhor modelo
                if grid_search.best_score_ > self.best_score:
                    self.best_score = grid_search.best_score_
                    self.best_model = grid_search.best_estimator_
            
            self.models = results
            return results
            
        except Exception as e:
            logger.error(f"Erro no treinamento dos modelos: {e}")
            raise
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """
        Avalia todos os modelos treinados.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            
        Returns:
            Dicionário com métricas de avaliação
        """
        try:
            evaluation_results = {}
            
            for name, model_info in self.models.items():
                model = model_info['model']
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calcular métricas
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                evaluation_results[name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'auc': auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Erro na avaliação dos modelos: {e}")
            raise
    
    def plot_results(self, evaluation_results: Dict[str, Dict], y_test: pd.Series, 
                    save_path: str = "../reports/figures/"):
        """
        Gera visualizações dos resultados.
        
        Args:
            evaluation_results: Resultados da avaliação
            y_test: Target de teste
            save_path: Caminho para salvar as figuras
        """
        try:
            # Criar diretório se não existir
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            # 1. Comparação de métricas
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            metrics = ['accuracy', 'f1_score', 'auc']
            metric_names = ['Accuracy', 'F1 Score', 'AUC']
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                values = [results[metric] for results in evaluation_results.values()]
                model_names = list(evaluation_results.keys())
                
                axes[i].bar(model_names, values)
                axes[i].set_title(f'{name} por Modelo')
                axes[i].set_ylabel(name)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Adicionar valores nas barras
                for j, v in enumerate(values):
                    axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}model_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # 2. Curvas ROC
            plt.figure(figsize=(10, 8))
            for name, results in evaluation_results.items():
                fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
                auc = results['auc']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Curvas ROC')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{save_path}roc_curves.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # 3. Feature importance (para o melhor modelo)
            if self.best_model and hasattr(self.best_model, 'feature_importances_'):
                feature_importance = self.best_model.feature_importances_
                feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature_{i}' for i in range(len(feature_importance))]
                
                # Ordenar por importância
                indices = np.argsort(feature_importance)[::-1]
                
                plt.figure(figsize=(12, 8))
                plt.title('Feature Importance - Melhor Modelo')
                plt.bar(range(len(feature_importance)), feature_importance[indices])
                plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.tight_layout()
                plt.savefig(f"{save_path}feature_importance.png", dpi=300, bbox_inches='tight')
                plt.show()
            
            logger.info(f"Visualizações salvas em: {save_path}")
            
        except Exception as e:
            logger.error(f"Erro na geração de visualizações: {e}")
            raise
    
    def save_model(self, model_name: str = "best_model", save_path: str = "../models/"):
        """
        Salva o melhor modelo e preprocessadores.
        
        Args:
            model_name: Nome do arquivo do modelo
            save_path: Caminho para salvar o modelo
        """
        try:
            # Criar diretório se não existir
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            # Salvar modelo
            model_file = f"{save_path}/{model_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(self.best_model, f)
            
            # Salvar preprocessadores
            preprocessors = {
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_selector': self.feature_selector
            }
            
            preprocessors_file = f"{save_path}/preprocessors.pkl"
            with open(preprocessors_file, 'wb') as f:
                pickle.dump(preprocessors, f)
            
            # Salvar metadados
            metadata = {
                'best_score': self.best_score,
                'model_type': type(self.best_model).__name__,
                'feature_names': list(self.best_model.feature_names_in_) if hasattr(self.best_model, 'feature_names_in_') else None
            }
            
            metadata_file = f"{save_path}/metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Modelo salvo em: {model_file}")
            logger.info(f"Preprocessadores salvos em: {preprocessors_file}")
            logger.info(f"Metadados salvos em: {metadata_file}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            raise


def main():
    """
    Função principal para executar o pipeline completo de modelagem.
    """
    try:
        logger.info("Iniciando pipeline de modelagem...")
        
        # Carregar dados processados
        df = pd.read_csv("../data/02_processed/merged_customer_data.csv")
        logger.info(f"Dados carregados: {df.shape}")
        
        # Inicializar trainer
        trainer = ModelTrainer()
        
        # Preparar dados
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)
        
        # Selecionar features
        X_train_selected = trainer.select_features(X_train, y_train)
        X_test_selected = X_test[X_train_selected.columns]
        
        # Treinar modelos
        results = trainer.train_models(X_train_selected, y_train)
        
        # Avaliar modelos
        evaluation_results = trainer.evaluate_models(X_test_selected, y_test)
        
        # Gerar visualizações
        trainer.plot_results(evaluation_results, y_test)
        
        # Salvar modelo
        trainer.save_model()
        
        logger.info("Pipeline de modelagem concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro no pipeline de modelagem: {e}")
        raise


if __name__ == "__main__":
    main()
