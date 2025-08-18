# Guia de Uso da Função `analysis_plots`

## Visão Geral

A função `analysis_plots` foi ajustada para atender aos requisitos de comportamento diferenciado da legenda em dois cenários:

1. **Quando exibe subplots em grade** (`plt.show()`) → Legenda automática do Seaborn
2. **Quando salva subplots individualmente** → Legenda forçada no canto superior direito externo

## Comportamento da Legenda

### Cenário 1: Exibição em Grade (`save_files=False`)
- **Legenda automática** do Seaborn
- Posicionamento natural e otimizado
- Sem interferência manual no posicionamento
- Ideal para análise exploratória interativa

### Cenário 2: Salvamento Individual (`save_files=True`)
- **Legenda forçada** no canto superior direito externo
- `bbox_to_anchor=(1.05, 1), loc='upper left'`
- Garante que a legenda não sobreponha o gráfico
- Melhor apresentação para relatórios e apresentações

## Exemplos de Uso

### 1. Histogramas Simples (sem legenda)
```python
from src.utils import analysis_plots

analysis_plots(
    data=df, 
    features=['idade', 'renda', 'tempo_contrato'], 
    histplot=True,
    kde=True,
    save_files=False  # Apenas exibe em grade
)
```

### 2. Histogramas com Legenda - Legenda Automática
```python
analysis_plots(
    data=df, 
    features=['idade', 'renda'], 
    histplot=True,
    kde=True,
    hue='churn',  # Legenda automática do Seaborn
    save_files=False
)
```

### 3. Gráficos de Barras com Legenda - Legenda Automática
```python
analysis_plots(
    data=df, 
    features=['genero', 'estado_civil'], 
    barplot=True,
    hue='churn',  # Legenda automática do Seaborn
    save_files=False
)
```

### 4. Salvar Gráficos Individuais com Legenda Forçada
```python
analysis_plots(
    data=df, 
    features=['idade', 'renda', 'genero'], 
    histplot=True,
    kde=True,
    hue='churn',  # Legenda forçada no canto superior direito externo
    save_files=True,
    save_path='./reports/figures/'
)
```

### 5. Gráficos de Barras Salvos com Legenda Forçada
```python
analysis_plots(
    data=df, 
    features=['genero', 'estado_civil', 'tipo_contrato'], 
    barplot=True,
    hue='churn',  # Legenda forçada no canto superior direito externo
    save_files=True,
    save_path='./reports/figures/'
)
```

### 6. Boxplots para Análise de Outliers
```python
analysis_plots(
    data=df, 
    features=['idade', 'renda', 'tempo_contrato'], 
    outliers=True,
    save_files=True,
    save_path='./reports/figures/'
)
```

### 7. Boxplots por Categoria
```python
analysis_plots(
    data=df, 
    features=['renda', 'tempo_contrato'], 
    boxplot=True,
    boxplot_x='churn',  # Agrupa por churn
    save_files=True,
    save_path='./reports/figures/'
)
```

## Parâmetros Principais

| Parâmetro | Tipo | Descrição | Padrão |
|-----------|------|-----------|--------|
| `data` | DataFrame | DataFrame com os dados | - |
| `features` | List[str] | Lista de colunas para plotar | - |
| `histplot` | bool | True para histogramas | True |
| `barplot` | bool | True para gráficos de barras | False |
| `outliers` | bool | True para boxplots de outliers | False |
| `boxplot` | bool | True para boxplots por categoria | False |
| `kde` | bool | True para adicionar linha de densidade | False |
| `hue` | str | Coluna para colorir/agrupar (gera legenda) | None |
| `save_files` | bool | True para salvar arquivos individuais | False |
| `save_path` | str | Caminho para salvar os arquivos | '.' |

## Comportamento da Legenda por Cenário

### Cenário: `save_files=False`
```python
# Legenda automática do Seaborn
ax.legend(title=hue)
```

### Cenário: `save_files=True`
```python
# Legenda forçada no canto superior direito externo
ax_single.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
```

## Estrutura de Arquivos Gerados

Quando `save_files=True`, os arquivos são salvos com a seguinte nomenclatura:
- `{feature}_{plot_type}.png`

Exemplos:
- `idade_histplot.png`
- `genero_barplot.png`
- `renda_outliers_boxplot.png`
- `tempo_contrato_distribution_boxplot.png`

## Notebook de Exemplo

Execute o notebook `exemplo_usage_analysis_plots.ipynb` para ver exemplos práticos de todos os cenários de uso.

## Notas Importantes

1. **Título Superior**: Apenas o título superior é mantido, removendo labels dos eixos X e Y
2. **Layout**: `tight_layout()` é aplicado automaticamente
3. **Qualidade**: Arquivos salvos têm DPI=300 para alta qualidade
4. **Memória**: Figuras individuais são fechadas automaticamente após salvar
5. **Diretórios**: O diretório de destino é criado automaticamente se não existir
