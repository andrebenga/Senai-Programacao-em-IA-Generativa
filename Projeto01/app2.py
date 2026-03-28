# LACUNA 1: Importe 'streamlit' com o apelido 'st'
import streamlit as st
# LACUNA 2: Importe 'joblib' para carregar o modelo
import joblib
# LACUNA 3: Importe 'pandas' como 'pd' (para formatar os dados de entrada)
import pandas as pd
import os # (Já vem pronto, para verificar se o arquivo existe)

# Nome do arquivo do modelo (deve ser o mesmo salvo pelo train.py)
NOME_ARQUIVO_MODELO = 'modelo_desempenho.pkl'

@st.cache_resource # Cache para carregar o modelo apenas uma vez
def carregar_modelo(caminho_modelo):
    """
    Carrega o modelo do arquivo .pkl.
    """
    if not os.path.exists(caminho_modelo):
        return None, None
    try:
        modelo = joblib.load(caminho_modelo)
        classes_modelo = modelo.classes_
        return modelo, classes_modelo
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None, None    

def main():
    """
    Função principal que executa o App Streamlit.
    """
    
    # --- Carregamento do Modelo ---
    modelo, classes_modelo = carregar_modelo(NOME_ARQUIVO_MODELO)
    
    # --- Título e Subtítulo ---
    # LACUNA 4: Dê um título ao seu App usando 'st.title()'
    st.title('Dados Aluno')
    st.subheader('Estudo de Caso da Imersão em IA (Aulas 1-3)')

    # Se o modelo não existir, exibe um aviso (pronto)
    if modelo is None:
        st.error(f"Arquivo do modelo ('{NOME_ARQUIVO_MODELO}') não encontrado.")
        st.warning("Execute o script 'python train.py' no terminal para treinar e criar o modelo.")
        st.stop() # Para a execução do app

    

    # --- Interface do Usuário (Entradas) na Sidebar ---
    # LACUNA 5: Crie um cabeçalho na 'sidebar' (barra lateral)
    # Dica: st.sidebar.header('...')
    st.sidebar.header('Insira os dados do Aluno:')

    # LACUNA 6: Crie um 'st.sidebar.slider' para 'Horas de Estudo'
    # (Nome: 'Média de Horas de Estudo/semana', min: 0, max: 20, padrão: 5)
    horas_estudo = st.sidebar.slider('Média de Horas de Estudo/semana', 0, 20, 5)

    # LACUNA 7: Crie um 'st.sidebar.number_input' para 'Faltas'
    # (Nome: 'Número Total de Faltas', min: 0, max: 50, padrão: 3)
    faltas = st.sidebar.number_input('Número Total de Faltas', min_value=0, max_value=50, value=3)

    # (Este vem pronto)
    nota_p1 = st.sidebar.number_input('Nota da Primeira Prova (0-10)', min_value=0.0, max_value=10.0, value=5.0, step=0.5)    

    # --- Botão de Previsão ---
    # LACUNA 8: Crie um 'st.sidebar.button' com o texto 'Prever Situação'
    # Dica: if st.sidebar.button('...'):
    if st.sidebar.button('Prever Situação'):

        # 1. Formatando os dados de entrada (pronto)
        # O Scikit-learn espera um DataFrame do Pandas
        dados_entrada = pd.DataFrame(
            [[horas_estudo, faltas, nota_p1]],
            columns=['horas_estudo', 'faltas', 'nota_p1']
        )
        
        st.write("Dados de Entrada:")
        st.dataframe(dados_entrada)

        # 2. Fazendo a Previsão de Probabilidade
        # LACUNA 9: Use 'modelo.predict_proba()' para obter as probabilidades
        # Dica: passe os 'dados_entrada'
        probabilidades = modelo.predict_proba(dados_entrada)

        # 3. Fazendo a Previsão Final (a classe)
        # LACUNA 10: Use 'modelo.predict()' para obter a classe final
        # Dica: passe os 'dados_entrada' e pegue o primeiro item [0]
        previsao = modelo.predict(dados_entrada)[0]

        # 4. Exibindo os Resultados
        st.header(f'Resultado da Previsão: {previsao}')

        # LACUNA 11: Use 'st.success()' se a 'previsao' for 'Aprovado'
        # e 'st.error()' caso contrário
        if previsao == 'Aprovado':
            st.success('Este aluno tem alta probabilidade de ser APROVADO.')
        else:
            st.error('Este aluno está em ZONA DE RISCO (Reprovado).')

        # (Bloco pronto para exibir métricas)
        st.write("--- Análise de Confiança da IA ---")
        prob_aprovado = probabilidades[0][list(classes_modelo).index('Aprovado')]
        prob_reprovado = probabilidades[0][list(classes_modelo).index('Reprovado')]
        col1, col2 = st.columns(2)
        col1.metric("Confiança em 'Aprovado'", f"{prob_aprovado*100:.2f}%")
        col2.metric("Confiança em 'Reprovado'", f"{prob_reprovado*100:.2f}%")




    st.markdown("---")
    st.write("Este App foi construído no curso de Programação em IA Generativa.")

# (Bloco pronto)
if __name__ == "__main__":
    main()