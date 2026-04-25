# detect_webcam.py
#
# Aula 6 - Projeto Prático: Detecção de Objetos em Tempo Real com YOLO
#
# Este script usa a biblioteca 'ultralytics' para carregar um modelo YOLOv8 pré-treinado
# e executar a detecção de objetos na webcam do seu computador.
#
# Instruções:
# 1. Certifique-se de que seu ambiente (venv) está ativo.
# 2. Certifique-se de ter instalado: pip install ultralytics opencv-python
# 3. Salve este arquivo como 'detect_webcam.py' na pasta do seu projeto.
# 4. Execute no terminal: python detect_webcam.py
# 5. Pressione 'q' na janela da webcam para sair.

from ultralytics import YOLO

def main():
    # ----------------------------------------------------
    # Bloco 1: Carregar o Modelo (Transfer Learning)
    # ----------------------------------------------------
    
    # Carregamos o modelo 'yolov8n.pt'
    # 'n' significa 'nano', a versão mais leve e rápida do YOLOv8.
    # O modelo já foi treinado no dataset COCO (80 classes).
    # O arquivo .pt será baixado automaticamente na primeira execução.
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"Erro ao carregar o modelo YOLO: {e}")
        print("Verifique sua conexão com a internet para baixar o modelo.")
        return

    print("--- Modelo YOLOv8 (nano) carregado com sucesso ---")
    print("Iniciando detecção na webcam...")
    print("Pressione 'q' na janela da webcam para encerrar.")

    # ----------------------------------------------------
    # Bloco 2: Executar a Detecção
    # ----------------------------------------------------
    
    # Chamamos a função .predict() do modelo
    try:
        # 'source=0' é o ID padrão para a primeira webcam conectada.
        # 'show=True' instrui o YOLO a abrir uma janela do OpenCV e exibir o vídeo ao vivo.
        # 'conf=0.5' (Confiança) só vai desenhar caixas para objetos com mais de 50% de certeza.
        results = model.predict(source='0', show=True, conf=0.5)

        # O código ficará "preso" nesta linha até você fechar a janela (pressionando 'q').
        # 'results' na verdade é um gerador que processa frame a frame.
    
    except Exception as e:
        print(f"Erro ao tentar acessar a webcam (source=0): {e}")
        print("Verifique se a webcam está conectada e não está sendo usada por outro programa (como Zoom ou Teams).")

    print("--- Detecção finalizada ---")

if __name__ == "__main__":
    main()