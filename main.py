import cv2
import numpy as np
import os

# Pasta de saída
def criar_pasta_saida(pasta="resultados"):
    os.makedirs(pasta, exist_ok=True)
    return pasta

# Leitura da imagem
def carregar_imagem(caminho_imagem):
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem: {caminho_imagem}")

    # Exibir a imagem original
    cv2.imshow("Imagem Original", imagem)
    cv2.waitKey(0)  # Espera até uma tecla ser pressionada
    cv2.destroyAllWindows()  # Fecha a janela

    return imagem


# 2 - Pré-processamento - Escala de cinza + equalização de histograma
def pre_processamento(imagem, pasta_saida):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    equalizada = cv2.equalizeHist(cinza)
    cv2.imwrite(f"{pasta_saida}/imagem_equalizada_pre_processamento.jpg", equalizada)
    return equalizada

# 3 - Modificação de Cores - Aumento de saturação no espaço HSV
def modificar_saturacao(imagem, aumento_percentual, pasta_saida):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s + (s * aumento_percentual / 100.0), 0, 255).astype(np.uint8)
    hsv_mod = cv2.merge([h, s, v])
    resultado = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f"{pasta_saida}/imagem_saturada.jpg", resultado)
    return resultado

# 4 - Ajuste de Contraste e Brilho - Contraste e brilho
def ajustar_contraste_brilho(imagem, alpha, beta, pasta_saida):
    ajustada = cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)
    cv2.imwrite(f"{pasta_saida}/imagem_ajustada_brilho_contraste.jpg", ajustada)
    return ajustada

# 5 - Redimensionamento e Interpolação - Redimensionamento
def redimensionar(imagem, escala, metodo, nome_arquivo, pasta_saida):
    altura, largura = imagem.shape[:2]
    nova_imagem = cv2.resize(imagem, (int(largura * escala), int(altura * escala)), interpolation=metodo)
    cv2.imwrite(f"{pasta_saida}/{nome_arquivo}", nova_imagem)
    return nova_imagem

    # 6 - Transformações Geométricas

# Rotação
def rotacionar(imagem, angulo, pasta_saida):
    altura, largura = imagem.shape[:2]
    centro = (largura // 2, altura // 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    rotacionada = cv2.warpAffine(imagem, matriz, (largura, altura))
    cv2.imwrite(f"{pasta_saida}/imagem_rotacionada_45.jpg", rotacionada)
    return rotacionada

# Espelhamento horizontal
def espelhar(imagem, pasta_saida):
    espelhada = cv2.flip(imagem, 1)
    cv2.imwrite(f"{pasta_saida}/imagem_espelhada.jpg", espelhada)
    return espelhada

# Recorte central
def recorte_central(imagem, largura_crop, altura_crop, pasta_saida):
    altura, largura = imagem.shape[:2]
    x_inicio = max(0, largura // 2 - largura_crop // 2)
    y_inicio = max(0, altura // 2 - altura_crop // 2)
    recorte = imagem[y_inicio:y_inicio+altura_crop, x_inicio:x_inicio+largura_crop]
    cv2.imwrite(f"{pasta_saida}/imagem_recortada_300x300.jpg", recorte)
    return recorte

#7 - Binarização com Otsu
def binarizar_otsu(imagem_cinza, pasta_saida):
    _, binarizada = cv2.threshold(imagem_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{pasta_saida}/imagem_otsu.jpg", binarizada)
    return binarizada

# Função principal que executa todo o fluxo
def processar_imagem(caminho_imagem, pasta_saida="resultados"):
    criar_pasta_saida(pasta_saida)
    imagem = carregar_imagem(caminho_imagem)
    cv2.imwrite(f"{pasta_saida}/imagem_original.jpg", imagem)

    # Pré-processamento
    imagem_cinza_equalizada = pre_processamento(imagem, pasta_saida)

    # Modificação de saturação
    imagem_hsv = modificar_saturacao(imagem, aumento_percentual=30, pasta_saida=pasta_saida)

    # Ajuste de contraste e brilho
    imagem_ajustada = ajustar_contraste_brilho(imagem, alpha=1.2, beta=50, pasta_saida=pasta_saida)

    # Redimensionamentos
    redimensionar(imagem, escala=0.5, metodo=cv2.INTER_CUBIC, nome_arquivo="imagem_redimensionada_50.jpg", pasta_saida=pasta_saida)
    redimensionar(imagem, escala=2.0, metodo=cv2.INTER_LINEAR, nome_arquivo="imagem_redimensionada_200.jpg", pasta_saida=pasta_saida)

    # Transformações geométricas
    rotacionar(imagem, angulo=45, pasta_saida=pasta_saida)
    espelhar(imagem, pasta_saida=pasta_saida)
    recorte_central(imagem, largura_crop=300, altura_crop=300, pasta_saida=pasta_saida)

    # Binarização Otsu
    binarizar_otsu(imagem_cinza_equalizada, pasta_saida=pasta_saida)

    print("Processamento concluído. Resultados salvos na pasta:", pasta_saida)

# Chamada da função principal (Exemplo de uso)
if __name__ == "__main__":
    processar_imagem("imagem_exame.jpg")
