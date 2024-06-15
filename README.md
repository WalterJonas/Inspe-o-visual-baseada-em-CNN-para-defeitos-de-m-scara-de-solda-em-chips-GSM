# Inspeção visual baseada em CNN para defeitos de máscara de solda em chips GSM

## Descrição
Este projeto implementa uma rede neural convolucional (CNN) para a detecção de falhas na máscara de solda (parte dourada) de chips GSM, visando a inspeção visual em uma linha de produção. O modelo foi treinado com imagens originais e validado utilizando imagens com ruídos e borramentos para verificar a robustez e a precisão do modelo em condições adversas. Para avaliação comparativa, também foram utilizados modelos de SVM e Random Forest. A abordagem avalia o desempenho dos modelos em diferentes condições de ruído e distorção, proporcionando uma análise abrangente da eficácia dos métodos aplicados na detecção de falhas na máscara de solda. De inicio fizemos a segmentação da lateral dos chips devido o foco principal da inspeção ser na parte dourada, que fica na lateral dos chips.

Tecnologias Utilizadas
- Python
- Tensorflow e Keras
- OpenCV

Técnicas Utilizadas
- Rede Neural Convolucional (CNN)
- Máquina de Vetores de Suporte (SVM)
- Floresta Aleatória (Random Forest)

Simulação de condições adversas
- Ruído Gaussiano
- Ruído Sal e Pimenta
- Borramento Gaussiano (Para simular uma câmera desfocada)


Borramento Gaussiano 👇

5x5
![5x5 (1)](https://github.com/WalterJonas/Inspecao-visual-baseada-em-CNN-para-deteccao-de-defeitos-de-mascara-de-solda-em-chips-GSM/assets/74218624/cb784d0f-9381-462a-b4f7-303e0c23833f)
7x7
![7x7](https://github.com/WalterJonas/Inspecao-visual-baseada-em-CNN-para-deteccao-de-defeitos-de-mascara-de-solda-em-chips-GSM/assets/74218624/4ef15f6a-d60b-4798-aab7-fb34f7ed9110)
9x9
![9x9](https://github.com/WalterJonas/Inspecao-visual-baseada-em-CNN-para-deteccao-de-defeitos-de-mascara-de-solda-em-chips-GSM/assets/74218624/7949fb3f-fc83-4baa-acc8-222ce9b2a927)
11x11
![11x11](https://github.com/WalterJonas/Inspecao-visual-baseada-em-CNN-para-deteccao-de-defeitos-de-mascara-de-solda-em-chips-GSM/assets/74218624/a88daf46-f85e-41a7-b448-182e869e2871)


Ruído Gaussiano 👇

0.005
![Gaussian-0 005](https://github.com/WalterJonas/Inspecao-visual-baseada-em-CNN-para-deteccao-de-defeitos-de-mascara-de-solda-em-chips-GSM/assets/74218624/079b3f67-7fc4-4df8-ad1e-c822509613dd)
0.01
![Gaussian-0 01](https://github.com/WalterJonas/Inspecao-visual-baseada-em-CNN-para-deteccao-de-defeitos-de-mascara-de-solda-em-chips-GSM/assets/74218624/737f0125-8790-4506-a1f7-2dfad6347723)
0.02
![Gaussian-0 02](https://github.com/WalterJonas/Inspecao-visual-baseada-em-CNN-para-deteccao-de-defeitos-de-mascara-de-solda-em-chips-GSM/assets/74218624/89b824a5-7153-4707-82bc-6df765ba418d)


Ruído Sal e Pimenta 👇

0.005
![S P-0 005](https://github.com/WalterJonas/Inspecao-visual-baseada-em-CNN-para-deteccao-de-defeitos-de-mascara-de-solda-em-chips-GSM/assets/74218624/9e36e729-8b24-4ded-98cc-63ad691f4e62)
0.01
![S P-0 01](https://github.com/WalterJonas/Inspecao-visual-baseada-em-CNN-para-deteccao-de-defeitos-de-mascara-de-solda-em-chips-GSM/assets/74218624/7de426e2-32ad-454c-83e7-20d6b916ce3f)
0.02 
![S P-0 02](https://github.com/WalterJonas/Inspecao-visual-baseada-em-CNN-para-deteccao-de-defeitos-de-mascara-de-solda-em-chips-GSM/assets/74218624/df2c69f8-6e6e-484b-aafc-6aa1e3aceb1d)




