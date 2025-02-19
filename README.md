## Creating a depth map using the MiDaS model.    
## Создание карты глубины с использованием модели MiDaS. 🗺

![OpenCV](https://img.shields.io/badge/OpenCV-%23FF6F00.svg?style=for-the-badge&logo=opencv&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Pillow](https://img.shields.io/badge/Pillow-2088FF?style=for-the-badge&logo=pillow&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-003B57?style=for-the-badge&logo=matplotlib&logoColor=white) ![Tkinter](https://img.shields.io/badge/Tkinter-0088FF?style=for-the-badge&logo=python&logoColor=white) 

Проект представляет собой систему для вычисления карты глубины из монохромных изображений с помощью модели MiDaS (Monocular Depth Estimation). Используя веб-камеру, приложение захватывает изображение, передает его в модель для предсказания глубины, а затем отображает результат в графическом интерфейсе.

### 🖥 Основные области интерфейса:
* Верхняя центральная часть (label1) — Видео с камеры: Показывает потоковое видео с веб-камеры в реальном времени.
* Левая часть (label2) — Карта глубины: Отображает вычисленную карту глубины для захваченного изображения.
* Правая часть (label3) — Исходное изображение: Показывает исходное изображение, захваченное с камеры.
* Кнопка Snapshot: Позволяет сделать снимок с камеры, сохранить его и создать карту глубины для этого снимка.

### 🚀 Как запустить:   
1. Установите необходимые зависимости: _pip install torch opencv-python matplotlib pillow_
2. Запустите _main.py_ для старта интерфейса.
3. Нажмите кнопку "Snapshot" для создания карты глубины.

### ❕ Примечание: 
Модель MiDaS работает только с одним изображением и генерирует предсказанную карту глубины, которая является результатом оценки, а не фактической глубиной сцены. 

<img src="example.png" width="500"/>


