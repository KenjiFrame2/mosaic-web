# Mosaic Generator

Программа для создания мозаичных изображений из набора тайлов.  
Поддерживает выбор исходного изображения и папки с тайлами, настройку размера сетки, шага, ограничений на использование тайлов, поворот, предпросмотр результата и сохранение готовой мозаики.

---

## Установка и запуск

### 1. Клонировать проект
```bash
git clone https://git.miem.hse.ru/stepa/mosaic-generator.git
cd mosaic-generator
```

### 2. Установить зависимости
Рекомендуется использовать виртуальное окружение:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 3. Запуск GUI
```bash
python main_gui.py
```

---

## Зависимости
Основные библиотеки:
- `numpy==2.3.3`
- `pillow==11.3.0`

Tkinter используется для графического интерфейса и входит в стандартную библиотеку Python.  
На Linux может понадобиться доустановить:
```bash
sudo apt-get install python3-tk
```

---

## Запуск в Docker
(опционально, если хочешь запускать через контейнер)  
```bash
docker build -t mosaic-app .
docker run -it --rm     -e DISPLAY=$DISPLAY     -v /tmp/.X11-unix:/tmp/.X11-unix     mosaic-app
```

На Windows/Mac для работы GUI понадобится X11 или VNC.

---

## Структура проекта
```
.
├── main.py          # Логика мозаики
├── main_gui.py      # Графический интерфейс (Tkinter)
├── requirements.txt # Зависимости
├── .gitignore
└── README.md
```

---

## Возможности
- Выбор исходного изображения и папки с тайлами  
- Настройка сетки: размер тайла, шаг  
- Поворот тайлов (0°, 90°, 180°, 270°)  
- Ограничение частоты использования тайлов  
- Простая цветокоррекция  
- Предпросмотр мозаики  
- Сохранение результата в PNG/JPEG/BMP/WebP  

---
