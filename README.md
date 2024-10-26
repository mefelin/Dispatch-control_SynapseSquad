# Диспетчерское Приложение для Мониторинга Железной Дороги

Данное приложение является диспетчерским интерфейсом для автоматического мониторинга железнодорожной станции. Оно позволяет отслеживать занятость путей, выявлять риски нахождения людей или спецтехники на путях, генерировать отчёты и предупреждать диспетчеров о потенциально опасных ситуациях. Приложение разработано с использованием Detectron2 для анализа видео в реальном времени.

---

## Возможности

- **Мониторинг занятости путей**: определяет, заняты ли пути поездами или другими объектами.
- **Обнаружение рисков**: определяет, находится ли человек или техника на путях, и предупреждает об опасности.
- **Генерация отчетов**: создаёт отчёт в формате Excel с подробностями о времени занятости путей и событиях риска.

## Структура проекта

- **`final.py`**: основной файл приложения, содержащий логику и интерфейс.
- **`model_final.pth`**: файл с предобученной моделью, используемой в приложении.
- **`rzd.png`**: изображение первого логотипа.
- **`rzd2.png`**: изображение второго логотипа.

## Требования

Для запуска приложения необходимы следующие зависимости:

- Python 3.7+
- `torch`
- `opencv-python`
- `pandas`
- `detectron2`
- `Pillow`
- `tkinter`

> **Примечание**: если планируется использование GPU, убедитесь, что на вашей машине установлена поддержка CUDA.

## Установка

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/yourusername/railway-monitoring-app.git
cd railway-monitoring-app
