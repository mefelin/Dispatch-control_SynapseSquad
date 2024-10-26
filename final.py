import os
import cv2
import numpy as np
import torch
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from PIL import ImageFont, Image, ImageDraw, ImageTk


class FileAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Диспетчерское Приложение")
        self.root.geometry("400x400")
        self.root.resizable(False, False)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "rzd.png")
        image = Image.open(image_path)
        resized_image = image.resize((88, 40), Image.LANCZOS)
        self.logo = ImageTk.PhotoImage(resized_image)

        second_image_path = os.path.join(script_dir, "rzd2.png")
        second_image = Image.open(second_image_path)
        resized_second_image = second_image.resize((88, 49), Image.LANCZOS)
        self.logo2 = ImageTk.PhotoImage(resized_second_image)

        # Отображение логотипа в правом нижнем углу
        self.logo_label = tk.Label(root, image=self.logo)
        self.logo_label.place(x=5, y=355)

        # Отображение второго логотипа в левом нижнем углу
        self.logo_label2 = tk.Label(root, image=self.logo2)
        self.logo_label2.place(x=310, y=350)

        # UI элементы
        # Текст "Скоро будет реализовано" над кнопкой выбора камеры
        self.soon_label = tk.Label(root, text="Скоро будет реализовано", fg="red", font=("Arial", 8, "bold"))
        self.soon_label.place(x=220, y=20)

        # UI элементы
        self.select_file_button = tk.Button(root, text="Выбрать файл", command=self.select_file, width=20, height=2)
        self.select_file_button.place(x=30, y=40)

        self.non_work_button = tk.Button(root, text="Выбрать камеру", width=20, height=2, state=tk.DISABLED)
        self.non_work_button.place(x=220, y=40)

        self.analyze_button = tk.Button(root, text="Запустить анализ", command=self.start_analysis, width=20, height=2)
        self.analyze_button.place(x=130, y=130)

        self.export_button = tk.Button(root, text="Выгрузить отчет", command=self.save_report, width=20, height=2)
        self.export_button.place(x=130, y=220)

        # Полоска прогресса
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=250)
        self.progress_bar.place(x=80, y=310)

        self.file_path = None

        # Настройка модели
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
        # Указание относительного пути к весам модели
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, "model_final.pth")
        self.cfg.MODEL.WEIGHTS = weights_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        MetadataCatalog.get("Railway_Dataset").thing_classes = [
            "Special Equipment", "Person", "MPC", "RailwayTrack", "Stacking Crane", "USO Platform", "Wagon", "Platform",
            "Hopper"
        ]
        self.predictor = DefaultPredictor(self.cfg)
        self.font = ImageFont.truetype("arial.ttf", 24)

        # Переменные для анализа путей и рисков
        self.right_track_points = [(428, 175), (294, 567), (411, 298), (433, 240), (194, 805), (163, 898)]
        self.left_track_points = [(217, 414), (305, 300), (260, 355), (360, 180), (103, 552), (101, 592), (24, 747), (367, 214), (121, 606)]
        self.train_classes = [0, 2, 4, 5, 6, 7, 8, 9]
        self.special_equipment_class = 0
        self.railway_track_class = 3

        self.left_occupied_duration = 0
        self.right_occupied_duration = 0
        self.left_free_duration = 0
        self.right_free_duration = 0
        self.person_risk_count = 0
        self.equipment_risk_count = 0
        self.time_scaling_factor = 0.4  # Коэффициент масштабирования времени


    def select_file(self):
        self.file_path = filedialog.askopenfilename(title="Выберите видеофайл")
        if self.file_path:
            messagebox.showinfo("Файл выбран", f"Файл для анализа: {self.file_path}")

    def start_analysis(self):
        if not self.file_path:
            messagebox.showwarning("Ошибка", "Сначала выберите файл для анализа.")
            return
        threading.Thread(target=self.analyze_video).start()

    def check_line_intersection(self, mask, line_points):
        for (x, y) in line_points:
            if mask[y, x]:
                return True
        return False

    def analyze_video(self):
        cap = cv2.VideoCapture(self.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(1000 / fps)  # Интервал между кадрами в миллисекундах
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress_bar["maximum"] = total_frames
        current_frame = 0

        # Временные метки для отслеживания состояния и накопления времени
        left_last_state = right_last_state = False
        left_last_change = right_last_change = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            outputs = self.predictor(frame)
            instances = outputs["instances"].to("cpu")
            pred_classes = instances.pred_classes
            pred_masks = instances.pred_masks

            left_track_occupied = False
            right_track_occupied = False
            person_risk_detected = False
            equipment_on_track_risk = False

            # Проверка перекрытия линии правого пути
            for i, mask in enumerate(pred_masks):
                class_id = pred_classes[i].item()
                if class_id in self.train_classes:
                    for j in range(len(self.right_track_points) - 1):
                        line_points = np.linspace(self.right_track_points[j], self.right_track_points[j + 1],
                                                  num=20).astype(int)
                        if self.check_line_intersection(mask.numpy(), line_points):
                            right_track_occupied = True
                            break
                    if right_track_occupied:
                        break

            # Проверка перекрытия линии левого пути
            for i, mask in enumerate(pred_masks):
                class_id = pred_classes[i].item()
                if class_id in self.train_classes:
                    for j in range(len(self.left_track_points) - 1):
                        line_points = np.linspace(self.left_track_points[j], self.left_track_points[j + 1],
                                                  num=20).astype(int)
                        if self.check_line_intersection(mask.numpy(), line_points):
                            left_track_occupied = True
                            break
                    if left_track_occupied:
                        break

            # Текущее время
            current_time = time.time()
            elapsed_time = (current_time - left_last_change) * self.time_scaling_factor

            # Обновление времени для левого пути
            if left_track_occupied:
                self.left_occupied_duration += elapsed_time
            else:
                self.left_free_duration += elapsed_time

            # Обновление времени для правого пути
            if right_track_occupied:
                self.right_occupied_duration += elapsed_time
            else:
                self.right_free_duration += elapsed_time

            # Обновление временных меток и состояний
            left_last_change = current_time
            left_last_state = left_track_occupied
            right_last_state = right_track_occupied

            # Проверка риска для человека и спецтехники на путях
            for i, mask in enumerate(pred_masks):
                class_id = pred_classes[i].item()
                if class_id == 1:  # Человек
                    for j, other_mask in enumerate(pred_masks):
                        other_class_id = pred_classes[j].item()
                        if other_class_id in self.train_classes or other_class_id == self.railway_track_class:
                            if (mask & other_mask).sum().item() > 0:
                                person_risk_detected = True
                                self.person_risk_count += 1
                                break
                elif class_id == 0:  # Спец техника
                    for j, other_mask in enumerate(pred_masks):
                        if pred_classes[j].item() == self.railway_track_class:
                            if (mask & other_mask).sum().item() > 0:
                                equipment_on_track_risk = True
                                self.equipment_risk_count += 1
                                break

            # Установка текста для флагов
            left_track_status = f"Левый путь занят {int(self.left_occupied_duration)} с" if left_track_occupied else f"Левый путь свободен {int(self.left_free_duration)} с"
            right_track_status = f"Правый путь занят {int(self.right_occupied_duration)} с" if right_track_occupied else f"Правый путь свободен {int(self.right_free_duration)} с"
            person_risk_status = "Риск: человек на путях" if person_risk_detected else ""
            equipment_risk_status = "Риск: техника на путях" if equipment_on_track_risk else ""

            # Конвертация в PIL для добавления текста
            display_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(display_frame)
            draw.text((10, 30), left_track_status, font=self.font,
                      fill=(255, 0, 0) if left_track_occupied else (0, 255, 0))
            draw.text((10, 60), right_track_status, font=self.font,
                      fill=(255, 0, 0) if right_track_occupied else (0, 255, 0))
            if person_risk_detected:
                draw.text((frame.shape[1] - 300, 30), person_risk_status, font=self.font, fill=(255, 165, 0))
            if equipment_on_track_risk:
                draw.text((frame.shape[1] - 300, 60), equipment_risk_status, font=self.font, fill=(255, 0, 0))

            frame_with_text = cv2.cvtColor(np.array(display_frame), cv2.COLOR_RGB2BGR)
            visualizer = Visualizer(frame_with_text[:, :, ::-1], MetadataCatalog.get("Railway_Dataset"), scale=1.2)
            out = visualizer.draw_instance_predictions(instances)
            cv2.imshow("Анализ видео", out.get_image()[:, :, ::-1])

            current_frame += 1
            self.progress_bar["value"] = current_frame
            self.root.update_idletasks()

            # Задержка для приближения скорости воспроизведения к реальной
            if cv2.waitKey(frame_interval) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.save_report()

    def save_report(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if save_path:
            data = {
                "Путь": ["Левый", "Правый"],
                "Суммарное время занятости (сек)": [self.left_occupied_duration, self.right_occupied_duration],
                "Суммарное время свободности (сек)": [self.left_free_duration, self.right_free_duration],
                "Риск: человек на путях": [self.person_risk_count, self.person_risk_count],
                "Риск: техника на путях": [self.equipment_risk_count, self.equipment_risk_count]
            }
            df = pd.DataFrame(data)
            df.to_excel(save_path, index=False)
            messagebox.showinfo("Готово", "Отчёт успешно сохранён в Excel.")


if __name__ == "__main__":
    root = tk.Tk()
    app = FileAnalyzerApp(root)
    root.mainloop()
