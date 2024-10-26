import os
import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def train_model():
    # Пути к файлам разметки и папке с изображениями
    images_path = "C:/Users/Litvein/Desktop/hakaton/rzhd2_frames_renamed"
    train_json = "C:/Users/Litvein/Desktop/hakaton/train/train_coco.json"
    test_json = "C:/Users/Litvein/Desktop/hakaton/test/test_coco.json"

    # Регистрация тренировочного и тестового датасетов
    register_coco_instances("Railway_Train_Dataset", {}, train_json, images_path)
    register_coco_instances("Railway_Test_Dataset", {}, test_json, images_path)

    # Настройка конфигурации Mask R-CNN
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("Railway_Train_Dataset",)
    cfg.DATASETS.TEST = ("Railway_Test_Dataset",)
    cfg.DATALOADER.NUM_WORKERS = 0  # Для Windows ставим 0

    # Параметры тренировки
    cfg.SOLVER.IMS_PER_BATCH = 2  # Размер батча
    cfg.SOLVER.BASE_LR = 0.00025  # Базовая скорость обучения
    cfg.SOLVER.MAX_ITER = 10000  # Увеличенное количество итераций
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Размер батча для обучения ROI
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # Количество классов

    # Путь для сохранения обученной модели
    output_dir = "C:/Users/Litvein/Desktop/hakaton/output_train"
    os.makedirs(output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = output_dir

    # Запуск тренировки
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Оценка модели на тестовом наборе
    evaluator = COCOEvaluator("Railway_Test_Dataset", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "Railway_Test_Dataset")
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == "__main__":
    train_model()
