"""PyQt GUI for the licence plate recognition system."""
from __future__ import annotations

import sys

from app.config import load_config, save_config
from app.database import PlateDatabase
from gui.common import (
    QApplication,
    QDesktopWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QThread,
    pyqtSignal,
    QPixmap,
    QImage,
)
from process_video_realtime import process_video_realtime


class VideoThread(QThread):
    frame_signal = pyqtSignal(QImage)
    text_signal = pyqtSignal(str)

    def __init__(self, video_path, settings, database):
        super().__init__()
        self.video_path = video_path
        self.settings = settings
        self.database = database

    def run(self):
        process_video_realtime(
            self.video_path,
            frame_callback=self.frame_signal.emit,
            text_callback=self.text_signal.emit,
            settings=self.settings,
            database=self.database,
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_config()
        self.database = PlateDatabase(self.settings.database)
        self.recognized_plates = set()
        self.video_threads = []

        self.initUI()
        self.load_existing_records()
        self.start_processing()

    def initUI(self):
        self.setWindowTitle("Распознавание автомобильных номеров")
        screen = QDesktopWidget().screenGeometry()
        window_width = int(screen.width() * 0.8)
        window_height = int(screen.height() * 0.8)
        self.setGeometry(
            (screen.width() - window_width) // 2,
            (screen.height() - window_height) // 2,
            window_width,
            window_height,
        )

        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        video_and_list_layout = QHBoxLayout()
        main_layout.addLayout(video_and_list_layout)

        self.video_label = QLabel(self)
        video_width = int(window_width * 0.7)
        video_height = int(window_height * 0.7)
        self.video_label.setFixedSize(video_width, video_height)
        self.video_label.setScaledContents(True)
        video_and_list_layout.addWidget(self.video_label)

        self.tab_widget = QTabWidget(self)
        list_width = int(window_width * 0.3)
        list_height = int(window_height * 0.7)
        self.tab_widget.setFixedSize(list_width, list_height)
        video_and_list_layout.addWidget(self.tab_widget)

        self.events_list = QListWidget(self)
        self.tab_widget.addTab(self.events_list, "События")

        self.search_list = QListWidget(self)
        self.tab_widget.addTab(self.search_list, "Поиск")

        self.lists_list = QListWidget(self)
        self.tab_widget.addTab(self.lists_list, "Списки")

        self.settings_tab = QWidget()
        self.init_settings_tab()
        self.tab_widget.addTab(self.settings_tab, "Настройки")

    def init_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)
        self.processing_settings_tab = QWidget()
        processing_layout = QFormLayout(self.processing_settings_tab)

        self.interval_edit = QLineEdit(self)
        self.interval_edit.setText(str(self.settings.processing.plate_image_send_interval))
        processing_layout.addRow("Отправка изображения каждые (кадров):", self.interval_edit)

        self.gpu_toggle = QLineEdit(self)
        self.gpu_toggle.setText(str(self.settings.ocr.gpu))
        processing_layout.addRow("OCR GPU (True/False):", self.gpu_toggle)

        layout.addWidget(self.processing_settings_tab)

        self.channels_settings_tab = QWidget()
        channels_layout = QFormLayout(self.channels_settings_tab)

        self.channel_count_edit = QLineEdit(self)
        self.channel_count_edit.setText(str(len(self.settings.app.video_paths)))
        channels_layout.addRow("Количество каналов (не больше 10):", self.channel_count_edit)

        self.video_path_edits = []
        for i in range(10):
            edit = QLineEdit(self)
            edit.setReadOnly(True)
            edit.mousePressEvent = lambda event, idx=i: self.select_video(event, idx)
            self.video_path_edits.append(edit)
            channels_layout.addRow(f"Путь к видео для канала {i + 1}:", edit)

        for i, path in enumerate(self.settings.app.video_paths):
            self.video_path_edits[i].setText(path)

        layout.addWidget(self.channels_settings_tab)

        save_button = QPushButton("Сохранить настройки", self)
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)

    def load_existing_records(self):
        for record in self.database.recent_plates():
            label = f"{record.plate} {record.region}".strip()
            self.events_list.addItem(QListWidgetItem(label))
            self.recognized_plates.add(label)
        self.refresh_lists_tab()

    def refresh_lists_tab(self):
        self.lists_list.clear()
        for name, description in self.database.lists():
            text = f"{name}: {description}" if description else name
            self.lists_list.addItem(QListWidgetItem(text))

    def start_processing(self):
        self.stop_processing()
        self.video_threads = []
        for video_path in self.settings.app.video_paths:
            video_thread = VideoThread(video_path, self.settings, self.database)
            video_thread.frame_signal.connect(self.update_frame)
            video_thread.text_signal.connect(self.update_text)
            video_thread.start()
            self.video_threads.append(video_thread)

    def stop_processing(self):
        for thread in getattr(self, "video_threads", []):
            if thread and thread.isRunning():
                thread.terminate()
                thread.wait()

    def select_video(self, event, index):
        self.stop_processing()
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi);;All Files (*)",
            options=options,
        )
        if file_name:
            self.video_path_edits[index].setText(file_name)

    def update_frame(self, image: QImage):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def update_text(self, text: str):
        if text and text not in self.recognized_plates:
            item = QListWidgetItem(text)
            self.events_list.addItem(item)
            self.recognized_plates.add(text)

    def save_settings(self):
        interval = self.interval_edit.text()
        if interval.isdigit():
            self.settings.processing.plate_image_send_interval = int(interval)

        gpu_value = self.gpu_toggle.text().strip().lower()
        if gpu_value in {"true", "false"}:
            self.settings.ocr.gpu = gpu_value == "true"

        new_channel_count = self.channel_count_edit.text()
        if new_channel_count.isdigit():
            channel_count = int(new_channel_count)
            if 0 < channel_count <= 10:
                new_paths = [edit.text() for edit in self.video_path_edits][:channel_count]
                self.settings.app.video_paths = [p for p in new_paths if p]
                self.start_processing()

        save_config(self.settings)

    def closeEvent(self, event):
        self.stop_processing()
        self.database.close()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
