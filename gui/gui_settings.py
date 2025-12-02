"""/gui/settings_gui.py"""
from gui.common import *

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Настройки')
        self.initUI()

    def initUI(self):
        self.tab_widget = QTabWidget(self)

        # Вкладка "Настройки обработки видео"
        self.processing_tab = QWidget()
        self.init_processing_tab()
        self.tab_widget.addTab(self.processing_tab, "Настройки обработки видео")

        # Вкладка "Каналы"
        self.channels_tab = QWidget()
        self.init_channels_tab()
        self.tab_widget.addTab(self.channels_tab, "Каналы")

        # Основной layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget)

        # Кнопки "Сохранить" и "Отмена"
        button_box = QHBoxLayout()
        save_button = QPushButton('Сохранить', self)
        save_button.clicked.connect(self.accept)
        button_box.addWidget(save_button)

        cancel_button = QPushButton('Отмена', self)
        cancel_button.clicked.connect(self.reject)
        button_box.addWidget(cancel_button)

        main_layout.addLayout(button_box)

    def init_processing_tab(self):
        layout = QFormLayout(self.processing_tab)

        # Поле для редактирования plate_image_send_interval
        self.interval_edit = QLineEdit(self)
        layout.addRow("Отправка изображения каждые (кадров):", self.interval_edit)

    def init_channels_tab(self):
        layout = QFormLayout(self.channels_tab)

        # Поле для ввода количества каналов
        self.channel_count_edit = QLineEdit(self)
        layout.addRow("Количество каналов (не больше 10):", self.channel_count_edit)

        # Поля для ввода путей к видео для каждого канала
        self.video_path_edits = []
        for i in range(10):
            video_path_edit = QLineEdit(self)
            video_path_edit.setReadOnly(True)
            video_path_edit.mousePressEvent = lambda event, idx=i: self.select_video(event, idx)
            self.video_path_edits.append(video_path_edit)
            layout.addRow(f"Путь к видео для канала {i+1}:", video_path_edit)

    def select_video(self, event, index):
        self.parent().stop_processing()  # Останавливаем текущий поток обработки видео
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        if file_name:
            self.video_path_edits[index].setText(file_name)

    def get_settings(self):
        interval = self.interval_edit.text()
        channel_count = self.channel_count_edit.text()
        video_paths = [edit.text() for edit in self.video_path_edits]
        return interval, channel_count, video_paths