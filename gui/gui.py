"""/gui/gui.py"""
from gui.common import *
from gui.gui_settings import SettingsDialog
from process_video_realtime import process_video_realtime

class VideoThread(QThread):
    frame_signal = pyqtSignal(QImage)
    text_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            process_video_realtime(self.video_path, self.frame_signal.emit, self.text_signal.emit)
        except Exception as e:
            print(f"Error during video processing: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recognized_plates = set()  # Множество для хранения распознанных номеров
        self.video_threads = []  # Инициализация списка потоков
        self.load_config()  # Загрузка конфигурации перед инициализацией UI
        self.initUI()
        self.start_processing()

    def load_config(self):
        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)

    def save_config(self):
        with open('config.yaml', 'w') as file:
            yaml.dump(self.config, file)

    def update_text(self, text):
        if text and text not in self.recognized_plates:
            item = QListWidgetItem(text)
            self.events_list.addItem(item)
            self.recognized_plates.add(text)  # Добавляем номер в множество

    def initUI(self):
        self.setWindowTitle('Программа для распознования автомобильных номеров')

        # Получение размеров экрана
        screen = QDesktopWidget().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()

        # Уменьшение размеров окна на 20%
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)

        # Расположение окна по центру экрана
        self.setGeometry((screen_width - window_width) // 2, (screen_height - window_height) // 2, window_width, window_height)

        # Основной виджет
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # Вертикальный layout для основного виджета
        main_layout = QVBoxLayout(main_widget)

        # Горизонтальный layout для виджетов видео и списка номеров
        video_and_list_layout = QHBoxLayout()
        main_layout.addLayout(video_and_list_layout)

        # Виджет для отображения видео
        self.video_label = QLabel(self)
        video_width = int(window_width * 0.7)  # Ширина видео (70% от ширины окна)
        video_height = int(window_height * 0.7)  # Высота видео (70% от высоты окна)
        self.video_label.setFixedSize(video_width, video_height)  # Установка фиксированного размера для видео
        self.video_label.setScaledContents(True)  # Масштабирование содержимого
        video_and_list_layout.addWidget(self.video_label)

        # Виджет для отображения списка распознанных номеров с вкладками
        self.tab_widget = QTabWidget(self)
        list_width = int(window_width * 0.3)  # Ширина списка (30% от ширины окна)
        list_height = int(window_height * 0.7)  # Высота списка (70% от высоты окна)
        self.tab_widget.setFixedSize(list_width, list_height)  # Установка фиксированного размера для списка
        video_and_list_layout.addWidget(self.tab_widget)

        # Вкладка "События"
        self.events_list = QListWidget(self)
        self.tab_widget.addTab(self.events_list, "События")

        # Вкладка "Поиск"
        self.search_list = QListWidget(self)
        self.tab_widget.addTab(self.search_list, "Поиск")

        # Вкладка "Списки"
        self.lists_list = QListWidget(self)
        self.tab_widget.addTab(self.lists_list, "Списки")

        # Вкладка "Настройки"
        self.settings_tab = QWidget()
        self.init_settings_tab()
        self.tab_widget.addTab(self.settings_tab, "Настройки")

        # Инициализация потока обработки видео
        self.video_thread = None

    def start_processing(self):
        self.stop_processing()  # Останавливаем текущий поток, если он запущен
        self.video_threads = []
        for video_path in self.config['video_paths']:
            video_thread = VideoThread(video_path)
            video_thread.frame_signal.connect(self.update_frame)
            video_thread.text_signal.connect(self.update_text)
            video_thread.start()
            self.video_threads.append(video_thread)

    def stop_processing(self):
        for video_thread in self.video_threads:
            if video_thread and video_thread.isRunning():
                video_thread.terminate()
                video_thread.wait()  # Дожидаемся завершения потока

    def update_frame(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

    def init_settings_tab(self):
        # Инициализация вкладки "Настройки"
        layout = QVBoxLayout(self.settings_tab)

        # Подвкладки для настроек
        self.settings_tab_widget = QTabWidget(self)
        layout.addWidget(self.settings_tab_widget)

        # Подвкладка "Настройки обработки видео"
        self.processing_settings_tab = QWidget()
        self.init_processing_settings_tab()
        self.settings_tab_widget.addTab(self.processing_settings_tab, "Настройки обработки видео")

        # Подвкладка "Настройки каналов"
        self.channels_settings_tab = QWidget()
        self.init_channels_settings_tab()
        self.settings_tab_widget.addTab(self.channels_settings_tab, "Настройки каналов")

        # Кнопка "Сохранить настройки"
        save_button = QPushButton('Сохранить настройки', self)
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)

    def init_processing_settings_tab(self):
        # Инициализация подвкладки "Настройки обработки видео"
        layout = QFormLayout(self.processing_settings_tab)

        # Поле для редактирования plate_image_send_interval
        self.interval_edit = QLineEdit(self)
        self.interval_edit.setText(str(self.config['plate_image_send_interval']))
        layout.addRow("Отправка изображения каждые (кадров):", self.interval_edit)

    def init_channels_settings_tab(self):
        # Инициализация подвкладки "Настройки каналов"
        layout = QFormLayout(self.channels_settings_tab)

        # Поле для ввода количества каналов
        self.channel_count_edit = QLineEdit(self)
        self.channel_count_edit.setText(str(len(self.config['video_paths'])))
        layout.addRow("Количество каналов (не больше 10):", self.channel_count_edit)

        # Поля для ввода путей к видео для каждого канала
        self.video_path_edits = []
        for i in range(10):
            video_path_edit = QLineEdit(self)
            video_path_edit.setReadOnly(True)
            video_path_edit.mousePressEvent = lambda event, idx=i: self.select_video(event, idx)
            self.video_path_edits.append(video_path_edit)
            layout.addRow(f"Путь к видео для канала {i+1}:", video_path_edit)

        # Заполняем поля текущими значениями
        for i, path in enumerate(self.config['video_paths']):
            self.video_path_edits[i].setText(path)

    def select_video(self, event, index):
        self.stop_processing()  # Останавливаем текущий поток обработки видео
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        if file_name:
            self.video_path_edits[index].setText(file_name)

    def save_settings(self):
        # Сохраняем настройки
        new_interval = self.interval_edit.text()
        new_channel_count = self.channel_count_edit.text()
        new_video_paths = [edit.text() for edit in self.video_path_edits]

        if new_interval.isdigit():
            self.config['plate_image_send_interval'] = int(new_interval)
        if new_channel_count.isdigit():
            channel_count = int(new_channel_count)
            if 0 < channel_count <= 10:
                self.config['video_paths'] = new_video_paths[:channel_count]
                self.start_processing()  # Перезапускаем обработку с новыми путями к видео
        self.save_config()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())