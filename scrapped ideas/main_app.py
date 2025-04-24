import sys
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
import time

# --- PyQt Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QDateEdit, QMessageBox, QSpinBox
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QDate, QUrl, Qt, QTimer, QThread, pyqtSignal, QObject, pyqtSlot
# CORRECTED IMPORT: Added QFont back
from PyQt6.QtGui import QFont, QPalette, QColor

# --- IMPORT FROM YOUR SRC FOLDER ---
try:
    from src.features.engineer import compute_features
    print("--- Successfully imported compute_features ---")
except ImportError as e:
    print(f"!!! ERROR importing compute_features: {e} !!!")
    print("!!! Feature computation will be disabled. !!!")
    compute_features = None

# --- Worker Object for Threading ---
# (Worker class definition remains the same as previous correct version)
class Worker(QObject):
    finished_task = pyqtSignal()
    data_ready = pyqtSignal(object, object, str)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._is_running = True

    @pyqtSlot(str, int, object)
    def do_update_task(self, ticker, sma_window, current_df_raw_from_main):
        print(f"--- Worker: Received task request for {ticker} ---")
        local_current_df_raw = current_df_raw_from_main

        try:
            # Fetching Logic
            if local_current_df_raw is None or local_current_df_raw.empty:
                 print(f"--- Worker: Fetching initial data for {ticker} ---")
                 initial_df_raw = self.fetch_stock_data_realtime(ticker, period="5d", interval="1m")
                 if initial_df_raw is None or initial_df_raw.empty:
                     self.error.emit(f"Failed to fetch initial data for {ticker}")
                     self.finished_task.emit(); return
                 local_current_df_raw = initial_df_raw
            else:
                print(f"--- Worker: Fetching latest data for {ticker} ---")
                latest_df = self.fetch_stock_data_realtime(ticker, period="1d", interval="1m")
                if latest_df is not None and not latest_df.empty:
                    combined_df = pd.concat([local_current_df_raw, latest_df])
                    local_current_df_raw = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
                    local_current_df_raw = local_current_df_raw.tail(1000)
                    print(f"--- Worker: Updated data. Total points: {len(local_current_df_raw)} ---")
                else:
                    print("--- Worker: No new data fetched, using previous data. ---")

            # Feature Processing (Conditional)
            df_features = None
            if compute_features and sma_window >= 2 and local_current_df_raw is not None and not local_current_df_raw.empty:
                print(f"--- Worker: Computing features with window {sma_window} ---")
                try:
                    df_features = compute_features(local_current_df_raw.copy(), window=sma_window)
                except Exception as e:
                     print(f"--- Worker: EXCEPTION computing features: {e} ---")
                     self.error.emit(f"Error computing features: {e}")
                     df_features = None
            else:
                 print(f"--- Worker: Skipping feature computation ---")

            # Emit Results
            print("--- Worker: Emitting data_ready signal ---")
            self.data_ready.emit(local_current_df_raw.copy() if local_current_df_raw is not None else None,
                                 df_features.copy() if df_features is not None else None,
                                 ticker)
        except Exception as e:
            print(f"--- Worker Task Error: {e} ---")
            self.error.emit(f"Error in worker task: {e}")
        finally:
            self.finished_task.emit()
            print("--- Worker Task Finished ---")

    def stop_processing(self):
        print("--- Worker Stop Processing Requested ---")
        self._is_running = False

    def fetch_stock_data_realtime(self, ticker, period="1d", interval="1m"):
         print(f"--- Fetching {ticker} | Period: {period} | Interval: {interval} ---")
         try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
            if df.empty:
                print(f"--- No data returned by yfinance for {ticker} ({period}/{interval}) ---")
                return None
            return df
         except Exception as e:
            print(f"--- EXCEPTION in fetch_stock_data_realtime for {ticker}: {e} ---")
            return None

# --- Main Application Window ---
class StockAppWindow(QMainWindow):
    trigger_update_task = pyqtSignal(str, int, object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Analysis Application (Real-Time Simulation - Reusable Thread)")
        self.setGeometry(100, 100, 1200, 800)

        # --- FONT DEFINITIONS ---
        # QFont was missing from imports in previous version, added back above
        self.label_font = QFont("Arial", 10)
        self.input_font = QFont("Courier New", 10)
        self.button_font = QFont("Arial", 11, QFont.Weight.Bold)
        self.date_font = QFont("Arial", 10)

        # --- Data Storage ---
        self._current_ticker = "AAPL"
        self._current_raw_df = None
        self._current_features_df = None

        # --- Threading Members ---
        self.worker_thread = None
        self.worker = None
        self._is_updating = False

        # --- Central Widget and Layouts / Input Controls ---
        # (Setup remains the same, applies fonts correctly now)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        controls_layout = QVBoxLayout()
        chart_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(chart_layout, 4)

        self.ticker_label = QLabel("Ticker Symbol:")
        self.ticker_label.setFont(self.label_font) # Apply font
        controls_layout.addWidget(self.ticker_label)
        self.ticker_input = QLineEdit(self._current_ticker)
        self.ticker_input.setFont(self.input_font) # Apply font
        self.ticker_input.editingFinished.connect(self.ticker_changed)
        controls_layout.addWidget(self.ticker_input)

        self.start_date_label = QLabel("Start Date (for initial load):")
        self.start_date_label.setFont(self.label_font) # Apply font
        controls_layout.addWidget(self.start_date_label)
        self.start_date_edit = QDateEdit(calendarPopup=True)
        self.start_date_edit.setDate(QDate.currentDate().addDays(-7))
        self.start_date_edit.setFont(self.date_font) # Apply font
        controls_layout.addWidget(self.start_date_edit)

        self.end_date_label = QLabel("End Date (for initial load):")
        self.end_date_label.setFont(self.label_font) # Apply font
        controls_layout.addWidget(self.end_date_label)
        self.end_date_edit = QDateEdit(calendarPopup=True)
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.setFont(self.date_font) # Apply font
        controls_layout.addWidget(self.end_date_edit)

        self.sma_window_spinbox = None
        if compute_features:
             self.sma_window_label = QLabel("SMA Window (0=Off):")
             self.sma_window_label.setFont(self.label_font) # Apply font
             controls_layout.addWidget(self.sma_window_label)
             self.sma_window_spinbox = QSpinBox()
             self.sma_window_spinbox.setMinimum(0)
             self.sma_window_spinbox.setMaximum(100)
             self.sma_window_spinbox.setValue(10)
             self.sma_window_spinbox.setFont(self.date_font) # Apply font
             controls_layout.addWidget(self.sma_window_spinbox)

        self.start_stop_button = QPushButton("Start Real-Time")
        self.start_stop_button.setCheckable(True)
        self.start_stop_button.clicked.connect(self.toggle_realtime)
        self.start_stop_button.setFont(self.button_font) # Apply font
        controls_layout.addWidget(self.start_stop_button)

        controls_layout.addStretch()

        self.chart_view = QWebEngineView()
        chart_layout.addWidget(self.chart_view)

        # --- Real-time update timer ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.request_data_update)
        self.update_interval_ms = 30000

        # --- Setup Reusable Thread and Worker ---
        self.setup_worker_thread()

        # --- Initial State ---
        self.display_chart(None)

    # --- Methods (setup_worker_thread, ticker_changed, toggle_realtime, etc.) ---
    # (All methods from the previous correct version remain the same)
    def setup_worker_thread(self):
        print("--- Setting up worker thread ---")
        self.worker_thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.worker_thread)
        self.worker.data_ready.connect(self.handle_data_update)
        self.worker.error.connect(self.handle_worker_error)
        self.worker.finished_task.connect(self.on_worker_finished_task)
        self.trigger_update_task.connect(self.worker.do_update_task)
        self.worker_thread.finished.connect(self.on_thread_finished)
        self.worker_thread.start()
        print("--- Worker thread started and running ---")

    def ticker_changed(self):
        new_ticker = self.ticker_input.text().upper()
        if new_ticker != self._current_ticker:
            print(f"--- Ticker changed from {self._current_ticker} to {new_ticker} ---")
            was_running = self.start_stop_button.isChecked()
            if was_running: self.toggle_realtime(False)
            self._current_ticker = new_ticker
            self._current_raw_df = None
            self._current_features_df = None
            self.display_chart(None)
            if was_running: self.toggle_realtime(True)

    def toggle_realtime(self, checked):
        if checked:
            if self._is_updating:
                 print("--- Cannot start: Update task already in progress ---")
                 self.start_stop_button.setChecked(False); return
            print("--- Starting Real-Time Updates ---")
            self.start_stop_button.setText("Stop Real-Time")
            self.request_data_update()
            if not self._is_updating:
                 print("--- Failed to start initial update request ---")
                 self.start_stop_button.setChecked(False)
                 self.start_stop_button.setText("Start Real-Time")
            else:
                 self.timer.start(self.update_interval_ms)
        else:
            print("--- Stopping Real-Time Updates ---")
            self.start_stop_button.setText("Start Real-Time")
            self.timer.stop()

    def request_data_update(self):
        if self._is_updating:
            print(f"--- Update request skipped (Task already running) ---")
            return
        if self.worker is None or self.worker_thread is None or not self.worker_thread.isRunning():
             print("--- Cannot update: Worker/Thread not ready. ---")
             self.start_stop_button.setChecked(False)
             self.start_stop_button.setText("Start Real-Time")
             self.timer.stop(); return

        print("--- Requesting Data Update Task ---")
        self._is_updating = True
        ticker = self._current_ticker
        sma_window = self.sma_window_spinbox.value() if self.sma_window_spinbox else 10
        df_to_pass = self._current_raw_df.copy() if self._current_raw_df is not None else None
        self.trigger_update_task.emit(ticker, sma_window, df_to_pass)

    @pyqtSlot()
    def on_worker_finished_task(self):
        print("--- Worker task finished signal received. Resetting update flag. ---")
        self._is_updating = False

    @pyqtSlot(object, object, str)
    def handle_data_update(self, df_raw, df_features, ticker):
        print(f"--- Main Thread: Received data_ready signal for {ticker} ---")
        if ticker != self._current_ticker:
            print(f"--- Ignoring stale data for {ticker}, current is {self._current_ticker} ---")
            return

        self._current_raw_df = df_raw
        self._current_features_df = df_features

        if self.start_stop_button.isChecked():
             fig = self.create_candlestick_chart(self._current_raw_df, self._current_features_df, self._current_ticker)
             self.display_chart(fig)
        else:
            print("--- Main Thread: Real-time stopped, ignoring UI update. ---")

    @pyqtSlot(str)
    def handle_worker_error(self, error_message):
        print(f"--- Main Thread: Received error signal: {error_message} ---")
        self.show_error_message(error_message)
        self._is_updating = False
        if self.start_stop_button.isChecked():
             self.start_stop_button.setChecked(False)
             self.toggle_realtime(False)

    def create_candlestick_chart(self, df_raw, df_features, ticker):
        # (Same as previous version)
        print("--- Inside create_candlestick_chart ---")
        if df_raw is None or df_raw.empty:
            print("!!! ERROR: Raw DataFrame is None or empty for charting !!!")
            return None
        else:
            print("--- Charting Raw data tail: ---")
            print(df_raw[['Open', 'High', 'Low', 'Close']].tail())

        try:
            fig = go.Figure(data=[go.Candlestick(x=df_raw.index,
                                                 open=df_raw['Open'], high=df_raw['High'],
                                                 low=df_raw['Low'], close=df_raw['Close'],
                                                 name=ticker)])
        except Exception as e:
            print(f"!!! EXCEPTION creating go.Candlestick: {e} !!!")
            self.show_error_message(f"Error creating candlestick figure: {e}")
            return None

        if df_features is not None and not df_features.empty:
            print("--- Charting Feature data tail: ---")
            print(df_features.tail())
            sma_col = [col for col in df_features.columns if col.startswith('sma_')]
            if sma_col:
                print(f"--- Adding SMA trace: {sma_col[0]} ---")
                try:
                    fig.add_trace(go.Scatter(x=df_features.index, y=df_features[sma_col[0]],
                                              mode='lines', name=sma_col[0].replace('_', ' ').upper(),
                                              line=dict(color='orange', width=1)))
                except Exception as e: print(f"!!! EXCEPTION adding SMA trace: {e} !!!")
            else: print("--- SMA column not found in features ---")
        else: print("--- No features to add to chart ---")

        try:
            fig.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Time",
                              yaxis_title="Price (USD)", xaxis_rangeslider_visible=False,
                              template="plotly_dark", legend_title_text='Legend')
        except Exception as e: print(f"!!! EXCEPTION updating layout: {e} !!!")

        print("--- Figure object created/updated for display ---")
        return fig

    def display_chart(self, fig):
        # (Same as before)
        print("--- Attempting to display chart in QWebEngineView ---")
        if fig is None:
             print("--- Figure is None, displaying placeholder HTML ---")
             placeholder_html = """
             <html><head><style>body{background-color: #111; color: #777; display: flex; justify-content: center; align-items: center; height: 100%; margin: 0; font-family: sans-serif;}</style></head>
             <body><h3>Chart Area: Start Real-Time or check ticker/console.</h3></body></html>
             """
             self.chart_view.setHtml(placeholder_html)
             return
        try:
            raw_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            print("--- Setting HTML in QWebEngineView ---")
            self.chart_view.setHtml(raw_html)
        except Exception as e:
            print(f"!!! EXCEPTION converting or setting HTML: {e} !!!")
            error_html = f"<html><body>Error displaying chart: {e}</body></html>"
            self.chart_view.setHtml(error_html)

    def show_error_message(self, message):
        # (Same as before)
        print(f"--- ERROR MESSAGE BOX: {message} ---")
        QMessageBox.warning(self, "Error", message)

    @pyqtSlot()
    def on_thread_finished(self):
        # (Same as before)
        print("--- Worker thread finished event loop. ---")

    def closeEvent(self, event):
        # (Same as before)
        print("--- Close Event Triggered ---")
        self.timer.stop()
        if self.worker is not None:
             self.worker.stop_processing()
        if self.worker_thread is not None and self.worker_thread.isRunning():
             print("--- Quitting worker thread event loop... ---")
             self.worker_thread.quit()
             print("--- Waiting for worker thread to finish on close... ---")
             if not self.worker_thread.wait(5000):
                 print("--- Worker thread did not finish gracefully, terminating. ---")
                 self.worker_thread.terminate()
                 self.worker_thread.wait()
             else:
                  print("--- Worker thread finished gracefully. ---")
        print("--- Exiting application ---")
        event.accept()

# --- Application Entry Point ---
# (Same as before)
if __name__ == "__main__":
    def excepthook(exc_type, exc_value, exc_tb):
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_tb)
        error_msg = f"{exc_type.__name__}: {exc_value}"
        print(f"!!! UNHANDLED EXCEPTION: {error_msg} !!!")
        try: QMessageBox.critical(None, "Unhandled Exception", error_msg)
        except: pass
        sys.exit(1)
    sys.excepthook = excepthook

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = StockAppWindow()
    window.show()
    sys.exit(app.exec())