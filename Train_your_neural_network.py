import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QComboBox,
    QSpinBox, QTextEdit, QProgressBar
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix


class NeuralNetworkGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HR Employee Resignation Prediction")
        self.setGeometry(200, 200, 800, 650)

        self.df = None
        self.data = None

        self.build_ui()

    # ================= UI =================
    def build_ui(self):
        layout = QVBoxLayout()

        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_data)

        self.clean_btn = QPushButton("Clean Data")
        self.clean_btn.clicked.connect(self.clean_data)

        self.activation = QComboBox()
        self.activation.addItems(["ReLU", "Sigmoid", "Tanh"])

        self.neurons = QSpinBox()
        self.neurons.setRange(1, 500)
        self.neurons.setValue(19)

        self.epochs = QSpinBox()
        self.epochs.setRange(10, 5000)
        self.epochs.setValue(100)

        self.lr = QComboBox()
        self.lr.addItems(["0.1", "0.01", "0.001"])

        self.progress = QProgressBar()

        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train)

        # 🔥 NEW BUTTON
        self.save_btn = QPushButton("Download Model")
        self.save_btn.clicked.connect(self.save_model)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        layout.addWidget(self.load_btn)
        layout.addWidget(self.clean_btn)
        layout.addWidget(self.activation)
        layout.addWidget(self.neurons)
        layout.addWidget(self.epochs)
        layout.addWidget(self.lr)
        layout.addWidget(self.progress)
        layout.addWidget(self.train_btn)

        # 🔥 ADD BUTTON HERE
        layout.addWidget(self.save_btn)

        layout.addWidget(self.output)

        self.setLayout(layout)

    # ================= DATA =================
    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load CSV", "", "CSV (*.csv)")
        if not path:
            return

        self.df = pd.read_csv(path)
        self.output.append(f"Loaded: {self.df.shape}")

    def clean_data(self):
        if self.df is None:
            QMessageBox.warning(self, "Error", "Load file first")
            return

        df = self.df.copy().dropna()

        drop_cols = ["Employee_ID", "Hire_Date"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        cat_cols = ["Department", "Gender", "Job_Title", "Education_Level"]
        cat_cols = [c for c in cat_cols if c in df.columns]
        df = pd.get_dummies(df, columns=cat_cols)

        if "Hire_Date" in self.df.columns:
            df["Hire_Year"] = pd.to_datetime(self.df["Hire_Date"]).dt.year

        df["Resigned"] = df["Resigned"].astype(int)

        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        self.data = df
        self.output.append(f"Cleaned: {df.shape}")

    # ================= ACTIVATIONS =================
    def act(self, x, name):
        if name == "ReLU":
            return np.maximum(0, x)
        if name == "Sigmoid":
            return 1 / (1 + np.exp(-x))
        return np.tanh(x)

    def act_deriv(self, x, name):
        if name == "ReLU":
            return (x > 0).astype(float)
        if name == "Sigmoid":
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        return 1 - np.tanh(x) ** 2

    # ================= TRAIN =================
    def train(self):
        if self.data is None:
            QMessageBox.warning(self, "Error", "Clean data first")
            return

        X = self.data.drop(columns=["Resigned"]).values
        y = self.data["Resigned"].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        input_size = X.shape[1]
        hidden = self.neurons.value()
        epochs = self.epochs.value()
        lr = float(self.lr.currentText())
        act_name = self.activation.currentText()

        W1 = np.random.randn(input_size, hidden) * 0.01
        W2 = np.random.randn(hidden, 1) * 0.01
        b1 = np.zeros((1, hidden))
        b2 = np.zeros((1, 1))

        losses = []

        for i in range(epochs):

            z1 = X_train @ W1 + b1
            a1 = self.act(z1, act_name)

            z2 = a1 @ W2 + b2
            a2 = 1 / (1 + np.exp(-z2))

            loss = -np.mean(y_train*np.log(a2+1e-8) + (1-y_train)*np.log(1-a2+1e-8))
            losses.append(loss)

            dz2 = a2 - y_train
            dW2 = a1.T @ dz2 / len(X_train)
            db2 = np.mean(dz2, axis=0, keepdims=True)

            da1 = dz2 @ W2.T
            dz1 = da1 * self.act_deriv(z1, act_name)

            dW1 = X_train.T @ dz1 / len(X_train)
            db1 = np.mean(dz1, axis=0, keepdims=True)

            W1 -= lr * dW1
            W2 -= lr * dW2
            b1 -= lr * db1
            b2 -= lr * db2

            self.progress.setValue(int(i / epochs * 100))

            if i % 10 == 0:
                self.output.append(f"Epoch {i} Loss {loss:.4f}")
                QApplication.processEvents()

        # ================= SAVE IN MEMORY =================
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2
        self.act_name = act_name

        # ================= TEST =================
        a1 = self.act(X_test @ W1 + b1, act_name)
        a2 = 1 / (1 + np.exp(-(a1 @ W2 + b2)))

        preds = (a2 > 0.5).astype(int)

        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        self.output.append("\nDone Training")
        self.output.append(f"Accuracy: {acc*100:.2f}%")
        self.output.append(f"Confusion Matrix:\n{cm}")

        plt.plot(losses)
        plt.title("Loss Curve")
        plt.show()

    # ================= SAVE MODEL =================
    def save_model(self):

        if not hasattr(self, "W1"):
            QMessageBox.warning(self, "Error", "Train model first")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            "",
            "Pickle File (*.pkl)"
        )

        if not path:
            return

        model = {
            "W1": self.W1,
            "W2": self.W2,
            "b1": self.b1,
            "b2": self.b2,
            "activation": self.act_name
        }

        with open(path, "wb") as f:
            pickle.dump(model, f)

        self.output.append("Model saved successfully ✔")


# ================= RUN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = NeuralNetworkGUI()
    win.show()
    sys.exit(app.exec_())