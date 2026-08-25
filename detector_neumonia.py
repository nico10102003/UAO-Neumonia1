#!/usr/bin/env python

"""Interfaz gráfica para la detección de neumonía."""

import csv
import tkinter as tk
from tkinter import filedialog, font, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from src.integrator import predict
from src.read_img import read_dicom_file, read_jpg_file


class App:
    """Aplicación gráfica para cargar imágenes y realizar predicciones."""

    def __init__(self):
        """Inicializa la interfaz gráfica y sus componentes."""
        self.root = tk.Tk()
        self.root.title("Detección de Neumonía")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        fonti = font.Font(weight="bold")

        # Labels
        self.lab1 = ttk.Label(
            self.root,
            text="Imagen Radiográfica",
            font=fonti,
        )

        self.lab2 = ttk.Label(
            self.root,
            text="Imagen con Heatmap",
            font=fonti,
        )

        self.lab3 = ttk.Label(
            self.root,
            text="Resultado:",
            font=fonti,
        )

        self.lab4 = ttk.Label(
            self.root,
            text="Cédula Paciente:",
            font=fonti,
        )

        self.lab5 = ttk.Label(
            self.root,
            text=(
                "SOFTWARE PARA EL APOYO AL DIAGNÓSTICO "
                "MÉDICO DE NEUMONÍA"
            ),
            font=fonti,
        )

        self.lab6 = ttk.Label(
            self.root,
            text="Probabilidad:",
            font=fonti,
        )

        # Variables
        self.ID = tk.StringVar()

        # Entrada de identificación
        self.text1 = ttk.Entry(
            self.root,
            textvariable=self.ID,
            width=10,
        )

        # Áreas de imágenes y resultados
        self.text_img1 = tk.Text(
            self.root,
            width=31,
            height=15,
        )

        self.text_img2 = tk.Text(
            self.root,
            width=31,
            height=15,
        )

        self.text2 = tk.Text(self.root)
        self.text3 = tk.Text(self.root)

        # Botones
        self.button1 = ttk.Button(
            self.root,
            text="Predecir",
            state="disabled",
            command=self.run_model,
        )

        self.button2 = ttk.Button(
            self.root,
            text="Cargar Imagen",
            command=self.load_img_file,
        )

        self.button3 = ttk.Button(
            self.root,
            text="Borrar",
            command=self.delete,
        )

        self.button4 = ttk.Button(
            self.root,
            text="PDF",
            command=self.create_pdf,
        )

        self.button6 = ttk.Button(
            self.root,
            text="Guardar",
            command=self.save_results_csv,
        )

        # Posiciones
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)

        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)

        self.text1.place(x=200, y=350)

        self.text2.place(
            x=610,
            y=350,
            width=90,
            height=30,
        )

        self.text3.place(
            x=610,
            y=400,
            width=90,
            height=30,
        )

        self.text_img1.place(
            x=65,
            y=90,
        )

        self.text_img2.place(
            x=500,
            y=90,
        )

        # Variables de la aplicación
        self.array = None
        self.label = None
        self.proba = None
        self.heatmap = None
        self.img1 = None
        self.img2 = None
        self.reportID = 0

        self.text1.focus_set()
        self.root.mainloop()

    def load_img_file(self):
        """Carga una imagen DICOM o convencional desde el sistema."""
        filepath = filedialog.askopenfilename(
            initialdir="/mnt/c/Users/Windows 11",
            title="Seleccionar imagen",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
            ),
        )

        if not filepath:
            return

        if filepath.lower().endswith(".dcm"):
            self.array, img2show = read_dicom_file(filepath)
        else:
            self.array, img2show = read_jpg_file(filepath)

        self.img1 = img2show.resize(
            (250, 250),
            Image.Resampling.LANCZOS,
        )

        self.img1 = ImageTk.PhotoImage(self.img1)

        self.text_img1.image_create(
            tk.END,
            image=self.img1,
        )

        self.button1["state"] = "enabled"

    def run_model(self):
        """Ejecuta la predicción y muestra el resultado y Grad-CAM."""
        if self.array is None:
            messagebox.showinfo(
                title="Advertencia",
                message="Primero debe cargar una imagen.",
            )
            return

        self.label, self.proba, self.heatmap = predict(
            self.array
        )

        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize(
            (250, 250),
            Image.Resampling.LANCZOS,
        )

        self.img2 = ImageTk.PhotoImage(self.img2)

        self.text_img2.image_create(
            tk.END,
            image=self.img2,
        )

        self.text2.delete(
            "1.0",
            tk.END,
        )

        self.text3.delete(
            "1.0",
            tk.END,
        )

        self.text2.insert(
            tk.END,
            self.label,
        )

        self.text3.insert(
            tk.END,
            f"{self.proba:.2f}%",
        )

    def save_results_csv(self):
        """Guarda la identificación y el resultado en historial.csv."""
        if self.label is None:
            messagebox.showinfo(
                title="Advertencia",
                message="Primero debe realizar una predicción.",
            )
            return

        with open(
            "historial.csv",
            "a",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.writer(
                csvfile,
                delimiter="-",
            )

            writer.writerow(
                [
                    self.text1.get(),
                    self.label,
                    f"{self.proba:.2f}%",
                ]
            )

        messagebox.showinfo(
            title="Guardar",
            message="Los datos se guardaron con éxito.",
        )

    def create_pdf(self):
        """Genera un reporte PDF con la predicción y el mapa de calor."""
        if self.label is None:
            messagebox.showinfo(
                title="Advertencia",
                message="Primero debe realizar una predicción.",
            )
            return

        pdf_path = f"Reporte{self.reportID}.pdf"

        try:
            report_image = Image.new(
                "RGB",
                (1000, 800),
                "white",
            )

            draw = ImageDraw.Draw(report_image)

            draw.text(
                (50, 40),
                "REPORTE DE DETECCIÓN DE NEUMONÍA",
                fill="black",
            )

            draw.text(
                (50, 100),
                f"Predicción: {self.label}",
                fill="black",
            )

            draw.text(
                (50, 140),
                f"Probabilidad: {self.proba:.2f}%",
                fill="black",
            )

            if self.heatmap is not None:
                heatmap = Image.fromarray(self.heatmap)
                heatmap = heatmap.resize((512, 512))

                report_image.paste(
                    heatmap,
                    (50, 200),
                )

            report_image.save(
                pdf_path,
                "PDF",
                resolution=100.0,
            )

            self.reportID += 1

            messagebox.showinfo(
                title="PDF",
                message="El PDF fue generado con éxito.",
            )

        except (OSError, TypeError, ValueError) as error:
            messagebox.showinfo(
                title="Error",
                message=(
                    "No fue posible generar el PDF:\n"
                    f"{error}"
                ),
            )

    def delete(self):
        """Borra los datos actuales y reinicia la interfaz."""
        answer = messagebox.askokcancel(
            title="Confirmación",
            message="Se borrarán todos los datos.",
            icon=messagebox.WARNING,
        )

        if not answer:
            return

        self.text1.delete(
            0,
            "end",
        )

        self.text2.delete(
            "1.0",
            "end",
        )

        self.text3.delete(
            "1.0",
            "end",
        )

        self.text_img1.delete(
            "1.0",
            "end",
        )

        self.text_img2.delete(
            "1.0",
            "end",
        )

        self.array = None
        self.label = None
        self.proba = None
        self.heatmap = None
        self.img1 = None
        self.img2 = None

        self.button1["state"] = "disabled"

        messagebox.showinfo(
            title="Borrar",
            message="Los datos se borraron con éxito.",
        )


def main():
    """Inicia la aplicación gráfica."""
    App()


if __name__ == "__main__":
    main()