import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk

# Project folders
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "project_outputs"

# Main window
root = tk.Tk()
root.title("Bank Marketing Classification Dashboard")
root.geometry("1200x700")
root.minsize(900, 550)

# Colours
BG_COLOR = "#F3F5F9"
SIDEBAR_COLOR = "#202B3C"
BUTTON_COLOR = "#34445D"
BUTTON_ACTIVE = "#46658A"
TEXT_COLOR = "#FFFFFF"

root.configure(bg=BG_COLOR)

# =========================
# LEFT SIDE: BUTTONS
# =========================

sidebar = tk.Frame(
    root,
    bg=SIDEBAR_COLOR,
    width=270
)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

title_label = tk.Label(
    sidebar,
    text="BANK MARKETING\nDASHBOARD",
    bg=SIDEBAR_COLOR,
    fg=TEXT_COLOR,
    font=("Arial", 17, "bold"),
    justify="left"
)
title_label.pack(padx=20, pady=(25, 30), anchor="w")

section_label = tk.Label(
    sidebar,
    text="VISUALIZATION RESULTS",
    bg=SIDEBAR_COLOR,
    fg="#AFC2DA",
    font=("Arial", 10, "bold")
)
section_label.pack(padx=20, pady=(0, 12), anchor="w")


# =========================
# RIGHT SIDE: RESULT AREA
# =========================

main_area = tk.Frame(root, bg=BG_COLOR)
main_area.pack(side="right", fill="both", expand=True)

result_title = tk.Label(
    main_area,
    text="Welcome",
    bg=BG_COLOR,
    fg="#202B3C",
    font=("Arial", 22, "bold")
)
result_title.pack(anchor="w", padx=30, pady=(25, 5))

description = tk.Label(
    main_area,
    text="Select a result from the left menu to display the visualization.",
    bg=BG_COLOR,
    fg="#64748B",
    font=("Arial", 11)
)
description.pack(anchor="w", padx=30, pady=(0, 20))

image_frame = tk.Frame(
    main_area,
    bg="white",
    highlightbackground="#D9E0EA",
    highlightthickness=1
)
image_frame.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=(0, 30)
)

image_label = tk.Label(
    image_frame,
    text="Your result will appear here",
    bg="white",
    fg="#64748B",
    font=("Arial", 14)
)
image_label.pack(expand=True, fill="both", padx=15, pady=15)

current_image = None


# =========================
# DISPLAY SELECTED PNG
# =========================

def show_image(filename, title):
    global current_image

    image_path = OUTPUT_DIR / filename

    if not image_path.exists():
        messagebox.showerror(
            "Image Not Found",
            f"Cannot find:\n{image_path}\n\n"
            "Please run your machine-learning script first."
        )
        return

    try:
        result_title.config(text=title)
        description.config(text=f"Displaying: {filename}")

        image = Image.open(image_path).convert("RGB")

        # Resize to fit the right-side result area
        available_width = max(image_frame.winfo_width() - 50, 500)
        available_height = max(image_frame.winfo_height() - 50, 350)

        image.thumbnail((available_width, available_height))

        current_image = ImageTk.PhotoImage(image)

        image_label.config(
            image=current_image,
            text=""
        )

    except Exception as error:
        messagebox.showerror(
            "Display Error",
            f"Unable to display image:\n{error}"
        )


# =========================
# LEFT MENU BUTTONS
# =========================

results = [
    ("Target Distribution",
     "target_distribution.png"),

    ("Numeric Distributions",
     "numeric_distributions.png"),

    ("Numeric Correlations",
     "numeric_correlations.png"),

    ("Logistic Regression",
     "confusion_matrix_logistic_regression.png"),

    ("Random Forest",
     "confusion_matrix_random_forest.png"),

    ("Gradient Boosting",
     "confusion_matrix_histogram_gradient_boosting.png"),

    ("Soft Voting Ensemble",
     "confusion_matrix_soft_voting_ensemble.png"),

    ("ROC Curve Comparison",
     "roc_curve_comparison.png"),
]

for button_text, filename in results:
    button = tk.Button(
        sidebar,
        text=button_text,
        command=lambda f=filename, t=button_text: show_image(f, t),
        bg=BUTTON_COLOR,
        fg=TEXT_COLOR,
        activebackground=BUTTON_ACTIVE,
        activeforeground=TEXT_COLOR,
        font=("Arial", 10),
        anchor="w",
        relief="flat",
        cursor="hand2",
        padx=15,
        pady=10
    )
    button.pack(fill="x", padx=15, pady=4)

root.mainloop()
