import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

regression_model = joblib.load("gpa_regression.joblib")
classification_model = joblib.load("gradeclass_classifier.joblib")
grade_labels = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
predicted_gpas = []
predicted_grades = []

root = tk.Tk()
root.title("Student Performance Prediction")
root.geometry("600x750")
root.resizable(False, False)

fields = {}

def add_field(label, row, widget_type="entry", values=None):
    tk.Label(root, text=label, anchor="w", width=25).grid(row=row, column=0, padx=10, pady=5, sticky="w")
    if widget_type == "entry":
        ent = tk.Entry(root, width=25)
        ent.grid(row=row, column=1, padx=10, pady=5)
        fields[label] = ent
    elif widget_type == "combo":
        cb = ttk.Combobox(root, values=values, width=22, state="readonly")
        cb.current(0)
        cb.grid(row=row, column=1, padx=10, pady=5)
        fields[label] = cb

add_field("Age", 0)
add_field("Gender", 1, "combo", ["--Select Gender--", "Male (0)", "Female (1)"])
add_field("Ethnicity", 2, "combo", ["--Select Ethnicity--", "Asian (0)", "Black (1)", "Hispanic (2)", "White (3)"])
add_field("Parental Education", 3, "combo", ["--Select Parental Education--",
                                              "Some High School (0)", "High School Graduate (1)", 
                                              "Some College (2)", "College Graduate (3)", "Postgraduate (4)"])
add_field("Study Time Weekly (hours)", 4)
add_field("Absences (number of days)", 5)
add_field("Tutoring", 6, "combo", ["--Select Tutoring--", "No (0)", "Yes (1)"])
add_field("Parental Support", 7, "combo", ["--Select Parental Support--", "None (0)", "Low (1)", "Moderate (2)", "High (3)", "Very High (4)"])
add_field("Extracurricular Activities", 8, "combo", ["--Select Extracurricular--", "No (0)", "Yes (1)"])
add_field("Sports Participation", 9, "combo", ["--Select Sports--", "No (0)", "Yes (1)"])
add_field("Music Participation", 10, "combo", ["--Select Music--", "No (0)", "Yes (1)"])
add_field("Volunteering", 11, "combo", ["--Select Volunteering--", "No (0)", "Yes (1)"])

def combo_to_int(combo_value):
    if "(" in combo_value and ")" in combo_value:
        return int(combo_value.split("(")[1].replace(")", ""))
    else:
        raise ValueError("Please select a valid option for all dropdowns.")

def manual_predict():
    try:
        column_names = [
            "Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly",
            "Absences", "Tutoring", "ParentalSupport", "Extracurricular",
            "Sports", "Music", "Volunteering"
        ]
        vals = [
            float(fields["Age"].get()),
            combo_to_int(fields["Gender"].get()),
            combo_to_int(fields["Ethnicity"].get()),
            combo_to_int(fields["Parental Education"].get()),
            float(fields["Study Time Weekly (hours)"].get()),
            float(fields["Absences (number of days)"].get()),
            combo_to_int(fields["Tutoring"].get()),
            combo_to_int(fields["Parental Support"].get()),
            combo_to_int(fields["Extracurricular Activities"].get()),
            combo_to_int(fields["Sports Participation"].get()),
            combo_to_int(fields["Music Participation"].get()),
            combo_to_int(fields["Volunteering"].get())
        ]
        df = pd.DataFrame([vals], columns=column_names)
        gpa_pred = regression_model.predict(df)[0]
        grade_pred = classification_model.predict(df)[0]
        grade_letter = grade_labels.get(grade_pred, str(grade_pred))
        predicted_gpas.append(gpa_pred)
        predicted_grades.append(grade_letter)
        messagebox.showinfo("Prediction Result",
                            f"Predicted GPA: {gpa_pred:.2f}\nPredicted Grade: {grade_letter}")
    except ValueError as ve:
        messagebox.showerror("Input Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {str(e)}")

feature_options = {
    "Age": (17, 22),
    "Gender": [0, 1],
    "Ethnicity": [0, 1, 2, 3],
    "ParentalEducation": [0, 1, 2, 3, 4],
    "StudyTimeWeekly": (0, 40),
    "Absences": (0, 20),
    "Tutoring": [0, 1],
    "ParentalSupport": [0, 1, 2, 3, 4],
    "Extracurricular": [0, 1],
    "Sports": [0, 1],
    "Music": [0, 1],
    "Volunteering": [0, 1]
}

def generate_random_student():
    return {
        "Age": np.random.randint(*feature_options["Age"]),
        "Gender": np.random.choice(feature_options["Gender"]),
        "Ethnicity": np.random.choice(feature_options["Ethnicity"]),
        "ParentalEducation": np.random.choice(feature_options["ParentalEducation"]),
        "StudyTimeWeekly": np.random.uniform(*feature_options["StudyTimeWeekly"]),
        "Absences": np.random.randint(*feature_options["Absences"]),
        "Tutoring": np.random.choice(feature_options["Tutoring"]),
        "ParentalSupport": np.random.choice(feature_options["ParentalSupport"]),
        "Extracurricular": np.random.choice(feature_options["Extracurricular"]),
        "Sports": np.random.choice(feature_options["Sports"]),
        "Music": np.random.choice(feature_options["Music"]),
        "Volunteering": np.random.choice(feature_options["Volunteering"])
    }

def auto_predict(n=250):
    global predicted_gpas, predicted_grades
    students = [generate_random_student() for _ in range(n)]
    df_students = pd.DataFrame(students)
    df_students["PredictedGPA"] = regression_model.predict(df_students)
    df_students["PredictedGradeClass"] = classification_model.predict(df_students)
    df_students["PredictedGrade"] = df_students["PredictedGradeClass"].map(grade_labels)
    predicted_gpas.extend(df_students["PredictedGPA"].tolist())
    predicted_grades.extend(df_students["PredictedGrade"].tolist())
    messagebox.showinfo("Success", f"{n} predictions generated and added successfully!")

def visualize():
    if not predicted_gpas:
        messagebox.showwarning("No Data", "Make predictions first.")
        return
    plt.style.use("ggplot")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].hist(predicted_gpas, bins=6, color='skyblue', edgecolor='black', alpha=0.7)
    avg_gpa = np.mean(predicted_gpas)
    med_gpa = np.median(predicted_gpas)
    axs[0].axvline(avg_gpa, color='red', linestyle='dashed', linewidth=2, label=f"Mean = {avg_gpa:.2f}")
    axs[0].axvline(med_gpa, color='green', linestyle='dashed', linewidth=2, label=f"Median = {med_gpa:.2f}")
    axs[0].set_title("Predicted GPA Distribution")
    axs[0].set_xlabel("GPA")
    axs[0].set_ylabel("Count")
    axs[0].legend()
    grade_order = list(grade_labels.values())
    grade_numeric = [grade_order.index(g) for g in predicted_grades]
    axs[1].scatter(predicted_gpas, grade_numeric, c=grade_numeric, cmap="viridis", s=100, edgecolor='k', alpha=0.7)
    if len(predicted_gpas) > 1:
        z = np.polyfit(predicted_gpas, grade_numeric, 1)
        p = np.poly1d(z)
        axs[1].plot(predicted_gpas, p(predicted_gpas), "r--", label="Trend Line")
    axs[1].set_yticks(range(len(grade_order)))
    axs[1].set_yticklabels(grade_order)
    axs[1].set_title("Predicted GPA vs Grade")
    axs[1].set_xlabel("GPA")
    axs[1].set_ylabel("Grade")
    axs[1].legend()
    grade_counts = Counter(predicted_grades)
    grades = list(grade_counts.keys())
    counts = list(grade_counts.values())
    axs[2].pie(counts, labels=grades, autopct="%1.1f%%", startangle=90, colors=plt.cm.tab10.colors)
    axs[2].set_title("Predicted Grade Distribution (%)")
    plt.tight_layout()
    plt.show()

tk.Button(root, text="Predict Manually", command=manual_predict, width=30, bg="green", fg="white").grid(row=13, column=0, columnspan=2, pady=10)
tk.Button(root, text="Generate 250 Predictions Automatically", command=auto_predict, width=40, bg="blue", fg="white").grid(row=14, column=0, columnspan=2, pady=10)
tk.Button(root, text="Visualize Predictions", command=visualize, width=30, bg="purple", fg="white").grid(row=15, column=0, columnspan=2, pady=10)

root.mainloop()
