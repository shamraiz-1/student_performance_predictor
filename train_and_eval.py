import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_excel("Student_Score_prediction.xlsx")

target_reg = "GPA"
target_cls = "GradeClass"

features = [
    "Age","Gender","Ethnicity","ParentalEducation","StudyTimeWeekly",
    "Absences","Tutoring","ParentalSupport","Extracurricular",
    "Sports","Music","Volunteering"
]

X = df[features]
y_reg = df[target_reg]
y_cls = df[target_cls]

Xtr, Xte, ytr, yte = train_test_split(X, y_reg, test_size=0.2, random_state=42)
Xtr_cls, Xte_cls, ytr_cls, yte_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)

cats = ["Gender","Ethnicity","ParentalEducation","Tutoring","ParentalSupport","Extracurricular","Sports","Music","Volunteering"]
nums = ["Age","StudyTimeWeekly","Absences"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ("num", StandardScaler(), nums)
])

reg_pipe = Pipeline([
    ("pre", preprocessor),
    ("reg", LinearRegression())
])
reg_pipe.fit(Xtr, ytr)
yp = reg_pipe.predict(Xte)
mae = mean_absolute_error(yte, yp)
rmse = np.sqrt(mean_squared_error(yte, yp))
r2 = r2_score(yte, yp)
print(f"[Linear Regression] MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")

poly_pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ("num", Pipeline([
        ("sc", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False))
    ]), nums)
])

reg_pipe_poly = Pipeline([
    ("pre", poly_pre),
    ("reg", LinearRegression())
])
reg_pipe_poly.fit(Xtr, ytr)
yp_poly = reg_pipe_poly.predict(Xte)
mae = mean_absolute_error(yte, yp_poly)
rmse = np.sqrt(mean_squared_error(yte, yp_poly))
r2 = r2_score(yte, yp_poly)
print(f"[Polynomial Regression] MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")

alt_features = [f for f in features if f not in ["Sports","Music"]]
Xtr_alt, Xte_alt = Xtr[alt_features], Xte[alt_features]
ytr_alt, yte_alt = ytr, yte

cats_alt = [c for c in cats if c not in ["Sports","Music"]]
nums_alt = nums

pre_alt = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cats_alt),
    ("num", StandardScaler(), nums_alt)
])

reg_pipe_alt = Pipeline([
    ("pre", pre_alt),
    ("reg", LinearRegression())
])
reg_pipe_alt.fit(Xtr_alt, ytr_alt)
yp_alt = reg_pipe_alt.predict(Xte_alt)
mae = mean_absolute_error(yte_alt, yp_alt)
rmse = np.sqrt(mean_squared_error(yte_alt, yp_alt))
r2 = r2_score(yte_alt, yp_alt)
print(f"[Alt Features - dropped ['Music','Sports']] MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}")

cls_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
cls_pipe.fit(Xtr_cls, ytr_cls)
yp_cls = cls_pipe.predict(Xte_cls)
acc = accuracy_score(yte_cls, yp_cls)
macro_f1 = f1_score(yte_cls, yp_cls, average="macro")
print(f"[Classification] ACC={acc:.3f} MacroF1={macro_f1:.3f}")
print("Labels:", cls_pipe.classes_)
print("Confusion Matrix:\n", confusion_matrix(yte_cls, yp_cls))

joblib.dump(reg_pipe, "gpa_regression.joblib")
joblib.dump(cls_pipe, "gradeclass_classifier.joblib")
