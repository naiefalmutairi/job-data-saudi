# الخطوه الاولى
# تنظيف وتحضير البيانات باستخدام Pandas

import pandas as pd

# قراءة ملف Excel
df = pd.read_excel(r'C:\Users\DELL\Desktop\Power BI\وظائف_السعودية\وظائف_السعودية.xlsx')  # تأكد من اسم الملف الصحيح

# عرض أول 5 صفوف
print(df.head())

# معلومات عامة
print(df.info())

# عدد الصفوف قبل التنظيف
print("عدد الصفوف قبل التنظيف:", len(df))

# عدد الصفوف المكررة
print("عدد الصفوف المكررة:", df.duplicated().sum())

# حذف المكررات (إن وُجدت)
df = df.drop_duplicates()

# القيم المفقودة
print("القيم المفقودة:\n", df.isnull().sum())

# حذف الصفوف التي تحتوي على بيانات ناقصة (اختياري)
df = df.dropna()

# إعادة ضبط الفهرسة
df = df.reset_index(drop=True)

# عدد الصفوف بعد التنظيف
print("عدد الصفوف بعد التنظيف:", len(df))

# حفظ نسخة نظيفة من البيانات بصيغة CSV أو Excel
df.to_excel('وظائف_السعودية_منظفة.xlsx', index=False)  # أو df.to_csv('clean_jobs.csv')

##########################################################################################
print('#' * 50)

# الخطوة الثانية
# تحليل البيانات واستكشافها باستخدام Pandas

print(df.columns.tolist())

# إزالة الفراغات من أسماء الأعمدة
df.columns = df.columns.str.strip()

# أكثر المناطق طلبًا للوظائف
print(df['المنطقة'].value_counts())

# أكثر المهن طلبًا
print(df['المسمى الوظيفي'].value_counts().head(10))

# عدد الوظائف حسب القطاع
print(df.groupby('القطاع')['عدد الوظائف'].sum())


##########################################################################################
print('#' * 50)

#الخطوة الثالثة
#  التصور البياني باستخدام Matplotlib / Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display

# إعادة تشكيل النصوص العربية
def reshape_ar(text):
    return get_display(arabic_reshaper.reshape(text))

# إعادة تشكيل أسماء المناطق
df['المنطقة_معدلة'] = df['المنطقة'].apply(reshape_ar)

# إعداد الرسم البياني
plt.figure(figsize=(10, 6))
sns.barplot(x='المنطقة_معدلة', y='عدد الوظائف', data=df, estimator=sum)

# عناوين الرسم والمحاور بالعربي مع إصلاح اتجاه الحروف
plt.title(reshape_ar("عدد الوظائف حسب المنطقة"))
plt.xlabel(reshape_ar("المنطقة"))
plt.ylabel(reshape_ar("عدد الوظائف"))

plt.rcParams['font.family'] = 'Arial'  # أو 'Tahoma' أو أي خط يدعم العربية

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

##########################################################################################
print('#' * 50)

# الخطوه الرابعة
# نموذج ذكاء اصطناعي بسيط – توقع عدد الوظائف

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ترميز الأعمدة النصية
le1 = LabelEncoder()
le2 = LabelEncoder()
df['المنطقة_encoded'] = le1.fit_transform(df['المنطقة'])
df['القطاع_encoded'] = le2.fit_transform(df['القطاع'])

# اختيار السمات
X = df[['المنطقة_encoded', 'القطاع_encoded']]
y = df['عدد الوظائف']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = LinearRegression()
model.fit(X_train, y_train)

# اختبار النموذج
score = model.score(X_test, y_test)
print(f"دقة النموذج: {score:.2f}")
######################################################## E N D ########################################################