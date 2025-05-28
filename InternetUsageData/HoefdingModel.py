import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_internet_usage.csv")

# Print basic structure
print("🔍 Dataset Info:")
print(df.info())
print("\n🧾 Summary Statistics:")
print(df.describe())

# Check for missing values
print("\n🧹 Missing Values:")
print(df.isnull().sum())

# Plot 1: Line plot of Internet usage over time (if time info exists)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    plt.figure(figsize=(10, 4))
    for col in df.columns:
        if col != 'Date':
            plt.plot(df['Date'], df[col], label=col)
    plt.title("Internet Usage Over Time")
    plt.xlabel("Date")
    plt.ylabel("Usage")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot 2: Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
