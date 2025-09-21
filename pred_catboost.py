import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ===============================
# Global configuration
# ===============================

# File paths for train and test datasets, upload the datasets as part of the repo
DATA_PATH_TRAIN = "german_credit_train.csv"
DATA_PATH_TEST = "german_credit_test.csv"

# Target column
TARGET = "Risk"

# Columns treated as numeric features
NUMERIC_COLS = ["LoanDuration", 
                "LoanAmount", 
                "CurrentResidenceDuration",
                "Age", 
                "InstallmentPercent", 
                "ExistingCreditsCount"
                ]

# Columns treated as categorical features
CATEGORICAL_COLS = [
    "CheckingStatus", 
    "CreditHistory", 
    "LoanPurpose", 
    "ExistingSavings",
    "EmploymentDuration", 
    "Sex", 
    "OthersOnLoan", 
    "OwnsProperty",
    "InstallmentPlans", 
    "Housing", 
    "Job", 
    "Dependents", 
    "Telephone", 
    "ForeignWorker"
]

# ===============================
# Data loading
# ===============================

@st.cache_data
def load_data():
    """
    Load training and test datasets.
    Caching ensures the data is not reloaded on every rerun of the Streamlit app.
    """
    train = pd.read_csv(DATA_PATH_TRAIN)
    test = pd.read_csv(DATA_PATH_TEST)
    return train, test

# ===============================
# Model training
# Although the logical workflow is usually EDA → Model → Prediction, in a Streamlit app we can train the model in the background first
# (for smoother user experience), while showing EDA first in the UI (which is more intuitive for end users).
# ===============================

@st.cache_resource
def train_model_with_split(train_df, random_state=100, test_size=0.2):
    """
    Train CatBoost on a train/validation split of the provided training dataframe.
    Returns:
      model, X_train, X_val, y_train, y_val
    Notes:
      - Categorical columns are explicitly converted to string.
      - Use column names instead of indices for cat_features to avoid index errors.
    """
    # Separate X and y
    X = train_df.drop(columns=[TARGET]).copy()
    y = train_df[TARGET].copy()

    # Split into train / validation
    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    # Convert categorical columns to string
    for c in CATEGORICAL_COLS:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype(str)
            X_val[c] = X_val[c].astype(str)

    # Train CatBoost using column names
    model = CatBoostClassifier(verbose=0, random_seed=random_state)
    model.fit(X_train, y_train, cat_features=[c for c in CATEGORICAL_COLS if c in X_train.columns])

    return model, X_train, X_val, y_train, y_val

# ===============================
# EDA: Categorical feature plots
# ===============================

def plot_categorical_risk_ratio(df, col, target=TARGET):
    """
    Plot the distribution of 'Risk' within each category of a categorical variable.
    Displays the proportion of 'Risk' vs 'No Risk' for each category.
    """
    prop = (
        df.groupby(col)[target]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=prop, x=col, y="proportion", hue=target, ax=ax)
    ax.set_title(f"Risk ratio by {col}")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ===============================
# EDA: Numeric feature plots
# ===============================

def plot_numeric_risk_ratio(df, col, bins=5, target=TARGET):
    """
    Plot the distribution of 'Risk' after binning a numeric variable.
    For each bin, show the proportion of 'Risk' vs 'No Risk'.
    """
    df["bin"] = pd.cut(df[col], bins=bins)
    prop = (
        df.groupby("bin")[target]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=prop, x="bin", y="proportion", hue=target, ax=ax)
    ax.set_title(f"Risk ratio by {col} bins")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    df.drop(columns="bin", inplace=True)

# ===============================
# Streamlit main app
# ===============================

def main():
    st.title("German Credit Risk Prediction")

    # Load datasets
    train, test = load_data()

    # Create tabs for different sections of the app
    tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", 
                                "🤖 Model Results", 
                                "🔮 Prediction"])

    # ---- Tab 1: Exploratory Data Analysis ----
    with tab1:
        # ==========================
        # Categorical Features
        # ==========================
        st.markdown("### 🎭 Categorical Features (Distribution by Risk)")

        cat_cols = st.columns(2)
        for i, col in enumerate(CATEGORICAL_COLS):
            with cat_cols[i % 2]:
                count_data = train.groupby([col, TARGET]).size().reset_index(name="count")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.barplot(data=count_data, x=col, y="count", hue=TARGET, ax=ax, palette="Set2")
                ax.set_title(f"{col} vs Risk")
                ax.bar_label(ax.containers[0], fmt='%d')
                ax.bar_label(ax.containers[1], fmt='%d')
                plt.xticks(rotation=45)
                st.pyplot(fig)

        # ==========================
        # Numeric Features
        # ==========================
        st.markdown("### 🔢 Numeric Features (Distribution by Risk)")
        # Optional: let user control global number of bins (comment out if you don't want interactive control)
        bins_slider = st.slider("Number of bins for numeric histograms", min_value=5, max_value=40, value=10)

        num_cols = st.columns(2)
        for i, col in enumerate(NUMERIC_COLS):
            with num_cols[i % 2]:
                fig, ax = plt.subplots(figsize=(6, 4))

                # Prepare data and bins (use slider value; fallback to 10 if slider removed)
                values = train[col].dropna()
                try:
                    bins = np.histogram_bin_edges(values, bins=bins_slider)
                except Exception:
                    bins = np.histogram_bin_edges(values, bins=10)

                # Bin the data and compute counts per bin per class
                df_tmp = train[[col, TARGET]].dropna().copy()
                df_tmp['bin'] = pd.cut(df_tmp[col], bins=bins, include_lowest=True)
                count_data = df_tmp.groupby(['bin', TARGET]).size().reset_index(name='count')
                count_data['bin_str'] = count_data['bin'].astype(str)

                # Pivot so each column is a Risk class (works for any number of classes)
                pivot = count_data.pivot(index='bin_str', columns=TARGET, values='count').fillna(0)
                classes = pivot.columns.tolist()
                x = np.arange(len(pivot))
                k = len(classes)
                width = 0.8 / max(k, 1)  # total width 0.8 split among classes
                palette = sns.color_palette('Set2', n_colors=k)

                # Draw side-by-side bars
                totals = pivot.sum(axis=1).values  # total per bin (for percent calc)
                for j, cls in enumerate(classes):
                    vals = pivot[cls].values
                    positions = x - 0.4 + j * width + width / 2
                    bars = ax.bar(positions, vals, width=width, label=str(cls), color=palette[j], edgecolor='k', linewidth=0.3)

                    # Annotate each bar with "count\npct%"
                    for pos, v, tot in zip(positions, vals, totals):
                        if v > 0 and tot > 0:
                            pct = v / tot * 100
                            # choose text color for contrast
                            txt_color = 'white' if pct > 20 else 'black'
                            ax.text(pos, v / 2, f"{int(v)}\n{pct:.1f}%", ha='center', va='center', fontsize=9, color=txt_color)

                # x ticks and labels
                ax.set_xticks(x)
                ax.set_xticklabels(pivot.index, rotation=40, ha='right', fontsize=9)
                ax.set_title(f"{col} distribution by Risk (counts + % inside bars)")
                ax.set_ylabel("Count")
                ax.legend(title=TARGET, bbox_to_anchor=(1.02, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)

        # ==========================
        # Correlation Heatmap
        # ==========================
        st.markdown("### 🔗 Correlation Heatmap (Numeric Variables)")
        corr = train[NUMERIC_COLS].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

        # ==========================
        # Chi-square test (categorical vs Risk)
        # ==========================
        st.markdown("### 📐 Chi-square Test Results (Categorical Features vs Risk)")
        from scipy.stats import chi2_contingency

        results = []
        for col in CATEGORICAL_COLS:
            contingency = pd.crosstab(train[col], train[TARGET])
            chi2, p, _, _ = chi2_contingency(contingency)
            results.append({"Feature": col, "p-value": p})

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        sig_features = results_df[results_df["p-value"] < 0.05]["Feature"].tolist()
        if sig_features:
            st.markdown(f"✨ **Significant association found for:** {', '.join(sig_features)}")


    # ---- Tab 2: Model Results (use train/validation split only) ----
    with tab2:
        # --- Intro ---
        st.markdown("""
        ## 🧠 Model Selected: Catboost

        CatBoost is a gradient boosting algorithm that handles categorical features natively,
        reduces the need for extensive preprocessing, and provides strong performance on tabular data.  
        For more details, see the [official CatBoost sklearn API documentation](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier).
        """)

        # Train model and get the splits (cached)
        model, X_train, X_val, y_train, y_val = train_model_with_split(train)

        # ----- Training & Validation Accuracy -----
        st.subheader("✅ Model Accuracy")
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)

        acc_df = pd.DataFrame({
            "Dataset": ["Train", "Validation"],
            "Accuracy": [train_acc, val_acc]
        })
        st.dataframe(acc_df.style.format({"Accuracy": "{:.3f}"}).background_gradient(cmap="Blues"))

        # ----- Confusion Matrix -----
        from sklearn.metrics import confusion_matrix
        st.subheader("📊 Validation Confusion Matrix")

        cm = confusion_matrix(y_val, val_preds, labels=["No Risk", "Risk"])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual No Risk", "Actual Risk"],
            columns=["Predicted No Risk", "Predicted Risk"]
        )

        st.dataframe(cm_df.style.background_gradient(cmap="Blues", axis=None))

        # ----- Classification Report -----
        from sklearn.metrics import classification_report
        st.subheader("📑 Validation Classification Report")

        report_dict = classification_report(
            y_val, val_preds,
            target_names=["No Risk", "Risk"],
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap="Greens", axis=0))

        # ----- Feature Importance -----
        st.subheader("🔥 Feature Importance (Top 15)")

        fi_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.get_feature_importance()
        }).sort_values("importance", ascending=False).head(15)

        st.dataframe(fi_df.style.background_gradient(cmap="Oranges", axis=0))


    # ---- Tab 3: Prediction ----
    with tab3:
        st.subheader("Custom / Test ID Prediction")
        st.markdown("""
        In this section, you can try two ways of generating predictions with the trained CatBoost model:

        1. **📝 Custom Input** – Manually enter feature values (numeric and categorical) to see the predicted risk.  
        2. **🆔 Test Set ID** – Select a record from the test dataset by its ID to view the prediction along with its feature values.  

        This allows both interactive "what-if" analysis with your own inputs, and quick evaluation on real test data.
        """)

        # Select mode
        mode = st.radio("Select prediction mode:", ["📝 Custom Input", "🆔 Test Set ID"])

        # Train the model (cached)
        model, X_train, X_val, y_train, y_val = train_model_with_split(train)

        if mode == "📝 Custom Input":
            input_data = {}

            # --- Numeric features (2 per row) ---
            st.markdown("### 🔢 Numeric Features")
            for i in range(0, len(NUMERIC_COLS), 2):
                cols = st.columns(2)
                for j, col in enumerate(NUMERIC_COLS[i:i+2]):
                    with cols[j]:
                        val = st.number_input(f"{col}", value=float(train[col].median()))
                        input_data[col] = val

            # --- Categorical features (3 per row) ---
            st.markdown("### 🏷️ Categorical Features")
            for i in range(0, len(CATEGORICAL_COLS), 3):
                cols = st.columns(3)
                for j, col in enumerate(CATEGORICAL_COLS[i:i+3]):
                    with cols[j]:
                        train_unique = sorted(train[col].unique())
                        train_unique_str = [str(v) for v in train_unique]
                        val = st.selectbox(f"{col}", options=train_unique_str)
                        input_data[col] = val

            # --- Build input DataFrame ---
            input_df = pd.DataFrame([input_data])
            for c in CATEGORICAL_COLS:
                input_df[c] = input_df[c].astype(str)
            input_df = input_df[X_train.columns]

            pool = Pool(data=input_df, cat_features=CATEGORICAL_COLS)
            pred = model.predict(pool)[0]

            st.markdown(f"### 🎯 Predicted Risk: **{pred}**")
            st.markdown("#### 📋 Input Data")
            st.dataframe(input_df)

        else:  # 🆔 Test Set ID mode
            test_ids = test['Id'].tolist()
            selected_id = st.selectbox("Select Test ID", options=test_ids)
            input_df = test[test['Id'] == selected_id].drop(columns=['Id'])

            for c in CATEGORICAL_COLS:
                input_df[c] = input_df[c].astype(str)
            input_df = input_df[X_train.columns]

            pool = Pool(data=input_df, cat_features=CATEGORICAL_COLS)
            pred = model.predict(pool)[0]

            st.markdown(f"### 🎯 Predicted Risk for ID {selected_id}: **{pred}**")
            st.markdown("#### 📋 Input Data")
            st.dataframe(input_df)



# Run the app
if __name__ == "__main__":
    main()
