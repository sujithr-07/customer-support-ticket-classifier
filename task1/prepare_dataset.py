
import pandas as pd
from sklearn.model_selection import train_test_split

categories = [
    "Technical Support",
    "Billing",
    "Account Access",
    "Shipping",
    "Product Inquiry"
]

priorities = ["Low", "Medium", "High", "Critical"]

problems = {
    "Technical Support": [
        "application crashes during startup",
        "software is running slowly",
        "unable to connect to the server",
        "feature is not working correctly",
        "receiving an unexpected system error"
    ],
    "Billing": [
        "incorrect amount charged",
        "invoice has missing details",
        "payment was not processed",
        "refund has not been received",
        "duplicate payment detected"
    ],
    "Account Access": [
        "unable to reset password",
        "account login is failing",
        "account has been locked",
        "verification code is not received",
        "unable to update account details"
    ],
    "Shipping": [
        "order delivery is delayed",
        "package has not arrived",
        "tracking information is incorrect",
        "wrong item was delivered",
        "package arrived damaged"
    ],
    "Product Inquiry": [
        "need product specifications",
        "requesting product availability",
        "need warranty information",
        "requesting product details",
        "need assistance choosing a product"
    ]
}

# Generate 500 synthetic tickets
records = []

for category in categories:
    for priority in priorities:
        for i in range(25):
            problem = problems[category][i % 5]

            records.append({
                "text": (
                    f"customer reports that {problem}. "
                    f"please investigate support request {i + 1:03d} "
                    f"for {category.lower()}."
                ),
                "category": category,
                "priority": priority
            })

df = pd.DataFrame(records)

# Clean text
df["text"] = df["text"].str.lower().str.strip()

# Remove duplicate tickets
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# Check class balance
print("Dataset shape:", df.shape)
print("\nCategory counts:")
print(df["category"].value_counts())

print("\nPriority counts:")
print(df["priority"].value_counts())

# Stratify on the combination of both labels
strata = df["category"] + "_" + df["priority"]

# 70% training, 30% temporary
train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=strata
)

# Split remaining 30% equally into validation and test
temp_strata = temp["category"] + "_" + temp["priority"]

validation, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp_strata
)

# Save datasets
train.to_csv("train.csv", index=False)
validation.to_csv("validation.csv", index=False)
test.to_csv("test.csv", index=False)

print("\nSplit sizes:")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))
print("Total:", len(train) + len(validation) + len(test))
