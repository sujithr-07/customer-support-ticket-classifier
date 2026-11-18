
import pandas as pd
import random
from sklearn.model_selection import train_test_split

random.seed(42)

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

priority_messages = {
    "Low": "please review this when convenient",
    "Medium": "please assist with this issue",
    "High": "this issue is urgent and needs attention",
    "Critical": "service is unavailable and immediate action is required"
}

records = []

ticket_id = 1

for category in categories:
    for priority in priorities:
        for i in range(25):

            problem = random.choice(problems[category])

            text = (
                f"customer reports {problem}. "
                f"{priority_messages[priority]}. "
                f"additional ticket reference {ticket_id}."
            )

            records.append({
                "text": text,
                "category": category,
                "priority": priority
            })

            ticket_id += 1

df = pd.DataFrame(records)

# Clean text
df["text"] = df["text"].str.lower().str.strip()

# Remove duplicates
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

# Check balance
print("Dataset shape:", df.shape)
print("\nCategory counts:")
print(df["category"].value_counts())

print("\nPriority counts:")
print(df["priority"].value_counts())

# Stratify on category and priority combination
df["strata"] = df["category"] + "_" + df["priority"]

train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["strata"]
)

validation, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["strata"]
)

# Remove helper column
for dataset in [train, validation, test]:
    dataset.drop(columns=["strata"], inplace=True)

train.to_csv("train.csv", index=False)
validation.to_csv("validation.csv", index=False)
test.to_csv("test.csv", index=False)

print("\nDataset split completed!")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))
