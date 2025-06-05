import pandas as pd
train_df = pd.read_csv('/Users/rithviks/Desktop/TIM147/FPA_FOD_Cleaned.csv')  # Replace with your actual path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ==== STEP 1: Define Features and Targets ====
categorical_cols = ["OWNER_DESCR", "STATE", "Des_Tp", "EVT", "NPL", "FRG"]
X = train_df[["DISCOVERY_DOY","FIRE_SIZE","LATITUDE","LONGITUDE","OWNER_DESCR","STATE","Des_Tp","EVT","rpms","pr_Normal","tmmn_Normal","tmmx_Normal","sph_Normal","srad_Normal","fm100_Normal","fm1000_Normal","bi_Normal","vpd_Normal","erc_Normal","DSF_PFS","EBF_PFS","PM25F_PFS","MHVF_PFS","LPF_PFS","NPL","RMP_PFS","TSDF_PFS","FRG","TRI_1km","Aspect_1km","Elevation_1km","Slope_1km","SDI","Annual_etr","Annual_precipitation","Annual_tempreture","Aridity_index","rmin","rmax","vs","NDVI-1day","CheatGrass","ExoticAnnualGrass","Medusahead","PoaSecunda"]]
Y = train_df[["EALR_PFS", "EBLR_PFS", "EPLR_PFS"]]

numerical_cols = [col for col in X.columns if col not in categorical_cols]

# ==== STEP 2: Preprocess ====
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)
y = Y.values

# ==== STEP 3: Split and Convert to Tensors ====
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# ==== STEP 4: DataLoader ====
batch_size = 1024
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ==== STEP 5: Model ====
class FireRegressor(nn.Module):
    def __init__(self, input_size):
        super(FireRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))  # Output in [0,1]
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FireRegressor(input_size=X_train_tensor.shape[1]).to(device)

# ==== STEP 6: Optimizer & Loss ====
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ==== STEP 7: Training Loop ====
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(50):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            val_output = model(X_val_batch)
            val_loss += loss_fn(val_output, y_val_batch).item()

    train_losses.append(total_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_fire_model.pt")

# ==== STEP 8: Load Best Model (Optional) ====
# model.load_state_dict(torch.load("best_fire_model.pt"))

# Create visualizations
import matplotlib.pyplot as plt

# ==== Evaluate on validation set ====
model.eval()
with torch.no_grad():
    predictions = model(X_val_tensor.to(device)).cpu().numpy()
    actuals = y_val_tensor.numpy()

# Plot Loss Curves
# After training:
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# Graphs
output_labels = ["EALR_PFS", "EBLR_PFS", "EPLR_PFS"]
from sklearn.metrics import r2_score

for i in range(3):
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals[:, i], predictions[:, i], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # perfect prediction line
    plt.title(f'{output_labels[i]}: Predicted vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    r2 = r2_score(actuals[:, i], predictions[:, i])
    plt.text(0.05, 0.9, f"R² = {r2:.2f}", transform=plt.gca().transAxes)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

sample_range = 50  # number of samples to plot

for i in range(3):
    plt.figure(figsize=(10, 4))
    plt.plot(actuals[:sample_range, i], label='Actual', marker='o')
    plt.plot(predictions[:sample_range, i], label='Predicted', marker='x')
    plt.title(f'{output_labels[i]} - Sample of {sample_range}')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for i in range(3):
    residuals = actuals[:, i] - predictions[:, i]
    plt.figure(figsize=(10, 4))
    plt.hist(residuals, bins=40, alpha=0.7)
    plt.title(f'Residuals Histogram for {output_labels[i]}')
    plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==== Print Predictions vs Actuals ====
output_labels = ["EALR_PFS", "EBLR_PFS", "EPLR_PFS"]

print(f"{'Sample':<8} | {'Actual':<45} | {'Predicted'}")
print("-" * 80)

for i in range(10):  # First 10 samples
    actual = actuals[i]
    pred = predictions[i]
    actual_str = ", ".join(f"{val:.4f}" for val in actual)
    pred_str = ", ".join(f"{val:.4f}" for val in pred)
    print(f"{i:<8} | {actual_str:<45} | {pred_str}")
