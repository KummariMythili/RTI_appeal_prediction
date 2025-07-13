import os

# Define the folder and file structure
structure = {
    "Data": ["your_dataset.csv", "your_dataset.xlsx", "your_dataset.json"],
    "Training": ["training_notebook.ipynb", "preprocess_data.ipynb"],
    "Evaluation": ["evaluation_and_tuning.ipynb", "best_model_saving.ipynb"],
    "App": [
        "app.py",  # Flask backend
        os.path.join("model", "model.pkl"),
        os.path.join("templates", "index.html"),
        os.path.join("static","style.css"),
        os.path.join("static", "javascript.js"),
        os.path.join("routes", "__init__.py"),  # Optional: start of modular routing
    ],
}

# Root-level files
root_files = [
    "README.md",
    "requirements.txt",
    "python_version.txt",
    "setup.exe",  # Placeholder
]

# Create folders and files
def create_project_structure(base_path="."):
    for folder, files in structure.items():
        for file in files:
            file_path = os.path.join(base_path, folder, file)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            open(file_path, "w").close()

    for file in root_files:
        open(os.path.join(base_path, file), "w").close()

    print("✅ Project folder structure generated successfully.")

# Run the function
if __name__ == "__main__":
    create_project_structure()
