# garbage_classification

**1. Install:**
   - VS Code
   - Python extensions
   - Git extension(usually built-in)
   - Jupyter extension - to still run notebooks when needed (Optional)

**2. Create 'Garbage Identification' project folder**

You can take reference from the folder sketch below:

```text
garbage-classifier/
├── data/                     - Dataset folder (training, validation, test images)
├── src/                      - Source code for model training and evaluation
│   ├── train.py              - Script to train the garbage classification model
│   ├── evaluate.py           - Script to test/evaluate trained model performance
│   └── utils.py              - Helper functions (data loading, metrics, etc.)
├── models/                   - Saved model weights or checkpoints
├── notebooks/                - Jupyter notebooks for experimentation
│   └── exploration.ipynb     - Initial data and model exploration notebook
├── requirements.txt          - Python dependencies list
├── README.md                 - Project overview and usage instructions
└── .gitignore                - Files/folders to ignore in Git tracking
```

**3. Create and activate a Python Environment**

In the VS Code terminal, write below code:

   ```text
   python -m venv venv        # create a clean virtual environment)
   
   #activate it...
    venv\Scripts\activate     # on Windows:

    source venv/bin/activate  # on macOS/Linux:
   ```

**4. Install dependencies**

in terminal:

   pip install torch torchvision torchaudio matplotlib numpy pandas scikit-learn tqdm pillow

**5. Initialize Git & Push to GitHub**

in terminal:

    git init
    git branch -M main
    git remote add origin https://github.com/<your-username>/garbage-classifier.git
    git add .
    git commit -m "v1: setup and base training script"
    git push -u origin main


