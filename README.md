# Generative Models Project

This project implements generative models for both image and text generation. The following models are included:

- **Deep Convolutional Neural Network (CNN)**: A model designed for generating images.
- **Transformer**: A model designed for generating text sequences.
- **Diffusion Model**: A model for generative tasks that can be applied to various data types.

## Project Structure

The project is organized as follows:

```
generative-models-project
├── src
│   ├── models
│   │   ├── cnn
│   │   │   └── deep_cnn.py
│   │   ├── transformer
│   │   │   └── transformer_model.py
│   │   └── diffusion
│   │       └── diffusion_model.py
│   ├── data
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── training
│   │   ├── train.py
│   │   └── trainer.py
│   ├── utils
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── inference
│       └── generate.py
├── notebooks
│   ├── model_exploration.ipynb
│   └── results_analysis.ipynb
├── configs
│   ├── cnn_config.yaml
│   ├── transformer_config.yaml
│   └── diffusion_config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd generative-models-project
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Use the `data_loader.py` and `preprocessing.py` scripts to load and preprocess your dataset.
2. **Model Training**: Run the `train.py` script to train the models. You can configure the training parameters in the respective YAML configuration files located in the `configs` directory.
3. **Generating Outputs**: After training, use the `generate.py` script in the `inference` directory to generate new samples from the trained models.
4. **Exploration and Analysis**: Utilize the Jupyter notebooks in the `notebooks` directory for model exploration and results analysis.

```bash
python scripts/evaluate_gan.py --model_path results/cnn_20250408_205105/cnn_final
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.