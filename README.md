# Deep Learning - Tensor Flow Potato Leaf Images Disease Classification Detection
![https://github.com/RickyDoan/DL-TF-Potato-Leaf-Images-Classification-Prediction/blob/main/leaf-potato-short-20s.gif]()

This project aims to detect diseases in tomato leaves using a deep learning model. It utilizes a Convolutional Neural Network (CNN) trained on a dataset of tomato leaf images with various diseases.

## Dataset

The dataset used for this project is the PlantVillage dataset, which contains images of tomato leaves with different diseases. The dataset is preprocessed to resize the images and split them into training and validation sets.

## Model

A CNN is used as the model for this project. The model is trained on the training dataset and validated on the validation dataset. The model architecture includes convolutional layers, max-pooling layers, a flatten layer, and dense layers.

## Training

The model is trained using the `model.fit()` function in TensorFlow. The optimizer used is Adam, and the loss function is SparseCategoricalCrossentropy. The model is trained for a certain number of epochs.

## Streamlit App

A Streamlit app is created to allow users to upload images of tomato leaves and get predictions. The app uses the trained model to make predictions and displays the results.

## Conclusion

This project demonstrates the use of deep learning for tomato leaf disease detection. The model achieves good accuracy in classifying diseases. The Streamlit app provides a user-friendly interface for making predictions.

## Usage

To use the Streamlit app, follow these steps:

1. Install the required packages: `pip install streamlit tensorflow joblib pillow`
2. Download the trained model from [link to your model file].
3. Run the Streamlit app: `streamlit run your_app.py`
4. Upload an image of a tomato leaf.
5. View the predicted disease and confidence.

## Future Work

- Improve the model accuracy by using a larger dataset or a different model architecture.
- Add more features to the Streamlit app, such as displaying the probability of each disease.
- Deploy the app to a cloud platform for wider access.
