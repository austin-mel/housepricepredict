# House Price Prediction

## Project Question

Can models predict house sale price from property details and listing-style images?

## Purpose

The project estimates house sale price from the data available in the notebook. It supports early pricing research and comparison of modeling approaches. It does not provide appraisals, lending decisions, tax assessments, insurance decisions, legal advice, or investment advice.

## Results Summary

The strongest notebook result came from the model that used both property details and VGG16 image features. It reported validation RMSE of $124,028. In practical terms, the model can find pricing patterns in this dataset, but its dollar estimates can still miss by six figures.

The property-details-only model also performed well. It reported $137,723 RMSE on the test split. The image-only CNN performed poorly at $456,119 validation RMSE. In this small dataset, photos alone did not predict sale price reliably.

These results support model comparison and early research. They do not support final pricing, appraisal, lending, tax, insurance, legal, or investment decisions.

## Dataset

Source: [Houses Dataset](https://github.com/emanhamed/Houses-dataset)

The notebook clones the dataset from GitHub when the local `Houses-dataset` folder does not exist.

### Inputs

- Property details: bedroom count, bathroom count, area, and ZIP code.
- Images: bathroom, bedroom, front exterior, and kitchen photos.

The notebook combines the four images for each home into one 128 by 128 image.

### Target

The target variable is `Price`. The models predict sale price in dollars.

### Size and Filtering

The dataset contains 535 homes before filtering. The notebook removes homes priced below $100,000 or above $900,000. That removes 129 rows and leaves 406 homes.

The notebook uses a 75 percent training split and a 25 percent validation or test split. The resulting split contains 304 training homes and 102 validation or test homes.

### Dataset Limits

- The dataset is small for image-based modeling.
- The price filter excludes homes below $100,000 and above $900,000.
- ZIP code provides the only location signal.
- The notebook does not include sale date, lot size, property condition notes, renovation history, school district, neighborhood amenities, mortgage rates, or market conditions.
- The dataset reflects the regions and time period captured by its source files. The results may not transfer to other markets or current housing conditions.

## Metric Definitions

The notebook reports mean squared error and RMSE.

- Mean squared error: measures the average squared gap between predicted price and actual price. Squaring gives larger mistakes more weight.
- RMSE: stands for root mean squared error. It converts mean squared error back into dollars. Lower RMSE means better performance.
- Training RMSE: measures model error on the homes used to train the model.
- Validation RMSE: measures model error on held-out homes used during model comparison.
- Test RMSE: measures model error on held-out homes used for final evaluation in the tabular model section.

An RMSE near $124,000 means errors of that scale drive the validation result. It does not mean every prediction misses by exactly $124,000.

## Models

All models in the notebook predict sale price. They are regression models, not classification models. The notebook evaluates them with mean squared error and RMSE.

### Custom CNN

The custom CNN uses TensorFlow and Keras. It takes the combined 128 by 128 home image as input. The model includes three convolution blocks with batch normalization, ReLU activation, max pooling, and dropout in the first two blocks. It then uses global average pooling, a dense layer, dropout, and one linear output for price prediction.

Saved custom CNN results:

| Metric | Value |
| --- | ---: |
| Training RMSE | $453,599 |
| Validation RMSE | $456,119 |

### Tabular Neural Network

The tabular model uses the property details. The notebook normalizes bedroom and bathroom counts. It one-hot encodes area and ZIP code. The model uses dense layers with 256, 128, 64, and 32 units, dropout, and one output for price prediction.

Saved tabular result:

| Metric | Value |
| --- | ---: |
| Test RMSE | $137,723 |

### Combined CNN Plus Tabular Model

The combined model uses two inputs. One branch processes the combined image with the custom CNN structure. The other branch processes the property details with the tabular network structure. The notebook concatenates both branches, adds dense layers, and predicts one sale price.

Saved combined model result:

| Metric | Value |
| --- | ---: |
| Validation RMSE | $144,223 |

### VGG16 Transfer Learning

The VGG16 model uses ImageNet pretrained weights through TensorFlow and Keras. The notebook loads VGG16 without its top layer, freezes the copied VGG16 layers, and adds dense regression layers for sale price prediction.

Saved VGG16 result:

| Metric | Value |
| --- | ---: |
| Training RMSE | $52,982 |
| Validation RMSE | $184,235 |

The training result is much stronger than the validation result. That gap indicates overfitting in the saved notebook output.

### VGG16 Plus Tabular Model

The final model combines VGG16 image features with property details. The image branch loads VGG16 without its top layer and flattens the image features. The tabular branch uses the same property-detail model structure as the combined CNN model. The notebook concatenates both branches, adds a dense layer, and predicts one sale price.

Saved VGG16 plus tabular result:

| Metric | Value |
| --- | ---: |
| Validation RMSE | $124,028 |

This model has the strongest saved validation result in the notebook.

The notebook plots learning curves for mean squared error and loss. It also plots actual-vs-predicted regression charts for the combined models.

## Detailed Results

The notebook reports root mean squared error, or RMSE. RMSE uses dollars in this project. Lower RMSE means better performance. An RMSE near $124,000 means errors of that scale drive the validation result, with larger errors weighted more heavily.

| Model | Reported Split | RMSE |
| --- | ---: | ---: |
| Image-only CNN | Validation | $456,119 |
| Property-details-only neural network | Test | $137,723 |
| Combined CNN plus property details | Validation | $144,223 |
| VGG16 transfer-learning image model | Validation | $184,235 |
| VGG16 transfer-learning image model | Training | $52,982 |
| VGG16 plus property details | Validation | $124,028 |

The VGG16 plus property-details model has the lowest reported validation RMSE at $124,028. The property-details-only neural network reports $137,723 RMSE on the test split. The image-only CNN reports the weakest result at $456,119 validation RMSE.

The VGG16 image-only model reports $52,982 training RMSE and $184,235 validation RMSE. That gap indicates overfitting in the notebook results.

The notebook does not report MAE, R-squared, confidence intervals, or error by price band.

## Use Limits

Use this project for learning, model comparison, and rough research only. Do not use the model for appraisal, lending, insurance, tax, legal, or high-stakes housing decisions.

The model depends on a small filtered dataset. It omits many factors that affect housing prices. It may not generalize outside the dataset.

## Reproducibility

1. Open `HousePricePrediction.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Install the packages imported by the notebook if needed: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `opencv-python`.
3. Run the notebook from top to bottom.
4. Let the notebook clone the Houses Dataset if the local `Houses-dataset` folder does not exist.
5. Open `HousePricePrediction.html` to view the rendered notebook without rerunning the analysis.

Results may change across hardware, TensorFlow or Keras versions, random initialization, and GPU availability.