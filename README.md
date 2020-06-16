# Low-T model

This is an app for finding the power law dependency of low temperature physical properties. This properties includes the electrical resistivity in metals and specific heat of all materials.

### Principle

The brief introduction of underlying physics of these properties can be found in these pages:

1. **[Electrical resistivity and conductivity from wiki](https://en.wikipedia.org/wiki/Electrical_resistivity_and_conductivity)**

2. **[Electrical Conduction in Metals and Semiconductors](https://link.springer.com/chapter/10.1007/978-3-319-48933-9_2)**

3. **[Debye model](https://en.wikipedia.org/wiki/Debye_model)**

4. For electronic and magnetic contributions to specific heat, more detailed discussions are elaborated in solid state physics textbooks.

### Model

* The single parameter model y = y<sub>0</sub> + Ax<sup>p</sup> is implemented by simply performing linear regression based on (x<sup>n</sup>, y) pairs. 

* The double parameter model y = y<sub>0</sub> + A<sub>1</sub>x<sup>p<sub>1</sub></sup> + A<sub>2</sub>x<sup>p<sub>2</sub></sup> uses (x<sup>p<sub>1</sub></sup>, x<sup>p<sub>2</sub></sup>, y) for fitting the model.

The mean squared error (MSE) is used by default to evaluate the performance of the model. The MSE as a function of power exponent provided in the app helps identify the best model. In the double-parameter model, this is a heatmap showing distribution of MSE for different combinations of p1 and p2.

A model with more than one parameter usually displays varied power exponent for different selected ranges. It would be helpful to play with the data range before determining the value of exponents.


### How to use

1. Use pip to install all required modules.

`pip install -r requirements`

2. The web browser will automatically open the app by running in Terminal/Command line

`python app.py`

3. Upload the data file and select x- and y-columns you want to investigate. A scatter figure and A tabling listing data statistics should show up on the right panel.

4. Use the slider on the left panel to specify the range you want to build the model upon.

5. Choose the exponent range. 
    - For low temperature resistivity, the range can be from 1 to 5 for different electron scattering mechanisms. 
    - For low temperature specific heat, the range should be between 3/2 to 3 based on thermal dynamic properties of phonons, electrons and magnons.

6. Specify a metric for evaluating the model performance.

7. Select the test_size. This parameter has the same definition as to the that in the `train_test_split` from `sklearn.model_selection` module, which indicates the ratio of data points for testing the model.
