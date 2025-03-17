# power-grid-time-series-anomaly-detection
Detecting Cyber Attacks on a Power Grid with ML

### Description

It is well known that viruses, command injections, denial of service (DOS), and other types of cyberattacks are common threats in today’s highly technologized and connected society. Threats to managed utilities are of special concern. 

This project explores the challenge of detecting and preventing attacks on power systems. Logistic regression, random forest, and LSTM predictive capacities are compared using attacks recorded on a simulated power grid.

### Data Source & Definitions

The dataset is provided by a research team at Mississippi State University and Oak Ridge National Laboratories with attacks recorded on a three-bus, two-line power system in a lab environment. Data comprise approximately 128 features and 78,000 rows of system measurements labeled for 37 different attack and non-attack scenarios. Waveforms were computed from synchrophasors, which became the primary training features. 

### Method

The attack classification problem was structured as a series of experiments having increasing complexity. 
Each classifier was trained and assessed starting with attacks/non-attacks as the target prediction, gradually building up to the fully detailed event labels in the original dataset.

Target variable – increasing class complexity
* binary classes: attack/no attack
* tertiary classes: attack/natural event/no event
* scenario broad class: k = 7
* full class: k = 37

18 features were selected through PCA and were also incorporated additively, trialing in three feature sets:
* One feature: R1 Power Wave
* 9 features: All Waves from R1
* 18 features: All Waves from R1 and R2

Each model's predictions were assessed using accuracy and F1. 

### Results

No model met the performance criteria of this project. A large class imbalance impacted all experiments, but each model also had individual weaknesses that impacted their accuracy.

* Logistic regression and random forest do not have many native capacities for dealing with sequenced data, training on one row at a time. 
* Logistic regression expects independent variables, instead power features are highly cross-correlated.
* Random forest had to find splits and predict samples based on nearly identical class distributions. 
* For LSTM, after samples were reshaped into tensors, there was not enough data for the model to train. 

Since no model met the performance criteria of this project, each of these findings present opportunities for future work:

1. Balance the target classes - through a combination of data augmentation and undersampling 
2. Trial windowed/lagged features to capture more timeseries variance
3. Trial models that may require less data - such as Naive Bayes, KNN, and sequential pattern mining.

### Usage

1. Download data from https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets to a subfolder, data/raw/
2. Run load_data.py
3. Explore the data - EDA found under notebooks/
4. Run a model - model notebooks found under models/
    
### References

Beaver, Justin M., Borges-Hink, Raymond C., Buckner, Mark A., "An Evaluation of Machine Learning Methods to Detect Malicious SCADA Communications," in the Proceedings of 2013 12th International Conference on Machine Learning and Applications (ICMLA), vol.2, pp.54-59, 2013. doi: 10.1109/ICMLA.2013.105 

Morris, T. (2021, April 20). Industrial Control System (ICS) cyber attack datasets. Tommy Morris, Ph.D. University of Alabama at Huntsville. https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets 
