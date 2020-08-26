<p>Identification of malignant from normal cells from microscopic images is very difficult because morphologically both types of cells appear similar. They are also very expensive and not widely available. Usually cancer cells are detected in advanced stages using microscopic images because of the medical expertise as those malignance cells are present in a much greater number as compared to normal people. Therefore, it is very important to detect those cells at an early stage for better cure and improving the survival of the subject. This is where cell classification via image processing comes into play to provide a solution which can be deployed easily at lesser cost.</p>
<p>Image classification models are proving to be very helpful in cancer cell detection problems. However, imaging processing models require very large datasets in order to obtain good results, which means expensive computation is required. This is where Apache Sparks comes into play. As a framework for distributed computation, Spark will allow us to build deep learning models which are very computationally heavy. There are many ways to do Deep learning with Apache Spark. These methods go from distributed DL with Keras and Pyspark, to Tensorflow on Spark or bigDL. In this project we will discuss the different methods that exist nowadays to work with deep learning models in Spark and we will implement some of these methods to see how they vary from each other.</p>
<p>For this project we will store the data in an Amazon Web Series S3 bucket and we will use Spark in two different platforms, one will be in an EMR cluster and the second will be using Amazon Sagemaker.</p>

<h1>Description of Experiments</h1>
<h3>Objectives</h3>
<p>The objective of this project is to classify leukemic B-lymphoblast cells (cancer cells) from normal B-lymphoid precursors (healthy cells). In order to do so, we will build a deep learning classifier model.</p>
<p>As we know, deep learning problems require a high computation power. Thus, the key of this project is to develop a deep learning image classifier in a distributed system. In order to do so, two approaches will be taken:
<ol>
<li>Building a deep learning model in Amazon SageMaker: We will build the image classifier in SageMaker using a Spark instance in the notebook. The model will be built using transfer learning and then it will be optimized for our data.</li>
<li>Build a deep learning model in a EMR Notebook with a Spark instance using a deep learning framework: We will build an image classifier to identify if a patient has cancer and needs treatment or if he is healthy using Tensorflow on a Spark instance created in an EMR notebook.</li>
</ol>
<p>In order to validate the results of our project we will test our models on unseen data and calculate the F1 score as our metric.
<h3>Data Set Description</h3>
<p>The dataset that will be used for this project was collected from a CodaLab competition for classification of leukemic cells from normal cells in microscopic images.</p>
<p>The dataset contains images of leukemic B-lymphoblast cells (malignant cells) and normal B-lymphoid cells. The data set has been preprocessed, as cells have been normalized and segmented from the original images. The images have a final size of roughly 300x300 pixels.</p>
<p>The data is divided in two folders, one for training the model and one for testing. The complete data set is composed of images from 118 patients. In each folder there are cell images from each patient.</p>
<p>All the images names follow the following standard naming convention:</p>
<ul>
<li>UID_P: P is the subject ID</li>
<li>UID_P_N: N is the number of image</li>
<li>UID_P_N_C:C represents the cell count</li>
<li>UID_P_N_diagnosis:all means cancer cell, hem means healthy cell.</li>
</ul>
<p>Training Test set:</p>
<p>The dataset contains a total of 73 patients of which 47 have cancer and 26 are healthy. The separation of images in training and testing will be done by patients instead of by images of malignant and healthy cells. By doing so, we will not mix images from the same patient in the training and testing.</p>
