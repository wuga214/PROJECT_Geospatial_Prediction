# Geospatial_Prediction

TO RUN
===
src/evaluation/CrossValidations.java //This will run Kruskal MST merging to look for the best partitioning, limited by searching space.
src/samping/Gibbs.java //Gibbs samping merging, through entire hypothesis space.

Progress
===
1. Baseline algorithms are good to work now.<br />
2. Cross-validation is now full functional.<br />
3. Gaussian processing for large data has risk to be stuck, it is better to test this algorithm under 1000 data point.In cross-validation, in default, there will 30% of total data point used to train model, therefore, the running time will be very long, please be aware of that.<br />
4. Gibbs Samping now full functional<br />

Table
===
![alt tag](https://github.com/wuga214/Geospatial_Prediction/blob/RuntimeImproved/eval.png)
![alt tag](https://github.com/wuga214/Geospatial_Prediction/blob/RuntimeImproved/realdata.png)

Comparasion
===
![alt tag](https://github.com/wuga214/Geospatial_Prediction/blob/master/Comparasion.png)

Synthetic data
===
3D Data Plot| Synthetic
------------ | -------------
![box](https://github.com/wuga214/Geospatial_Prediction/blob/master/plots/figure_1.png) | ![circles](https://github.com/wuga214/Geospatial_Prediction/blob/master/plots/figure_2.png)
![stair](https://github.com/wuga214/Geospatial_Prediction/blob/master/plots/figure_3.png) | ![cake](https://github.com/wuga214/Geospatial_Prediction/blob/master/plots/figure_4.png)

3D Results Plots
===
Box:
![box](https://github.com/wuga214/Geospatial_Prediction/blob/RuntimeImproved/plots/result_box.png)
Circles:
![circles](https://github.com/wuga214/Geospatial_Prediction/blob/RuntimeImproved/plots/result_circles.png)
Stair:
![stair](https://github.com/wuga214/Geospatial_Prediction/blob/RuntimeImproved/plots/result_stair.png)
Cake:
![cake](https://github.com/wuga214/Geospatial_Prediction/blob/RuntimeImproved/plots/result_cake.png)

Gibbs Result Plot with bagging
===
![GibbsBox](https://github.com/wuga214/Geospatial_Prediction/blob/RuntimeImproved/BaggingPrediction.png)
