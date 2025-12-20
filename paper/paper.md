---
title: 'PyGALAX: An Open-Source Python Toolkit for Advanced Explainable Geospatial Machine Learning'
tags:
  - Python
  - Geographically Weighted Regression
  - Automated Machine Learning
  - Explainable GeoAI
authors:
  - name: Pingping Wang
    orcid: 0009-0008-8722-167X
    affiliation: 1
  - name: Yihong Yuan
    orcid: 0000-0001-6266-9744
    corresponding: true
    affiliation: 1
  - name: Lingcheng Li
    orcid: 0000-0002-6834-5844
    affiliation: 2
  - name: Yongmei Lu
    orcid: 0000-0003-1994-3458
    affiliation: 1
affiliations:
 - name: Department of Geography and Environmental Studies, Texas State University, San Marcos, TX, USA
   index: 1
 - name: Atmospheric, Climate, and Earth Sciences Division, Pacific Northwest National Laboratory, Richland, WA, USA
   index: 2
date: 20 December 2025
bibliography: paper.bib
---

# Summary

PyGALAX is a Python package for geospatial analysis that integrates automated machine learning (AutoML) and explainable artificial intelligence (XAI) techniques to analyze spatial heterogeneity in both regression and classification tasks. It automatically selects and optimizes machine learning models for different geographic locations and contexts while maintaining interpretability through SHAP (SHapley Additive exPlanations) analysis. PyGALAX builds upon and improves the GALAX framework (Geospatial Analysis Leveraging AutoML and eXplainable AI) [@Wang2025GALAX], which has proven to outperform traditional geographically weighted regression (GWR) methods. Critical enhancements in PyGALAX from the original GALAX framework include automatic bandwidth selection and flexible kernel function selection, providing greater flexibility and robustness for spatial modeling across diverse datasets and research questions.

PyGALAX not only inherits all the functionalities of the original GALAX framework but also packages them into an accessible, reproducible, and easily deployable Python toolkit while providing additional options for spatial modeling. It effectively addresses spatial non-stationarity and generates transparent insights into complex spatial relationships at both global and local scales, making advanced geospatial machine learning methods accessible to researchers and practitioners in geography, urban planning, environmental science, and related fields.

# Statement of need

Understanding spatially heterogeneous and non-linear relationships is a persistent challenge in geography and environmental sciences, such as urban analytics, human mobility studies, and public health [@Roy2024; @Wang2025COVID; @Wang2025Mobility; @Yuan2025]. Traditional approaches such as GWR provide valuable insights into spatial non-stationarity but are constrained by their linear assumptions and limited ability to handle complex, high-dimensional data [@Brunsdon1996; @Fotheringham2017]. In contrast, modern machine learning techniques can model complex, non-linear relationships but often ignore spatial context and operate as "black boxes," leading to reduced interpretability and limited utilization in decision-making applications [@Nagarajah2019; @Wang2021FLAML]. While recent advances have introduced machine learning into spatial analysis, existing tools lack the flexibility needed for diverse research applications. For example, PyGRF [@Sun2024] is a comparable Python package that implements the established Geographical Random Forests method for spatial analysis [@Yuan2025Crime]. While PyGRF advances spatial modeling by incorporating Random Forests into a geographic framework, it is constrained to a single algorithm and primarily focuses on regression tasks.

To address this methodological gap, GALAX was recently developed as a hybrid analytical framework that integrates AutoML and XAI within a spatial modeling structure [@Wang2025GALAX]. GALAX enables researchers to automatically identify optimal machine learning algorithms for each geographic location and context [@Wang2021FLAML], incorporate spatial weighting into the workflow [@Brunsdon1996], and interpret both global and local relationships through SHAP-based explainability [@Lundberg2017]. This integration enables a transparent, adaptive, and data-driven understanding of spatially varying processes, representing a significant advancement beyond both traditional GWR and non-spatial machine learning approaches.

Building upon the success of the GALAX conceptual framework, PyGALAX is created to operationalize GALAX as an open-source Python package, while providing additional spatial analysis flexibility for practitioners and researchers. Specifically, PyGALAX enhances the original GALAX model through automatic bandwidth and kernel selection and a user-friendly implementation. These improvements make spatial AutoML workflows accessible to a wider research community and ensure reproducibility across studies. Furthermore, PyGALAX supports both regression and classification tasks, expanding its applicability to diverse application domains such as mobility analysis, environmental monitoring, and public health research, areas where understanding both the "what" and the "why" of spatial patterns is critical for evidence-based decision-making [@Georganos2021; @Li2022].

# Installation

PyGALAX is compatible with Python 3.9 and later and can be easily installed using either pip or directly from its GitHub repository. To install the latest stable release from the Python Package Index (PyPI):
```bash
pip install PyGALAX
```

Alternatively, the latest development version can be installed directly from GitHub:
```bash
git clone https://github.com/Pingping9/PyGALAX
cd PyGALAX
pip install .
```

# Key features

PyGALAX offers several distinctive capabilities that make it suitable for advanced spatial analysis (Figure 1 and Table 1):

- **Spatial AutoML integration**: PyGALAX automates the GALAX framework that implements geographically weighted AutoML, where different machine learning algorithms (e.g., Random Forest, XGBoost, Extra Trees) are automatically selected and optimized for each spatial location based on local data characteristics. This approach enables the capture of varying relationship structures across geographic space.

- **Fixed or Adaptive bandwidth selection**: PyGALAX provides multiple bandwidth selection methods, including Incremental Spatial Autocorrelation (ISA) analysis [@Sun2024] and performance-based optimization [@Brunsdon1996], ensuring optimal spatial scale selection for different datasets and research objectives.

- **Explainable spatial AI**: Through SHAP integration, PyGALAX provides detailed explanations of model predictions, revealing how different features contribute to outcomes across geographic space. This includes both the importance of local features and the spatial patterns of variable influence.

- **Unified regression and classification**: Unlike many spatial analysis tools that focus exclusively on continuous outcomes, PyGALAX seamlessly handles both regression and classification tasks, making it versatile for diverse research applications.

- **Flexible kernel functions**: PyGALAX supports multiple spatial weighting schemes (e.g., bisquare, gaussian, exponential) and both fixed and adaptive bandwidth approaches [@Brunsdon1996], allowing customization for different spatial processes.

- **Parallel processing**: Built-in support for multicore processing enables efficient analysis of large spatial datasets.

![PyGALAX Methodological Framework Extended from @Wang2025GALAX.\label{fig:framework}](figure1.png)

: Example PyGALAX Commands []{label="tab:commands"}

| **Command** | **Description** |
|------|--------------|
| `Kernel(coords[i], coords, bw, fixed=False, function='bisquare')` | Create a kernel weighting matrix for spatial dependence modeling. |
| `search_bw_lw_ISA(X, y, coords)` | Run standalone ISA to estimate an optimal bandwidth. |
| `search_bandwidth(X, y, coords, automl_settings)` | Perform bandwidth optimization using AutoML model performance metrics. |
| `model = GALAX(coords, y, X, task='regression', bw=None, kernel='bisquare')` | Initialize a GALAX model with coordinates (coords), target variable (y), and features (X). The model automatically selects the optimal bandwidth when bw=None. |
| `results = model.fit()` | Fit the model using either ISA-based or performance-based bandwidth optimization, depending on data and task type. |
| `results.summary()` | Display summary statistics for global and local model performance (e.g., R2, RMSE, or accuracy, precision, recall, and F1 score). |
| `results.save_results(filename)` | Save model outputs (including SHAP values and predictions) as a Joblib file. |
| `results.get_detailed_shap_for_location(i)` | Retrieve detailed SHAP interpretation for location i. |

# Model architecture

PyGALAX is built on established Python libraries, including scikit-learn, FLAML for AutoML functionality, and SHAP for explainability. The modular architecture allows for easy extension and customization while maintaining computational efficiency through joblib-based parallel processing.

PyGALAX follows object-oriented design principles with clear separation between spatial weighting (Kernel class), model fitting (GALAX class), and results analysis (GALAXResults class). This design facilitates both ease of use for standard applications and extensibility for advanced research needs.

PyGALAX's modular structure enables future enhancements in multiple directions. The architecture allows for integration of additional AutoML backends (such as AutoGluon) and alternative XAI beyond SHAP. In addition, it provides a foundation for extending PyGALAX handles spatiotemporal data through geographically and temporally weighted frameworks, and for incorporating multi-scale analysis capabilities that account for varying spatial scales of different geographic processes. 

# Applications

PyGALAX enables advanced spatial analysis across diverse research domains:

- **Urban and transportation planning**: Analyzing spatial patterns in human mobility [@Wang2025GALAX], identifying factors influencing travel behavior across different urban contexts, and understanding how infrastructure and socioeconomic variables affect transportation choices.

- **Environmental science and ecology**: Modeling spatial variations in environmental quality, examining relationships between land use patterns and ecosystem services, and predicting pollution distribution with location-specific drivers.

- **Public health**: Investigating geographic disparities in health outcomes, identifying spatially varying risk factors for disease transmission, and optimizing healthcare resource allocation based on local population characteristics.

- **Crime analysis**: Examining spatial inequality in crime patterns, analyzing neighborhood effects on criminal activity, and understanding how contextual factors influence crime rates across different regions.

- **Real estate and economics**: Predicting property values with spatially varying determinants, analyzing local market dynamics, and identifying factors driving economic development in different geographic areas.

- **Agriculture and natural resources**: Modeling crop yields with location-specific environmental and management factors, optimizing resource allocation based on spatial heterogeneity, and predicting soil properties across agricultural landscapes.

- **Tourism and recreation analytics**: Understanding spatial patterns in visitor flows, modeling heterogeneous factors influencing site attractiveness, and optimizing infrastructure placement in parks or cultural destinations.

- **Disaster resilience and emergency management**: Identifying spatially varying drivers of vulnerability, modeling community-level resilience, and optimizing evacuation or resource-deployment strategies based on local conditions.

# Acknowledgements

The first and second authors received funding support from the 2024-2025 Texas State University College of Liberal Arts Seed Grant.

# References
