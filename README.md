# Traffic severity using TDA
folder **Version1** with an initialize analysis thoughts and **Version2** record a final version of the analyze.<br/> This conference paper has been include in the transportation safety section of [Chinese Institute of Transportation](https://drive.google.com/file/d/1Xwn50CmsidK9w3uSz0Jawzatufz-S2pV/view)<br/>
**Version3** includes an improved analysis for traffic data. Since PCA isn't applicable for labeled data, therefore, Multiple correspondence analysis (MCA) has been introduced as the primary improvement. Additionally, analytical methods have been applied to Mapper for enhanced analysis. This version has been submitted to the National Science and Technology Council (R.O.C) and is currently under review. <br/>
**Version4** builds on Version3 but introduces a refinement by splitting the "Driver" category into "Car" and "Motor."


Note that folder **tdamapper & tests** is from [this](https://github.com/lucasimi/tda-mapper-python) package, this is for specifice use for clustering optimization.

## Abstract
<i>International Conference and Annual Meeting of the Chinese Institute of Transportation</i>

This conference paper was included in Volume Four under **Transportation Safety**
: [Traffic accident risk identification method considering multi-dimensional road characteristic data: based on topological data analysis](https://drive.google.com/drive/u/0/folders/1rTrdAq0yBrjAuvs8IkQd--Cw67biqoYC)

In the study of traffic accident risks, driving behavior and weather conditions are often considered the primary influencing factors, but the role of road design cannot be overlooked. This research employs the Mapper algorithm from Topological Data Analysis (TDA) to analyze high-dimensional road design features, including lane configurations, road types, traffic signals, and surface conditions, to explore their association with the severity of traffic accidents. The study focuses on dissecting the process of the Mapper method, aiming to identify road design characteristics that may lead to higher fatality rates through topologically distinctive structures. Additionally, a comparison with traditional clustering methods is conducted to highlight the unique advantages of TDA in handling complex structural data. The research results show that topological data analysis can accurately capture the hidden structure in high-dimensional data. In addition to identifying road features that may lead to high mortality, it can also provide insightful explanatory models and provides new analytical perspectives and strategic reference suggestions for road safety design.