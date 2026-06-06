# Agentic Data Analysis

This is an autonomous AI agent built in LangGraph that can perform data analysis on a provided dataset.This uses and leverages the flexibilty of the langgraph of implementing complex logics and custom tools.



## Getting Setup

If you want to use the same dataset as me, you can download it from Kaggle below:

https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets/data 

Otherwise feel free to upload your own dataset!

Youll then need to install the requirements by running the following command:

```bash
pip install -r requirements.txt
```

and run the streamlit dashboard by running the following command:

```bash
streamlit run app.py --server.maxUploadSize 2000
```

Update the API key in the data_analysis_streamlit_app.py file with your own.

Enjoy!
