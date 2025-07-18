# StockSentiX: Stock Price Prediction Using X Sentiment

StockSentiX is a machine learning project that predicts stock prices by analyzing sentiment from posts on X. It leverages natural language processing (NLP) with the Hugging Face RoBERTa model to extract sentiment from real-time X data, combining it with historical stock price data to forecast market trends using an LSTM (Long Short-Term Memory) model. The project uses datasets of X posts and stock prices from October 2021 to September 2022, along with Jupyter Notebooks for fine-tuning the RoBERTa model, correlation analysis, data evaluation, and price prediction, providing actionable insights for traders and investors.

## Datasets

- **sentiment-tweets.csv**:

  - Contains Twitter (X) mentions of Dell from January 1 to September 30, 2022.
  - Each tweet is annotated with sentiment and emotion labels.
  - Used to fine-tune and assess the Hugging Face RoBERTa model’s performance for sentiment analysis.

- **stock_tweets.csv**:

  - Includes 80,793 tweets about the top 25 companies, collected from October 2021 to September 2022.
  - Provides a broad dataset for sentiment analysis across multiple companies.

- **stock_yfinance_data.csv**:

  - Provides historical stock price data for selected companies, covering October 2021 to September 2022.
  - Used for correlation analysis and stock price prediction with the LSTM model.

## Notebooks

- **Fine_tuning_model.ipynb**:

  - Jupyter Notebook containing scripts to fine-tune the Hugging Face RoBERTa model for sentiment analysis using the `sentiment-tweets.csv` dataset.

- **Correlation_Consistency.ipynb**:

  - Jupyter Notebook with scripts to calculate the correlation and consistency percentage between X post sentiment and stock price movements.

- **Evaluation_Data.ipynb**:

  - Jupyter Notebook with scripts to evaluate the tweet and stock price datasets, including visualizations like graphs for various factors.

- **Stock_Price_Predict.ipynb**:

  - Jupyter Notebook with scripts to train and evaluate an LSTM model for stock price prediction using sentiment scores and historical data.

## Features

- **Sentiment Analysis**: Extract positive, negative, or neutral sentiment from X posts using the fine-tuned Hugging Face RoBERTa model.
- **Stock Price Prediction**: Train an LSTM model to predict stock prices based on sentiment scores and historical data, capturing temporal dependencies.
- **Real-Time Data Integration**: Scrape live X posts via the X API for up-to-date sentiment analysis (requires an X developer account).
- **Data Visualisation**: Generate interactive charts to visualise stock price trends and sentiment correlations.
- **Robust Codebase**: Built with Python for efficient data processing, model training, and analysis.

## Setup

To run StockSentiX locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/gaganpreet13/StockSentiX.git
   cd StockSentiX
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure Python 3.8+ is installed. Key dependencies include `transformers` for Hugging Face RoBERTa, `tensorflow` for LSTM modeling, `nltk`, `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`.

3. **Configure Environment**:

   - Create a `.env` file in the project root.
   - Add your X API credentials to scrape tweets:

     ```env
     X_API_KEY=your_key
     X_API_SECRET=your_secret
     X_ACCESS_TOKEN=your_token
     X_ACCESS_TOKEN_SECRET=your_token_secret
     ```
   - Obtain credentials by creating an X developer account at https://developer.x.com.
   - Optionally, add stock data API keys (e.g., Alpha Vantage, Yahoo Finance) if used:

     ```env
     STOCK_API_KEY=your_stock_api_key
     ```

4. **Run Notebooks**:

   - Open Jupyter Notebook:

     ```bash
     jupyter notebook
     ```
   - Navigate to `Fine_tuning_model.ipynb`, `Correlation_Consistency.ipynb`, `Evaluation_Data.ipynb`, or `Stock_Price_Predict.ipynb` and run the cells as needed.

5. **Scrape Tweets**:

   - Create an X developer account at https://developer.x.com.
   - Obtain API credentials (API Key, API Secret, Access Token, Access Token Secret).
   - Update the scripts in the notebooks (e.g., `Stock_Price_Predict.ipynb`) to include your credentials for scraping real-time X posts.

## Technologies

- **Python**: Core language for data processing, machine learning, and scripting.
- **Hugging Face Transformers (RoBERTa)**: Pre-trained NLP model fine-tuned for sentiment analysis of X posts.
- **TensorFlow**: Framework for building and training the LSTM model for stock price prediction.
- **NLTK**: Library for additional NLP tasks and sentiment analysis.
- **X API**: Source for real-time X post data.
- **Pandas**: Data manipulation and analysis for stock and sentiment data.
- **Scikit-learn**: Machine learning utilities for model evaluation and preprocessing.
- **Matplotlib/Seaborn**: Data visualisation for trends and correlations.
- **Jupyter Notebook**: Interactive environment for data analysis and visualisation.

## License

This project is licensed under the MIT License.