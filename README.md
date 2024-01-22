
# Twitter Sentiment Analysis

## Overview

This Python script performs sentiment analysis on tweets collected from Twitter using the snscrape library. It includes features such as data cleaning, sentiment analysis, and visualization of sentiment distribution.

## Prerequisites

-   Python (version XYZ)
-   Required Python libraries (install using `pip install -r requirements.txt`)

## Installation

1.  Clone or download the repository to your local machine.
2.  Install the required Python libraries using `pip install -r requirements.txt`.

## Usage

1.  Run the `scrap_tweet` function to collect Twitter data based on the specified query and maximum tweet count.
2.  Clean the collected data by removing unnecessary elements and noise from the tweets.
3.  Perform sentiment analysis using TextBlob, a natural language processing library.
4.  Visualize the sentiment distribution with histograms and box plots.

## Script Explanation

-   **Data Scraping:**
    
    -   The script uses snscrape to scrape tweets based on a specified query and maximum tweet count.
-   **Data Cleaning:**
    
    -   The `clean_text` function removes mentions, URLs, and emojis from the tweet text.
-   **Sentiment Analysis:**
    
    -   TextBlob is used to calculate the polarity and subjectivity of each tweet.
    -   Tweets are labeled as positive, negative, or neutral based on their polarity.
-   **Visualization:**
    
    -   The script includes visualizations of sentiment distribution using histograms and box plots.

## Customization

-   Modify the `query` variable to change the search criteria for scraped tweets.
-   Experiment with different cleaning methods or sentiment analysis libraries.

## Results

-   The sentiment distribution is displayed through histograms, box plots, and other visualizations.

## Credits

-   The script uses the snscrape library for Twitter data scraping.
-   Sentiment analysis is performed using TextBlob.
