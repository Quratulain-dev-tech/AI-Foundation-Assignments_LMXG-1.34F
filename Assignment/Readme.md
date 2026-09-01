# Parallel News Web Scraper

A Python-based parallel web scraping project designed to efficiently collect and preprocess news articles from multiple URLs.

## Project Description

This project uses **Python, Requests, and BeautifulSoup** to scrape news articles from multiple web pages. Parallel/concurrent processing is used to improve scraping efficiency and reduce the overall execution time.

The scraped data is cleaned and preprocessed before being stored in CSV files. The project also keeps track of URLs that could not be scraped successfully.

## Features

* Scrapes news articles from multiple URLs
* Uses **Requests** for sending HTTP requests
* Uses **BeautifulSoup** for HTML parsing and data extraction
* Performs parallel/concurrent web scraping
* Preprocesses scraped article data
* Stores scraped and preprocessed data in CSV files
* Tracks failed URLs separately
* Uses a requirements file for project dependencies

## Project Files

* `scrape_all_urls.py` — Main script for scraping articles from multiple URLs
* `preprocess_data.py` — Cleans and preprocesses the scraped data
* `scraped_all_articles.csv` — Raw scraped article data
* `preprocessed_all_articles.csv` — Processed and cleaned article data
* `scraping_failed.csv` — URLs that failed during scraping
* `urls.csv.json` — Source URLs used for scraping
* `requirements.txt` — Required Python libraries

## Technologies Used

* Python
* Requests
* BeautifulSoup
* Web Scraping
* Parallel/Concurrent Processing
* CSV Data Processing

## Purpose

The main purpose of this project is to demonstrate efficient web scraping, HTML parsing, data preprocessing, file handling, and parallel processing using Python.
