import urllib2
import base64
import os
import pandas as pd

USER = "89d61164743acfb8d2bb7aa8e6582000"
PASSWORD = "cf4b0dedd3ee1ece4e6fa80d09d029c1"

NEWS_URL = "https://api.intrinio.com/news.csv?ticker={}"
GOOGLE_URL = "https://finance.google.com/finance/historical?q={}&startdate={}&output=csv" #(%Jan 1, 2014)


def google_request(ticker, base_url=GOOGLE_URL, startdate="Jan%201,%202014"):
	endpoint = base_url.format(ticker, startdate)
	print(endpoint)
	response = urllib2.urlopen(endpoint)
	return response

def intrinio_api_request(ticker, base_url=NEWS_URL, user=USER, password=PASSWORD):
	endpoint = base_url.format(ticker)
	auth = base64.b64encode("{}:{}".format(user, password))
	req = urllib2.Request(endpoint, headers={'Authorization': 'Basic {}'.format(auth) })
	response = urllib2.urlopen(req)
	return response

def response_to_file(response, filename):
	with open(filename, "w") as f:
		f.write(response.read())

def load_tickers(sector, folder=os.path.join("..", "data")):
	sector = sector.lower()
	sub_folder = os.path.join(folder, sector, "companies")
	result = set()
	for f in os.listdir(sub_folder):
		result = result | set(pd.read_csv(os.path.join(sub_folder, f))['Symbol'].unique())
	return result


def load_news_data(sector, ticker):
	filename = os.path.join("..", "data", sector, "news", ticker + ".csv")
	return pd.read_csv(filename, header=1)

def load_stocks_data(sector, ticker):
	filename = os.path.join("..", "data", sector, "stocks", ticker + ".csv")
	return pd.read_csv(filename)