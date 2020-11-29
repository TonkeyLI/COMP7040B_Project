# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:27:00 2020

@author: USER
"""

# This is needed when we need to do web scraping daily
import schedule 

# Check time if needed
import time 

# Use pip install to install twint. Do not use git repo to install git otherwise errors might occur
import twint 

# For avoiding scraping error
import nest_asyncio

import pandas as pd
import numpy as np


#The following codes extend the dataframe so we can visually see the dataframe more easily
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_columns', None)
pd.set_option('display.max_rows', None)

# For avoiding scraping error
nest_asyncio.apply()

# Initialize twint
c = twint.Config()

# Search for the twitter ID, "CAgovernor"
c.Search = "CAgovernor"

# Enable saving to a csv
c.Store_csv = True

# Data starts from 2020-04-01 until now
c.Since = "2020-04-01 00:00:00"

# Store the web scraped data into a csv
c.Output = "California.csv"

#c.Hide_output = True -> Set this to avoid too much data being displayed

# Start searching
twint.run.Search(c)

# Read the csv into a dataframe if needed
Tweets_df = pd.read_csv("California.csv")
