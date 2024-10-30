import requests
import os
import time
import pandas as pd

def get_access_token():
    url = "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'password',
        'username': 'username',
        'password': 'password',
        'response_type': 'id_token',
        'scope': 'scope',
        'client_id': 'client_id'
    } # enter your account's info
    response = requests.post(url, headers=headers, data=data)
    return response.json().get('access_token')

def get_wind_power_data(access_token, page , datefrom , dateto):
    url = "https://api.ercot.com/api/public-reports/archive/DATASET_NAME" # Enter the Dataset ID from ERCOT website
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Ocp-Apim-Subscription-Key": "Subscription-Key" # Enter the Subscription key specific to your account
    }
    params = {
        "size": 1000,
        "page": page,
        "postDatetimeFrom": datefrom,
        "postDatetimeTo": dateto,
    }
    retries = 500
    backoff_factor = 0.5
    for i in range(retries):
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait = backoff_factor * (2 ** i)
            print(f"Rate limit exceeded for initial call. Retrying in {wait} seconds...")
            time.sleep(wait)
        else:
            print(f"Failed to retrieve page {page}. Status code: {response.status_code}, Response: {response.text}")
            return None
    return None

def download_and_save_zip(url, access_token, subscription_key, filename):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Ocp-Apim-Subscription-Key": subscription_key
    }
    retries = 500
    backoff_factor = 0.5
    for i in range(retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            return
        elif response.status_code == 429:
            wait = backoff_factor * (2 ** i)
            #print(f"Rate limit exceeded. Retrying in {wait} seconds...")
            time.sleep(wait)
        else:
            print(f"Failed to download file. Status code: {response.status_code}, Response: {response.text}")
            return None
    return None

def process_all_pages(subscription_key):
    print('Processing wind, 1h data: ')
    if not os.path.exists("wind_h"):
        os.makedirs("wind_h")
    #year = [2019 , 2020 , 2021 , 2022]
    year = [2023]
    for what_year in year:
        current_page = 1
        total_pages = 1
        if not os.path.exists(f"wind_h/{what_year}"):
            os.makedirs(f"wind_h/{what_year}")
        
        datefrom = str(what_year)+ '-01-01T00:00'
        dateto = str(what_year+1) + '-01-02T00:00'
        
        while current_page <= total_pages:
            print('current_page:', current_page)
            access_token = get_access_token()
            data = get_wind_power_data(access_token, current_page , datefrom , dateto)
            if current_page == 1:
                total_pages = data['_meta']['totalPages']

            for archive in data['archives']:
                post_datetime = pd.to_datetime(archive['postDatetime'])
                download_url = archive['_links']['endpoint']['href']
                filename = f"wind_h/{what_year}/wind_h_{post_datetime}.zip"
                download_and_save_zip(download_url, access_token, subscription_key, filename)
            current_page += 1
        
subscription_key = "subscription_key" # Enter Subscription key
process_all_pages(subscription_key)
