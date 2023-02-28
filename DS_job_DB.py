from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from pymongo import MongoClient
from datetime import datetime,timedelta
import os
from dotenv import load_dotenv
load_dotenv()

# due to limit of storage, only 3 are chosen to be scraped job posting from.
c_dict= {
#   "Australia": "https://au.indeed.com",
#   "Canada": "https://ca.indeed.com",
#   "Ireland": "https://ie.indeed.com",
  "Hong Kong": "https://hk.indeed.com",
#   "New Zealand": "https://ca.indeed.com",
#   "United Kingdom": "https://uk.indeed.com", 
  "United States of America": "https://indeed.com",
  "Singapore": "https://sg.indeed.com",
  }

def get_job_urls(URL: str) -> list:
    '''
    Extracts job urls from the search result page given by URL
    '''
    driver = webdriver.Chrome()
    driver.get(URL)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    job_urls = [a['href'] for a in soup.find_all(
        'a', {"id": lambda x: x and x.startswith('job_')})]
    return job_urls


def get_job_info(base_url: str, job_url: str) -> tuple:
    '''
    Extracts job info from the job URL (customized by country)
    '''
    job_url = base_url + job_url
    driver = webdriver.Chrome()
    driver.get(job_url)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    title = soup.find('h1').text
    job_info_main = soup.find(
        'div', {"class": lambda x: x and x.startswith('jobsearch')})
    job_info = job_info_main.find(
        'div', {"class": lambda x: x and x.startswith(
            'jobsearch-CompanyInfoWithoutHeaderImage')}
    )
    company = job_info.find(
        'div', {'class': lambda x: x and x.startswith('icl-u')}).text
    job = job_info.find_all(
        'div', {'class': None})
    location = [i.text for i in job][-1]

    try:
        salary_emp, salary_est = [li.text for li in job_info_main.find(
            'ul', {'class': lambda x: x and x.startswith('css-1lyr5hv')}).find_all('li')]
    except AttributeError:
        salary_emp, salary_est = '', ''

    job_description = job_info_main.find('div', {'id': 'jobDescriptionText'})
    description = [d.text.lstrip().replace('\n',"").replace('\t',"")
                   for d in job_description.find_all(['p', 'div'])]
    description = ' '.join(description)

    return (title, company, location, salary_est, description, job_url)


def search_indeed():
    '''
    Extracts data science related jobs info from countries
    '''
    data = {
        'title': [],
        'company': [],
        'location': [],
        'salary_est': [],
        'description': [],
        'link': [],
        'country': [],
        's_date': []
    }

    for key in c_dict:
        base_url = c_dict[key]
        for i in range(0, 6*10, 10):
            URL = base_url +f'/jobs?q=data+science&start={i}'
            job_urls = get_job_urls(URL)
            for job_url in tqdm(job_urls, total=len(job_urls)):
                title, company, location, salary_est, description,link = get_job_info(base_url, job_url)
                data['title'].append(title)
                data['company'].append(company)
                data['description'].append(description)
                data['location'].append(location)
                data['salary_est'].append(salary_est)
                data['link'].append(link)
                data['country'].append(key)
                data['s_date'].append(datetime.today().strftime('%Y-%m-%d'))

    res = pd.DataFrame(data)
    return res

if __name__ == "__main__":
    # Scrape Indeed Job Postings
    df = search_indeed()
    df.reset_index(inplace=True)
    data_dict = df.to_dict("records")

    # Connect to MongoDB
    uri = os.getenv('MONGO_URI')
    client =  MongoClient(uri)
    db = client['ResumeAI_DB']
    collection = db['Indeed_jobs']

    # Deletes historical documents from the collection: (1 month ago data)
    subtracted_date = pd.to_datetime(datetime.today().strftime('%Y-%m-%d')) - timedelta(days=30)
    subtracted_date = subtracted_date.strftime("%Y-%m-%d")
    collection.delete_many({"date": {"$lt": subtracted_date}})
    # Insert collection
    collection.insert_many(data_dict)