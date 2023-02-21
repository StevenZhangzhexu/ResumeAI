from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import iso3166
from selenium import webdriver

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


def get_job_info(country_code: str, job_url: str) -> tuple:
    '''
    Extracts job info from the job URL (customized by country)
    '''
    job_url = f'https://{country_code}.indeed.com'+job_url
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


def search_indeed(country, city='', job_title='', num_pages=1):
    if job_title:
        job_title = job_title.replace(' ', '%20')

    # Default value for Singapore
    country_code ='sg'

    # Country code search
    for countries, information in iso3166.countries_by_name.items():
        if country in countries:
            country_code = iso3166.countries_by_name[countries].alpha2.lower()
            break

    data = {
        'title': [],
        'company': [],
        'location': [],
        'salary_est': [],
        'description': [],
        'link': []
    }

    for i in range(0, num_pages*10, 10):
        URL = f'https://{country_code}.indeed.com/jobs?q={job_title}&l={city}&start={i}'
        job_urls = get_job_urls(URL)
        for job_url in tqdm(job_urls, total=len(job_urls)):
            title, company, location, salary_est, description,link = get_job_info(country_code, job_url)
            data['title'].append(title)
            data['company'].append(company)
            data['description'].append(description)
            data['location'].append(location)
            data['salary_est'].append(salary_est)
            data['link'].append(link)

    res = pd.DataFrame(data)
    print(res)
    return res
