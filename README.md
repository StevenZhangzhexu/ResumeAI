# ResumeAI

![language-python-blue](https://img.shields.io/badge/language-python-blue)
![language-css-blueviolet](https://img.shields.io/badge/language-css-blueviolet)
![language-yaml-yellowgreen](https://img.shields.io/badge/language-yaml-yellowgreen)
![NLP-Spacy-ff69b4](https://img.shields.io/badge/NLP-Spacy-ff69b4)
![NLP-GPT--3-orange](https://img.shields.io/badge/NLP-GPT--3-orange)
![database-MongoDB-brightgreen](https://img.shields.io/badge/database-MongoDB-brightgreen)
![cloud--app-streamlit-red](https://img.shields.io/badge/cloud--app-streamlit-red)


Welcome to ResumeAI! This application employs a Spacy model to analyze a CV and generates a cover letter based on the CV and job description, using a fine-tuned [Davinci model](https://beta.openai.com/docs/models/overview) based on OpenAI's GPT-3. Furthermore, it leverages web scraping and a matching algorithm to suggest suitable job positions that align with the individual's CV.






This repository contains:

1. The [README](/readme.md) for details of the app and a link to [the app ](https://resumeai.streamlit.app/).
2. [.streamlit](/.streamlit) and [static](/static) folders for the layout and style of the app.
3. A [models](/models) folder for Spacy model and a [.json](models/qualifications.jsonl) file for the cutomized entity ruler.
4. A [.github/workflows](.github/workflows) to use CI/CD pipline - a [Git Action](Update_DB.yml) to automatically run [script](/DS_job_DB.py) to scrape and store data to MongoDB on a daily basis.
5. [Pipfile](/Pipfile) - the requirements for the repo in the virtual environment.
6. The main [script](/app.py) of the App.


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
	- [Locally](#Locally)
	- [On Cloud](#Cloud)
- [Algorithm](#Algorithm)
- [Data & Automation](#Data&Automation)
- [Example Readmes](#example-readmes)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [License](#license)

## Background

As a job seeker, you may find the job application process to be lengthy and tedious, which usually involves manually reviewing and updating your CV, carefully scanning job requirements for positions you're interested in, and determining if you're a good fit for the role. Sometimes, you may need to craft a tailored Cover Letter for a specific position. I understand that this process can be overwhelming. Therefore I have developed this app ResumeAI to make the process as seamless and stress-free as possible!

The goals for this App are:

1. Give advises on your resume writing using STAR (situation, task, action, result) principle. (The STAR principle is an effective way to structure your resume bullet points to showcase your accomplishments and highlight your skills. )
2. Evaluate whether you are a good fit for a particular job posting.
3. Create a cover letter that aligns with your CV and the job description.
4. Provide recommendations for current, relevant job opportunities that match your qualifications.

## Install

If you intend to use the app locally, you need to install the required packages and dependencies for production usage like this: 
```sh
$ pipenv install
```

You can also install **dev-packages** if you want to work on developing this app:
```sh
$ pipenv install --dev
```

## Usage

### - Locally

After installation, run the following command in the terminal to activate the virtual environment:
```sh
$ pipenv shell
```
> Remember to set secrect (OPENAI API KEY & MONGODB KEY) in both local environment and streamlit.

```sh
#.env file
# Set the OpenAI API key
OPENAI_API_KEY=<your-api-key>

# Set the MongoDB URI
MONGODB_URI=<your-mongodb-uri>
```

```sh
#app.py
from dotenv import load_dotenv
import os #provides ways to access the Operating System and allows us to read the environment variables

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
```


Then run the app:
```sh
$ streamlit run app.py
```

### - On Cloud

Alternatively, there is a global executable version of app hosted on the streamlit cloud, aliased as `ResumeAI`.
To use the web app, please go to [ResumeAI](https://resumeai.streamlit.app/). 


## Algorithm

### NLP
I leveraged a fine-tuned **Spacy** model with a customized entity ruler for **entity detection**. For **cosine similarity** computation, I used the **CountVectorizer**. In addition, I applied the **GPT-3 Davinci model** to generate cover letters based on users' CVs.

### Matching Score

$Match_Score = 25 * Education_Score  + \alpha * Text\_CosineSimilarity + (75-\alpha) * Skill\_Score$,

where **Text_CosineSimilarity** =
```math
\cos ({\bf CV},{\bf JD})= {{\bf CV} {\bf JD} \over \|{\bf CV}\| \|{\bf JD}\|} = \frac{ \sum_{i=1}^{n}{{\bf CV}_i{\bf JD}_i} }{ \sqrt{\sum_{i=1}^{n}{({\bf CV}_i)^2}} \sqrt{\sum_{i=1}^{n}{({\bf JD}_i)^2}} }
```

and $\alpha$ is adjusted according to the difference in text size between the resume and job description.

## Data & Automation
The GitHub action is used to automate web scraping and data updates in MongoDB Atlas on a daily basis. (The policy is to remove historical data that is more than one month old.) The app retrieves data from MongoDB to provide quick recommendations.







## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) © Steven Zhang Zhexu
