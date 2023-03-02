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

1. [The README](/readme.md) for details of the app and a link to [the app ](https://resumeai.streamlit.app/).
2. [.streamlit](/.streamlit) and [static](/static) folders for the layout and style of the app.
3. A link to [a generator](https://github.com/RichardLitt/generator-standard-readme) you can use to create standard READMEs.
4. [A badge](#badge) to point to this spec.
5. [Examples of standard READMEs](example-readmes/) - such as this file you are reading.

maintained ([work in progress](https://github.com/RichardLitt/standard-readme/issues/5)).

Standard Readme is designed for open source libraries. Although it’s [historically](#background) made for Node and npm projects, it also applies to libraries in other languages and package managers.


## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
	- [Locally](#Locally)
	- [On Cloud](#On Cloud)
- [Algorithm](#Algorithm)
- [Data & Automation](#Data & Automation)
- [Example Readmes](#example-readmes)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [License](#license)

## Background

As a job seeker, you may find the job application process to be lengthy and tedious, which usually involves manually reviewing and updating your CV, carefully scanning job requirements for positions you're interested in, and determining if you're a good fit for the role. Sometimes, you may need to craft a tailored Cover Letter for a specific position. I understand that this process can be overwhelming. Therefore I have developed this app ResumeAI to make the process as seamless and stress-free as possible!



> Remember: the documentation, not the code, defines what a module does.

~ [Ken Williams, Perl Hackers](http://mathforum.org/ken/perl_modules.html#document)

Writing READMEs is way too hard, and keeping them maintained is difficult. By offloading this process - making writing easier, making editing easier, making it clear whether or not an edit is up to spec or not - you can spend less time worrying about whether or not your initial documentation is good, and spend more time writing and using code.

By having a standard, users can spend less time searching for the information they want. They can also build tools to gather search terms from descriptions, to automatically run example code, to check licensing, and so on.

The goals for this repository are:

1. A well defined **specification**. This can be found in the [Spec document](spec.md). It is a constant work in progress; please open issues to discuss changes.
2. **An example README**. This Readme is fully standard-readme compliant, and there are more examples in the `example-readmes` folder.
3. A **linter** that can be used to look at errors in a given Readme. Please refer to the [tracking issue](https://github.com/RichardLitt/standard-readme/issues/5).
4. A **generator** that can be used to quickly scaffold out new READMEs. See [generator-standard-readme](https://github.com/RichardLitt/generator-standard-readme).
5. A **compliant badge** for users. See [the badge](#badge).

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
### Matching Score

>$Match\_Score = 25 * Education\_Score  + \alpha * Text\_CosineSimilarity + (75-\alpha) * Skill\_Score$,

where **Text_CosineSimilarity** =
**$\begin{equation}
\cos ({\bf CV},{\bf JD})= {{\bf CV} {\bf JD} \over \|{\bf CV}\| \|{\bf JD}\|} = \frac{ \sum_{i=1}^{n}{{\bf CV}_i{\bf JD}_i} }{ \sqrt{\sum_{i=1}^{n}{({\bf CV}_i)^2}} \sqrt{\sum_{i=1}^{n}{({\bf JD}_i)^2}} }
\end{equation}$**

and $\alpha$ is adjusted according to the difference in text size between the resume and job description.

## Data & Automation


```





## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) © Steven Zhang Zhexu
