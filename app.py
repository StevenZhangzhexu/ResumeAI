import streamlit as st
import spacy
from annotated_text import annotated_text
import re
import pandas as pd
import docx2txt
import PyPDF2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib.ticker import MaxNLocator
import plotly.graph_objects as go
import seaborn as sns; sns.set(style="whitegrid")
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from datetime import date
import io
import openai
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import job_scraper as js
from pymongo import MongoClient


@st.cache_resource(show_spinner=False)
def load_models():
    chinese_model = spacy.load("./models/zh/")
    english_model = spacy.load("./models/en/")
    models = {"en": english_model, "zh": chinese_model}
    return models

@st.cache_resource(show_spinner=False)
def init_connection():
    user = st.secrets["db_username"]
    password = st.secrets["db_pswd"]
    cluster_name = st.secrets["cluster_name"]
    uri = f'mongodb+srv://{user}:{password}@{cluster_name}.ebvevig.mongodb.net/?retryWrites=true&w=majority'
    return MongoClient(uri)

@st.cache_data(ttl=1200)
def get_data():
    db = client.ResumeAI_DB
    items = db.Indeed_jobs.find()
    items = list(items)  # make hashable for st.cache_data
    return pd.DataFrame(items)

def process_text(doc):
    tokens = []
    count = 0
    prep = ["from","to","after","before","in"]
    nlist = ["competition", "test", "contest", "challenage", "race"]
    rlist = ["result","high","low","top","prize","medal","badge"]
    verbs = ['hold','create','cut','conduct','save','leverage','process','increase','decrease','reduce','maintain','slash','raise','boost','decline','add','implement','lead','accelerate','achieve','advance','deduct','enhance','expand','deliver','expedit','further','gain','generat','improve','lift','maximize','outpace','reconcile','reduce','save','stimulate','sustaine','yield','centralize','clarify','convert','customize','digitize','integrate','merge','modernize','modify','overhaul','redesign','refine','refocus','rehabilitate','remodel','reorganize','replace','restructure','revamp','revitalize','simplify','standardize','streamline','strengthen','transform','update','upgrad','sell','calculate','evaluate','forecast','measure','track','quantify','earn','exceed','outperform','operate','win']

    for sent in doc.sents:
        star = 0
        flag = 0
        for token in sent:
            if (token.pos_ == "VERB" and token.lemma_ in verbs) & (star == 0):
                star = 1
                tokens.append((token.text, "Action", "#8ef"))
            elif (token.pos_ == "NUM" or token.ent_type_ == "ORDINAL") & (token.ent_type_ != "DATE") & (star == 1):
                tokens.append((token.text, "Result", "#ffff80"))
                star = 0
                flag = 1
            elif token.lemma_ in nlist:
                tokens.append((token.text, "Task", "#afa"))
                star = 1
            elif (token.lemma_ in rlist) & (star == 1):
                tokens.append((token.text, "Result", "#ffff80"))
                star = 0
                flag = 1
            elif (token.pos_ == "PUNCT" and token == sent[-1]) & (star == 1 and flag == 0):
                tokens.append((token.text, "Consider adding results in this sentence", "#faa"))
                count += 1 
            else:
                tokens.append(" " + token.text + " ")
                continue

    res = [x for x in tokens]
    for i in range(1,len(tokens)-2):
        if (doc[i-1].lemma_ in prep or doc[i+1].lemma_ in prep or doc[i-1].dep_ == 'punct' or doc[i+1].dep_ == 'punct' or doc[i-1].ent_type_ == "DATE" or doc[i+1].ent_type_ == "DATE"):
            try: 
                int(doc[i].text) 
                if int(doc[i].text) in range(1960,2024):
                    res[i] = " " + doc[i].text + " "
            except: 
                continue

    return count,res

def extract_qualif(doc,flag=0):
    skills = []
    tokens = []
    extract_qualif.edu = 0
    for token in doc:
        if ("SKILL" in token.ent_type_):
            tokens.append((token.text, "skill", "linear-gradient(90deg, #9BE15D, #00E3AE)"))
        elif ("DEGREE" in token.ent_type_):
            tokens.append((token.text, "education", "#8ef"))
        else:
            tokens.append(" " + token.text + " ")   

    for ent in doc.ents:
        if "SKILL" in ent.label_:
            skill = re.findall(r'(?<=\|)\S*', ent.label_)[0]
            skills.append(skill)
        elif "DEGREE" in ent.label_ and flag==0:
            lvl = re.findall(r'(?<=\|)\S*', ent.label_)[0]
            if lvl=='BS-LEVEL' and extract_qualif.edu < 1:
                extract_qualif.edu = 1
            elif lvl=='MS-LEVEL' and extract_qualif.edu < 2:
                extract_qualif.edu = 2
            elif lvl=='PHD-LEVEL' and extract_qualif.edu < 3:
                extract_qualif.edu = 3       
        elif "DEGREE" in ent.label_ and flag==1:
            lvl = re.findall(r'(?<=\|)\S*', ent.label_)[0]
            if lvl=='BS-LEVEL':
                extract_qualif.edu = 1
            elif lvl=='MS-LEVEL' and extract_qualif.edu >2 or extract_qualif.edu == 0:
                extract_qualif.edu = 2
            elif lvl=='PHD-LEVEL' and extract_qualif.edu == 0:
                extract_qualif.edu = 3   

    unique_skills = list(set(skills))
    return unique_skills, tokens

def parse_time(doc,tokens):
    res = ""
    res_token  = [x for x in tokens]
    for i in range(len(doc)):
        if doc[i].lemma_ in ['now', 'today','present'] and doc[i-1].lemma_ in ['to','till','–'] and doc[i].dep_ != 'advcl':
            ret = str(date.today())
            res = res + ret[:4] +"/"+ ret[5:7]+"/"+ ret[8:10] +" "
            res_token[i] = (ret[:4] +"/"+ ret[5:7]+"/"+ ret[8:10], "DATE", "#fea")
            continue
        try:
            ret = str(parse(doc[i].text))
            year = int(ret[:4])
            if (year < 1950 or year > date.today().year) and doc[i].shape_=='dddd':
                res_token[i] = " " + doc[i].text + " "
            elif doc[i].shape_ == 'dddd' and '+' in [doc[i-1].text[0], doc[i-2].text[0]]:
                res_token[i] = " " + doc[i].text + " "
            elif "DATE" in doc[i].ent_type_ :
                res = res + doc[i].text  + " "
                res_token[i] = (doc[i].text, "date", "#fea")
            elif '%' in doc[i].text or doc[i].ent_type_ == 'PERCENT':
                res_token[i] = " " + doc[i].text + " "
            elif len(re.split('\.|\-|\/', doc[i].text))>1:
                if len (re.split('\.|\-|\/', doc[i].text)[-1])<4 and  len (re.split('\.|\-|\/', doc[i].text)[0])<4:
                    res_token[i] = " " + doc[i].text + " "
                    continue
                res = res + ret[:4] +"/"+ ret[5:7]+"/"+ ret[8:10] +" "
                res_token[i] = (ret[:4] +"/"+ ret[5:7]+"/"+ ret[8:10], "DATE", "#fea")
            else:
                res_token[i] = " " + doc[i].text + " "
        except:
            res = res + doc[i].text  + " "
            if ("DATE" in doc[i].ent_type_):
                res_token[i] = (doc[i].text, "date", "#fea")
        
    return res, res_token

def YoE(doc,skills,job_key):
    # list of keywords of YoE
    kw_1 = ['year','years','Year','Years']
    kw_2 = ['day','days','months','month','quater','quaters']
    dt_flg=0
    skill_flg=0
    dic = {k: 0.5 for k in skills}
    for ent in doc.ents:
        if "DATE" in ent.label_:
            if any(x in ent.text for x in kw_1):
                if job_key ==1:
                    try:
                        yr = re.findall('\d+',ent.text)[0]
                    except:
                        yr =0
                    yoe = int(yr)
                    skill_flg=1
                    skill_set = set()
                #yoe = int(re.sub('[^0-9]', '', ent.text))
            elif any(x in ent.text for x in kw_2):
                if job_key ==1:
                    skill_flg=0
                    continue
            elif dt_flg==0 and job_key ==0:
                dt_flg=1
                try:
                    start_date = parse(ent.text)
                except:
                    dt_flg=0
                    continue
                skill_flg=0
            elif dt_flg==1 and job_key ==0:
                try:
                    end_date = parse(ent.text)
                    yoe = relativedelta(end_date, start_date).years
                except:
                    continue
                if yoe>=0:
                    dt_flg=0
                    skill_flg=1
                    skill_set = set()
                else:
                    try:
                        start_date = parse(ent.text)
                    except:
                        continue
                    skill_flg=0
        elif "SKILL" in ent.label_ and re.findall(r'(?<=\|)\S*', ent.label_)[0] in skills and skill_flg==1:
            skill = re.findall(r'(?<=\|)\S*', ent.label_)[0]
            if skill in skill_set:
                continue 
            else:
                skill_set.add(skill)
                dic[re.findall(r'(?<=\|)\S*', ent.label_)[0]] += yoe
    return dic

def sim(str1,str2):
    str1=str1.replace("\n","")
    str2=str2.replace("\n","")
    comp = [str1,str2]
    cv=CountVectorizer()
    count_matrix=cv.fit_transform(comp)
    MatchPercentage=cosine_similarity(count_matrix)[0][1]*100
    MatchPercentage=round(MatchPercentage,2)
    return MatchPercentage

# app layout
# Force responsive layout for columns also on mobile
st.write(
    """<style>
    [data-testid="column"] {
        width: calc(50% - 1rem);
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
    </style>""",
    unsafe_allow_html=True,
)
# Render Streamlit page
image = Image.open('./static/logo.png')
st.image(image)
st.markdown(
    "*This application employs a Spacy model to analyze a CV and generates a cover letter based on the CV and job description, using a fine-tuned [Davinci model](https://beta.openai.com/docs/models/overview) based on OpenAI's GPT-3. Furthermore, it leverages web scraping and a matching algorithm to suggest suitable job positions that align with the individual's CV.*"
)


models = load_models()
with st.sidebar:
    st.header("The Navigation Pane")
    selected_language = st.sidebar.selectbox("Select a language", options=["en"])
    # widget to choose which continent to display
    ft_list = ["Analyse CV", "Assess Qualification", "Generate Cover Letter", "Recommend DS Jobs(fast)","Recommend Jobs(slow)"]
    ft = st.selectbox(label = "Choose a function", options = ft_list)
    if ft=='Recommend DS Jobs(fast)':
        country = st.selectbox('Country', options=['Singapore','Hong Kong','United States of America'])
    if ft=='Recommend Jobs(slow)':
        st.subheader("Filter Conditions")
        country = st.text_input('Country*', 'Singapore')
        city = st.text_input('City', '')
        title = st.text_input('Title*', 'Data Scientist')
        pages = st.selectbox('Choose number of scrapped pages*', options = [i for i in range(2,21)])
        bt = st.button("**Start Search**",key="1")



selected_model = models[selected_language]

st.subheader("Resume/CV")
text_input = st.text_area("Type  a CV text")

uploaded_file = st.file_uploader("or Upload a file", type=["doc", "docx", "pdf", "txt"])
if uploaded_file is not None:
    if 'document' in uploaded_file.type:
        text_input = docx2txt.process(uploaded_file)
    elif 'pdf' in uploaded_file.type:
            uploaded_file.seek(0)
            file = uploaded_file.read()
            pdf = PyPDF2.PdfReader(io.BytesIO(file))
            text_input =''
            for page in range(len(pdf.pages)):
                text_input += (pdf.pages[page].extract_text())
    else:
        text_input = uploaded_file.getvalue()
        text_input = text_input.decode('utf-8',errors='replace')

st.subheader("Job posting/JD")
text_JD = st.text_area("Type a Job Posting text")
st.markdown("---")

doc = selected_model(text_input)
doc_JD =  selected_model(text_JD)


if ft=="Analyse CV":
    count,tokens = process_text(doc)
    if count == 0 and len(tokens)>0:
        with open('./static/cong.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.markdown('<p class="span">Good job! Your resume is following the STAR principle.</p>',unsafe_allow_html=True)
    elif count > 0:
        st.subheader("Note:")
        st.markdown(f"**There are {count} sentence(s) need to be refined.**")
    else:
        with open('./static/notification.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.markdown('<p class="span">Please type or upload a CV.</p>',unsafe_allow_html=True)
    with st.expander("See Highlighs in Resume"):
        annotated_text(*tokens)


if ft=="Assess Qualification":
    if len(doc)==0:
        with open('./static/notification.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.markdown('<p class="span">Please type or upload a CV.</p>',unsafe_allow_html=True)
    elif len(doc_JD)==0:
        with open('./static/notification.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.markdown('<p class="span">Please input a Job Posting.</p>',unsafe_allow_html=True)
    else:
        nlp = selected_model
        # Add entity ruler to pipe
        if "entity_ruler" not in nlp.pipe_names:
            skill_path = "models/qualifications.jsonl"
            ruler = nlp.add_pipe("entity_ruler",  config={"overwrite_ents": True})
            ruler.from_disk(skill_path)

        # Extract skills from CV 
        doc = nlp(text_input)
        skills, tokens = extract_qualif(doc)
        # Extract education level
        edu_cv = extract_qualif.edu
        # Extract YOE
        res, res_token = parse_time(doc, tokens)
        doc1 = nlp(res)

        # Extract skills from JD
        doc_JD =  nlp(text_JD)
        skills_JD, tokens_JD = extract_qualif(doc_JD,1)
        # Extract education level
        edu_jd = extract_qualif.edu
        res_J, res_token_J = parse_time(doc_JD, tokens_JD)
        doc2 = nlp(res_J)

        if len(text_input) > 5 *len(text_JD):
            sim_pt = 25
        elif len(text_input) > 2.5 *len(text_JD):
            sim_pt = 35
        else:
            sim_pt = 40
        # Calculate Matching Score
        MatchPercentage = sim(text_input,text_JD)
        sim_score = MatchPercentage/100 * sim_pt
        edu = 1 if edu_cv >= edu_jd and edu_cv!=0 else 0
        edu_score = edu * 25
        level = ['Bachelor\'s or higer','Master or higer','PHD or higer']

        set_a = set(skills[:])
        set_b = set(skills_JD[:])
        skill_same = set_a & set_b
        y1 = YoE(doc1,skills,0)
        d1 = pd.DataFrame(y1.items(), columns=['Skills','Your YoE'])
        y2 = YoE(doc2,skills_JD,1)
        d2 = pd.DataFrame(y2.items(),  columns=['Skills','YoE Required by The Postition'])
        df = pd.concat([d1.set_index('Skills'), d2.set_index('Skills')], axis=1, join="outer").sort_values(by=['YoE Required by The Postition', 'Your YoE'])
        skill_pct = (len(df[df['Your YoE']>=df['YoE Required by The Postition']]) + len(skill_same))/max(1,len(set_b))
        skill_score = skill_pct/2 * (75-sim_pt)
        match_score = skill_score + edu_score + sim_score

        if match_score>=90:
            color = "lightgreen"
        elif match_score>=70:
            color = 'cyan'
        elif match_score>=50:
            color = '#00BFFF'
        else:
            color = 'orange'

        with open('./static/title.css') as tt:
            st.markdown(f'<style>{tt.read()}</style>', unsafe_allow_html=True)
        st.markdown('<p class="font">Comparing Qualifications</p>', unsafe_allow_html=True)
        st.subheader("**Summary**")
        st.markdown(f"You have skills in **{str(skills)[1:-1]}** based on your resume")
        if edu==0 and edu_jd>0:
            st.markdown(f"The position requires skills in **{str(skills_JD)[1:-1]}** and education level of **{level[edu_jd-1]}** based on JD.")
        else:
            st.markdown(f"The position requires skills in **{str(skills_JD)[1:-1]}** based on JD")
        if match_score > 50:
            st.markdown(f"Your matching score with the role is {match_score:.2f}. You can go ahead and apply for this position.")
        else:
            st.markdown(f"Your matching score with the role is {match_score:.2f}. It seems that this position may not be the best fit for you, so you might want to consider exploring other opportunities.")
        fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = match_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color }},
        title = {'text': "Matching Score"}))
        st.plotly_chart(fig)

        st.subheader("**Details**")
        with st.expander("See Highlighs in Resume"):
            annotated_text(*res_token)
        with st.expander("See Highlighs in JD"):
            annotated_text(*res_token_J)

        st.subheader("**Visualization of Skills**")
        # plot skills comparison on app
        fig, ax = plt.subplots()
        fig.set_size_inches(20, 20)
        if len(set_a) > 0 and len(set_b) > 0:    
            venn = venn2([set_a,set_b],set_labels=['skills from resume','skills from JD'])
            if len(set_a-set_b) > 0:
                venn.get_label_by_id('100').set_text('\n'.join(map(str,set_a-set_b)))
                venn.get_label_by_id('100').set_size(20)
            if len(set_a&set_b) > 0:
                venn.get_label_by_id('110').set_text('\n'.join(map(str,set_a&set_b)))
                venn.get_label_by_id('110').set_size(20)
            if len(set_b-set_a) > 0:
                venn.get_label_by_id('010').set_text('\n'.join(map(str,set_b-set_a)))
                venn.get_label_by_id('010').set_size(20)
            venn.get_label_by_id('A').set_size(25)
            venn.get_label_by_id('B').set_size(25)
            st.pyplot(fig)

        st.subheader("**Visualization of Years of Experience**")
        # plot YOE comparison
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.set_size_inches(8, 17)
        try:
            df.plot.barh(ax=ax, color={'YoE Required by The Postition': "forestgreen", 'Your YoE':"lightsalmon"})
            st.pyplot(fig)
        except:
            st.markdown("no skills")


if ft == "Generate Cover Letter":
    if len(doc)==0:
        with open('./static/notification.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.markdown('<p class="span">Please type or upload a CV.</p>',unsafe_allow_html=True)
    elif len(doc_JD)==0:
        with open('./static/notification.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.markdown('<p class="span">Please input a Job Posting.</p>',unsafe_allow_html=True)
    elif len(doc)+len(doc_JD)>1700:
        with open('./static/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.markdown(f'<p class="span">Please Reduce Your Input Size. <br> Current Input Size:  {len(doc)+len(doc_JD)} tokens. <br>  Maximun Input Size: 1700 tokens.</p>', unsafe_allow_html=True)
    else:
        new_prompt = 'Write a cover letter based on below resume and the job posting: Resume: ' + doc.text.replace('\n',"").replace('\t',"") + ' Job Posting: ' + doc_JD.text.replace('\n',"").replace('\t',"") + '\n'
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        answer = openai.Completion.create(
        model='text-davinci-002',
        prompt = new_prompt,
        max_tokens=250,
        temperature=0
        )
        st.header("**Cover Letter**")
        st.markdown(answer['choices'][0]['text'])

def score(text_input, skills, d1, text_JD ):
    # Extract skills from JD
    doc_JD =  nlp(text_JD)
    skills_JD, tokens_JD = extract_qualif(doc_JD,1)
    # Extract education level
    edu_jd = extract_qualif.edu
    res_J, _ = parse_time(doc_JD, tokens_JD)
    doc2 = nlp(res_J)

    if len(text_input) > 5 *len(text_JD):
        sim_pt = 25
    elif len(text_input) > 2.5 *len(text_JD):
        sim_pt = 35
    else:
        sim_pt = 40
    # Calculate Matching Score
    MatchPercentage = sim(text_input,text_JD)
    sim_score = MatchPercentage/100 * sim_pt
    edu = 1 if edu_cv >= edu_jd and edu_cv!=0 else 0
    edu_score = edu * 25
    set_a = set(skills[:])
    set_b = set(skills_JD[:])
    skill_same = set_a & set_b
    y2 = YoE(doc2,skills_JD,1)
    d2 = pd.DataFrame(y2.items(),  columns=['Skills','YoE Required by The Postition'])
    df = pd.concat([d1.set_index('Skills'), d2.set_index('Skills')], axis=1, join="outer").sort_values(by=['YoE Required by The Postition', 'Your YoE'])
    skill_pct = (len(df[df['Your YoE']>=df['YoE Required by The Postition']]) + len(skill_same))/max(1,len(set_b))
    skill_score = skill_pct/2 * (75-sim_pt)
    match_score = skill_score + edu_score + sim_score
    return match_score

if ft == "Recommend DS Jobs(fast)":
    if doc:
        nlp = selected_model
        # Add entity ruler to pipe
        if "entity_ruler" not in nlp.pipe_names:
            skill_path = "models/qualifications.jsonl"
            ruler = nlp.add_pipe("entity_ruler",  config={"overwrite_ents": True})
            ruler.from_disk(skill_path)
        # Extract skills from CV 
        doc = nlp(text_input)
        skills, tokens = extract_qualif(doc)
        # Extract education level
        edu_cv = extract_qualif.edu
        # Extract YOE
        res, res_token = parse_time(doc, tokens)
        doc1 = nlp(res)
        y1 = YoE(doc1,skills,0)
        d1 = pd.DataFrame(y1.items(), columns=['Skills','Your YoE'])
        # Get data from MongoDB
        client = init_connection()
        df = get_data()
        df = df[df['country']==country].drop_duplicates(subset=df.columns.difference(['s_date']))
        df['match_score'] = df.apply(lambda x: score(text_input, skills, d1, x['description']), axis=1)
        df = df[['title','company','location','description','link','country','s_date','match_score']].reset_index()
        st.subheader("**Full list**")
        st.dataframe(df)
        st.subheader("**Recommendation list**")
        st.dataframe(df[df['match_score']>50])
    else:
        with open('./static/notification.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.markdown('<p class="span">Please type or upload a CV.</p>',unsafe_allow_html=True)



if ft == "Recommend Jobs(slow)" and bt:
    if doc:
        nlp = selected_model
        # Add entity ruler to pipe
        if "entity_ruler" not in nlp.pipe_names:
            skill_path = "models/qualifications.jsonl"
            ruler = nlp.add_pipe("entity_ruler",  config={"overwrite_ents": True})
            ruler.from_disk(skill_path)
        # Extract skills from CV 
        doc = nlp(text_input)
        skills, tokens = extract_qualif(doc)
        # Extract education level
        edu_cv = extract_qualif.edu
        # Extract YOE
        res, res_token = parse_time(doc, tokens)
        doc1 = nlp(res)
        y1 = YoE(doc1,skills,0)
        d1 = pd.DataFrame(y1.items(), columns=['Skills','Your YoE'])

        df = js.search_indeed(job_title = title, country = country, num_pages = pages)
        if len(df)<1:
            st.dataframe(df)
        else:
            df['match_score'] = df.apply(lambda x: score(text_input, skills, d1, x['description']), 
                            axis=1)

            st.subheader("**Full list**")
            st.dataframe(df)
            st.subheader("**Recommendation list**")
            st.dataframe(df[df['match_score']>50])
    else:
        with open('./static/notification.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        st.markdown('<p class="span">Please type or upload a CV.</p>',unsafe_allow_html=True)



    
