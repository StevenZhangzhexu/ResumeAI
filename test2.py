from pymongo import MongoClient
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


# # user = st.secrets["db_username"]
# # password = st.secrets["db_pswd"]
# # cluster_name = st.secrets["cluster_name"]
# # uri = f'mongodb+srv://{user}:{password}@{cluster_name}.ebvevig.mongodb.net/?retryWrites=true&w=majority'

# # items = db.Indeed_jobs.find()
# # x = list(items)[0]
# # print(client.server_info())
# ## collection.delete_many({})

uri = os.getenv('MONGO_URI')
client =  MongoClient(uri)
db = client['ResumeAI_DB']
collection = db['test']
d = {'id':'63f9f8bcf8a6927cfc7e52db', 'index':0,
's_date':"2023-02-25"}
collection.insert_one(d)
#collection.delete_many({})