# project/server/main/tasks.py

import datetime
from google.cloud import storage
from itertools import chain
import json
from nltk.corpus import wordnet
import numpy as np
import os
import pandas as pd
import psycopg2
import requests
import time
import torch
import traceback
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import uuid

# Custom packages
import project.server.main.keyword_extractor as ky
import project.server.main.asn1 as asn1

###############################
# PREPROCESSING AND CONSTANTS #
###############################

# Whether to log data or not
LOG_DATA = False

# Google CLoud login
STORAGE_CLIENT = storage.Client.from_service_account_json('molten_key.json')
BUCKET = 'biotech_lee'
LOG_PATH = 'mrc_logs'
ERROR_LOG_PATH = 'mrc_error_logs'
FEEDBACK_LOG_PATH = 'mrc_feedback_logs'

# Connect to DB
def connect_to_postgres_db():
    return psycopg2.connect("dbname='postgres' user='postgres' host='34.107.41.140' port='5432' password='xxxxxxxx'")

# Distilbert is much faster than BERT, while barely having an impact in performance
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# Quantization for a speedup of around 30%
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

#####################
# LOGGING FUNCTIONS #
#####################

def answer_is_valid(answer):
    """
    Given an answer from the model, return True if the answer is considered
    valid and False otherwise.
    """
    return len(answer)>1 and '[CLS]' not in answer and '[SEP]' not in answer

def log_data(user_id, question, keywords, nOutputs, start_year, end_year, df, total_n_articles, total_time, db_time):
    """
    Function to log all the relevant data of a particular question.
    """
    search_uuid = str(uuid.uuid1())
    date, time = str(datetime.datetime.now()).split()

    answer_dicts = []
    for index, row in df.iterrows():
        if answer_is_valid(row['answer']):
            answer_dicts.append({'pmid': row['pmid'], 'answer': row['answer']})
        else:
            answer_dicts.append({'pmid': row['pmid'], 'answer': ''})

    log_dict = {
        'id': search_uuid,
        'User': user_id,
        'Date': date,
        'Time': time,
        'question': question,
        'keywords': keywords,
        'nOutputs': nOutputs,
        'start_year': start_year,
        'end_year': end_year,
        'n_matches': total_n_articles,
        'answers': answer_dicts,
        'total_execution_time': total_time,
        'db_execution_time': db_time
    }

    output_string = json.dumps(log_dict)

    bucket = STORAGE_CLIENT.get_bucket(BUCKET)
    output_file = LOG_PATH + '/' + search_uuid + '.json'
    blob = bucket.blob(output_file)
    blob.upload_from_string(output_string)

    return

def log_error(user_id, question, keywords, nOutputs, min_year, max_year, error_message):
    search_uuid = str(uuid.uuid1())
    date, time = str(datetime.datetime.now()).split()
    log_dict = {
        'id': search_uuid,
        'User': user_id,
        'Date': date,
        'Time': time,
        'question': question,
        'keywords': keywords,
        'nOutputs': nOutputs,
        'start_year': min_year,
        'end_year': max_year,
        'error_message': error_message
    }

    output_string = json.dumps(log_dict)

    bucket = STORAGE_CLIENT.get_bucket(BUCKET)
    output_file = ERROR_LOG_PATH + '/' + search_uuid + '.json'
    blob = bucket.blob(output_file)
    blob.upload_from_string(output_string)

    return

def log_feedback(params):
    date, time = str(datetime.datetime.now()).split()
    params['Date'] = date
    params['Time'] = time

    output_string = json.dumps(params)

    bucket = STORAGE_CLIENT.get_bucket(BUCKET)
    output_file = FEEDBACK_LOG_PATH + '/' + params['task_id'] + '.json'
    blob = bucket.blob(output_file)
    blob.upload_from_string(output_string)

    return

#####################
# UTILITY FUNCTIONS #
#####################

def get_abstracts_from_postgres(keywords, nOutputs, year_range):
    """
    Given a list of keywords, this function returns a pandas DataFrame
    containing all the relevant information of the articles that contain the
    given keywords in the abstract.
    """
    max_n = 10000

    conn = connect_to_postgres_db()

    with conn.cursor() as cur:
        if len(keywords)==1:
            sorted_words = keywords
            query = "SELECT pmid FROM prod.mrc_keywords WHERE keyword='{word}' LIMIT {max_n}".format(word=sorted_words[0], max_n=max_n)
        else:
            keywords_string = "('" + "','".join(keywords) + "')"
            cur.execute("SELECT keyword, n FROM prod.mrc_keyword_frequency WHERE keyword IN {ks};".format(ks=keywords_string))
            records = cur.fetchall()
            sorted_words = [x[0] for x in sorted(records, key=lambda item: int(item[1]))]
            if len(sorted_words)==0:
                df = pd.DataFrame()
                return df
            query = "SELECT pmid FROM prod.mrc_keywords WHERE keyword='{word}'".format(word=sorted_words[0])

        cur.execute(query)
        records = cur.fetchall()
        pmids = set(x[0] for x in records)

        for i in range(1, len(sorted_words)):
            if len(pmids)==0:
                df = pd.DataFrame()
                return df
            word = sorted_words[i]
            pmids_string = "('" + "','".join(pmids) + "')"
            if i==len(sorted_words)-1:
                query = "SELECT pmid FROM prod.mrc_keywords WHERE keyword='{word}' AND pmid IN {ps} LIMIT {max_n}".format(word=word, ps=pmids_string, max_n=max_n)
            else:
                query = "SELECT pmid FROM prod.mrc_keywords WHERE keyword='{word}' AND pmid IN {ps}".format(word=word, ps=pmids_string)
            cur.execute(query)
            records = cur.fetchall()
            pmids = set(x[0] for x in records)

        pmids_string = "('" + "','".join(pmids) + "')"
        cur.execute("SELECT * from prod.mrc_data WHERE pmid in {ps}".format(ps=pmids_string))
        records = cur.fetchall()

    df = pd.DataFrame(records, columns=['pmid', 'abstract', 'article_title', 'article_date', 'journal', 'authors'])
    df = df[(df['article_date'] >= str(year_range[0])+'-01-01') & (df['article_date'] <= str(year_range[1])+'-12-31')]

    conn.close()

    return df

def generate_mrc_data(json_data):
    output_items = json_data.split('\n')
    mrc_data = []
    keys_needed = ['pmid', 'abstract', 'article_title', 'article_date', 'journal', 'authors']
    for item in output_items:
        if len(item) == 0:
            continue
        try:
            obj = json.loads(item)
            data_dict = {}
            data_dict['pmid'] = obj['pmid']
            if 'abstract' not in obj['medent'].keys():
                print('No abstract', obj['pmid'])
                continue
            data_dict['abstract'] = obj['medent']['abstract']
            if 'title' in obj['medent']['cit'].keys():
                if 'name' not in obj['medent']['cit']['title'].keys():
                    print('No title', obj['pmid'])
                    continue
                data_dict['article_title'] = obj['medent']['cit']['title']['name']
            else:
                data_dict['article_title'] = '-'
            data_dict['article_date'] = obj['medent']['em_std']['year'] + '-' + obj['medent']['em_std']['month'] + '-' + obj['medent']['em_std']['day']
            if 'authors' in obj['medent']['cit'].keys():
                if 'names_std' in obj['medent']['cit']['authors'].keys():
                    data_dict['authors'] = ','.join([x['nameml'] for x in obj['medent']['cit']['authors']['names_std'] if 'nameml' in x.keys()])
                elif 'names_ml' in obj['medent']['cit']['authors'].keys():
                    data_dict['authors'] = ','.join(obj['medent']['cit']['authors']['names_ml'])
                else:
                    raise Exception('no author names')
            if 'from_journal' in obj['medent']['cit'].keys():
                data_dict['journal'] = obj['medent']['cit']['from_journal']['title']['iso_jta']
            else:
                data_dict['journal'] = obj['medent']['cit']['from_book']['title']['name']
                if 'authors' not in data_dict.keys():
                    if 'names_std' in obj['medent']['cit']['from_book']['authors'].keys():
                        for a in obj['medent']['cit']['from_book']['authors']['names_std']:
                            data_dict['authors'] = ','.join(list(a.values()))
                    elif 'names_ml' in obj['medent']['cit']['from_book']['authors'].keys():
                        data_dict['authors'] = ','.join(obj['medent']['cit']['from_book']['authors']['names_ml'])
                    else:
                        raise Exception('no author names')
            for k in keys_needed:
                if k not in data_dict.keys():
                    data_dict[k] = ''
            mrc_data.append(data_dict)
        except Exception as e:
            print("Unexpected error", obj['pmid'])

    for x in mrc_data:
        x['abstract'] = x['abstract'].replace('\\.', '').replace('\\xAE', '')

    return mrc_data

def sort_keywords(keywords):
    """
    Given a list of keywords as extracted by the function
    keyword_extractor.extract_keywords, return a sorted list according to the
    frequency of each keyword in the PubMed abstracts.
    The lower the frequency of the keyword, the lower its index in the array.
    The frequency of keywords is stored in the PostgreSQL database defined
    above, in the table prod.mrc_keyword_frequency.
    """
    if len(keywords)<2:
        return keywords

    conn = connect_to_postgres_db()
    with conn.cursor() as cur:
        keywords_string = "('" + "','".join(keywords) + "')"
        cur.execute("SELECT keyword, n FROM prod.mrc_keyword_frequency WHERE keyword IN {ks};".format(ks=keywords_string))
        records = cur.fetchall()
        sorted_keywords = [x[0] for x in sorted(records, key=lambda item: int(item[1]))]
    conn.close()

    return sorted_keywords

def pubmed_keyword_search(keywords, nOutputs, year_range):
    """
    Given a set of keywords, return the pmids related to them as given by the
    eutils PubMed API.
    For detailed information on the eutils PubMed API, see
    https://www.ncbi.nlm.nih.gov/books/NBK25499/
    """
    url_base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&usehistory=y'
    params = {
        'term': ' '.join(keywords),
        'sort': 'relevance',
        'field': 'abstract',
        'mindate': str(year_range[0]),
        'maxdate': str(year_range[1]),
        'retmax': str(nOutputs)
    }
    search_url = url_base
    for k,v in params.items():
        search_url += '&' + k + '=' + v

    search_r = requests.post(search_url)
    search_data = search_r.json()
    pmids = search_data['esearchresult']['idlist']

    return pmids

def fetch_pubmed_data_from_pmids(pmids):
    """
    Given a list of pmids, fetch the PubMed data from the eutils API.
    """
    pmids_string = ','.join(pmids)
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=" + pmids_string
    fetch_r = requests.post(fetch_url)
    fetch_data = fetch_r.content
    pubmed_data_string = fetch_data.decode("utf-8")
    with open("Output.txt", "w") as text_file:
        text_file.write(pubmed_data_string)
    json_output = asn1.to_json("Output.txt")
    os.remove("Output.txt")
    data = generate_mrc_data(json_output)
    abstract_df = pd.DataFrame(data)
    return abstract_df

def get_related_pmids(pmid):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&db=pubmed&id=" + pmid + "&cmd=neighbor_score"
    search_r = requests.post(search_url)
    search_data = search_r.json()
    pmids = search_data['esearchresult']['idlist']
    return pmids

def get_abstracts_from_pubmed_2(keywords, nOutputs, year_range):
    def remove_duplicates(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    sorted_keywords = sort_keywords(keywords)
    pmids = []
    while len(pmids)<int(nOutputs) or len(sorted_keywords)!=0:
        new_pmids = pubmed_keyword_search(sorted_keywords, nOutputs, year_range)
        pmids += new_pmids
        pmids = remove_duplicates(pmids)
        sorted_keywords = sorted_keywords[:-1]

    if len(pmids)>int(nOutputs):
        pmids = pmids[:int(nOutputs)]

    if len(pmids)==0:
        print(f"Nothing found with keywords {keywords}")
        return pd.DataFrame()

    if len(pmids) < int(nOutputs):
        new_pmids = get_related_pmids(pmids[0]) # First article is the most relevant one
        pmids += new_pmids
        pmids = remove_duplicates(pmids)

    abstract_df = fetch_pubmed_data_from_pmids(pmids)

    return abstract_df

def get_abstracts_from_pubmed(keywords, nOutputs, year_range):
    url_base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&usehistory=y'
    params = {
        'term': ' '.join(keywords),
        'sort': 'relevance',
        'field': 'abstract',
        'mindate': str(year_range[0]),
        'maxdate': str(year_range[1]),
        'retmax': str(nOutputs)
    }
    search_url = url_base
    for k,v in params.items():
        search_url += '&' + k + '=' + v

    search_r = requests.post(search_url)
    search_data = search_r.json()
    pmids = search_data['esearchresult']['idlist']

    # Comment the next couple of lines if using our postgres DB
    webenv = search_data['esearchresult']['webenv']
    fetch_url_start = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&query_key=1'
    params = {
        'webenv': webenv,
        'retmax': str(nOutputs)
    }
    fetch_url = fetch_url_start
    for k,v in params.items():
        fetch_url += '&' + k + '=' + v
    fetch_r = requests.post(fetch_url)
    fetch_data = fetch_r.content
    pubmed_data_string = fetch_data.decode("utf-8")
    with open("Output.txt", "w") as text_file:
        text_file.write(pubmed_data_string)
    json_output = asn1.to_json("Output.txt")
    os.remove("Output.txt")
    data = generate_mrc_data(json_output)
    abstract_df = pd.DataFrame(data)

    """
    conn = connect_to_postgres_db()
    with conn.cursor() as cur:
        pmids_string = "('" + "','".join(pmids) + "')"
        cur.execute("SELECT * from prod.mrc_data WHERE pmid in {ps}".format(ps=pmids_string))
        records = cur.fetchall()

    df = pd.DataFrame(records, columns=['pmid', 'abstract', 'article_title', 'article_date', 'journal', 'authors'])
    df = df[(df['article_date'] >= str(year_range[0])+'-01-01') & (df['article_date'] <= str(year_range[1])+'-12-31')]
    conn.close()
    """

    return abstract_df

def get_final_answers(df):
    """
    Given the abstracts_df dataframe, return a dictionary such that
    the keys are the answers given by the abstract and the values are the
    final answers to be shown in Mr.C's results table.
    """
    answers = list(df.answer.values)
    unique_answers, answer_counts = np.unique(answers, return_counts=True)
    count_sort_ind = np.argsort(-answer_counts)
    answer_synonyms = {}
    for ans in unique_answers:
        synonyms = wordnet.synsets(ans)
        answer_synonyms[ans] = set(chain.from_iterable([word.lemma_names() for word in synonyms]))

    final_answers = {}
    for i in range(len(count_sort_ind)):
        synonym_found = False
        for j in range(i):
            if not len(answer_synonyms[unique_answers[count_sort_ind[i]]].intersection(answer_synonyms[unique_answers[count_sort_ind[j]]]))==0:
                final_answers[unique_answers[count_sort_ind[i]]] = unique_answers[count_sort_ind[j]]
                synonym_found = True
                break
        if not synonym_found:
            final_answers[unique_answers[count_sort_ind[i]]] = unique_answers[count_sort_ind[i]]

    return final_answers

################
# ACTUAL TASKS #
################

def extract_keywords(question):
    """
    Given a question, this function returns a list of keywords.
    """
    keywords = ky.lemmatize_sentence(question)
    return keywords

def process_question(params):
    user_id = params["user_id"]
    question = params["question"]
    nOutputs = params["nOutputs"]
    min_year = params["min_year"]
    max_year = params["max_year"]
    search_from_pubmed = params["pubmed"]
    if not search_from_pubmed:
        keywords = params["keywords"]

    year_range = (min_year, max_year)

    try:
        start_time = time.time()

        if search_from_pubmed:
            if 'keywords' not in params.keys():
                keywords = ky.lemmatize_sentence(question)
            else:
                keywords = params['keywords']
            db_start_time = time.time()
            abstract_df = get_abstracts_from_pubmed_2(keywords, nOutputs, year_range)
            db_end_time = time.time()
            db_time = db_end_time - db_start_time
        else:
            if len(keywords)<=1: # If keywords not specified in URL
                keywords = ky.lemmatize_sentence(question)
            else:
                keywords = keywords.split(' ')

            db_start_time = time.time()
            abstract_df = get_abstracts_from_postgres(keywords, nOutputs, year_range)
            db_end_time = time.time()
            db_time = db_end_time - db_start_time

        total_n_articles = len(abstract_df)
        abstract_df = abstract_df.sample(min(int(nOutputs), total_n_articles), random_state=1)

        abstract_df['answer'] = ''
        abstract_df['start_score'] = ''
        abstract_df['end_score'] = ''

        # Calculate the answer for every abstract using the NLP DL model
        for index, row in abstract_df.iterrows():
            text = row['abstract']
            inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]

            n_tokens = len(input_ids)
            input_length = min(n_tokens, 512)

            inputs = {k:v[:,:input_length] for k,v in inputs.items()}
            answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores)
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end+1]))

            row['answer'] = answer
            row['start_score'] = answer_start_scores[0][answer_start].item()
            row['end_score'] = answer_end_scores[0][answer_end].item()

        end_time = time.time()
        total_time = end_time - start_time

        if LOG_DATA:
            log_data(user_id, question, keywords, nOutputs, min_year, max_year, abstract_df, total_n_articles, total_time, db_time)

        # Make sure that the answer is valid. Otherwise, drop row.
        abstract_df = abstract_df[abstract_df['answer'].apply(lambda x: answer_is_valid(x))]
        abstract_df = abstract_df[abstract_df['start_score']+abstract_df['end_score']>3]
        abstract_df.reset_index(drop=True, inplace=True)

        if len(abstract_df)==0:
            return '{}'

        final_answers = get_final_answers(abstract_df)
        abstract_df['final_answer'] = ''
        for index, row in abstract_df.iterrows():
            row['final_answer'] = final_answers[row['answer']]

        result_json = abstract_df.to_json()
        keywords = ' '.join(keywords)
        result_json = result_json[:-1]+',"total_n_articles":'+str(total_n_articles)+',"keywords":"'+keywords+'"}'
    except:
        error_message = traceback.format_exc()
        log_error(user_id, question, keywords, nOutputs, min_year, max_year, error_message)

    return result_json
