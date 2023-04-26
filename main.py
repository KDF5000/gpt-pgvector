#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import signal
import re
import time
import json

import openai
import psycopg2
import tiktoken

from retrying import retry
from pyrate_limiter import (Duration, RequestRate, Limiter)

openai.api_key = os.getenv("OPENAI_API_KEY")
pg_con = None

rate = RequestRate(20, Duration.MINUTE)
limiter = Limiter(rate)

system_content = """
 `You are a helpful assistant. When given CONTEXT you answer questions using only that information,
  and you always format your output in markdown. You include code snippets if relevant. If you are unsure and the answer
  is not explicitly written in the CONTEXT provided, you say "Sorry, I don't know how to help with that."  If the CONTEXT includes
  source URLs include them under a SOURCES heading at the end of your response. Always include all of the relevant source urls
  from the CONTEXT, but never list a URL more than once (ignore trailing forward slashes when comparing for uniqueness).
  Never include URLs that are not in the CONTEXT sections. Never make up URLs`
"""

user_content = """CONTEXT:
  Next.js is a React framework for creating production-ready web applications. It provides a variety of methods for fetching data, a built-in router, and a Next.js Compiler for transforming and minifying JavaScript code. It also includes a built-in Image Component and Automatic Image Optimization for resizing, optimizing, and serving images in modern formats.
  SOURCE: nextjs.org/docs/faq
  
  QUESTION: 
  what is nextjs?    
"""

assistant_content = """
Next.js is a framework for building production-ready web applications using React. It offers various data fetching options, comes equipped with an integrated router, and features a Next.js compiler for transforming and minifying JavaScript. Additionally, it has an inbuilt Image Component and Automatic Image Optimization that helps resize, optimize, and deliver images in modern formats.
```js
function HomePage() {
  return <div>Welcome to Next.js!</div>
}

export default HomePage
```

SOURCES:
https://nextjs.org/docs/faq`;
"""

user_message = """CONTEXT:
{} 
USER QUESTION:
{}
"""

def num_tokens_from_string(string, encoding_name="cl100k_base"):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def connect(db, host, user, password):
    con = None
    try:
        con = psycopg2.connect(database=db, host=host, user=user, password=password)
    except psycopg2.DatabaseError as e:
        print(f'Error {e}')
        sys.exit(1)
    return con

@retry(stop_max_attempt_number=10)
@limiter.ratelimit('OPENAI_API', delay=True)
def get_embedding(text, model="text-embedding-ada-002"):
    # print("[{}] start to gen embedding".format(time.time()))
    text = text.replace("\n", " ")
    res = openai.Embedding.create(input = [text], model=model)
    # print("[{}] generate embedding succ!".format(time.time()))
    #print(res)
    if res:
        return res['data'][0]['embedding']
    return []

def create_embedding(db, content, url, embedding):
    sql = """INSERT INTO documents(content, url, embedding) 
             VALUES(%s, %s, %s) RETURNING id;"""
    # print(sql)
    id = None
    if not db:
        return None 
    try:
        cur = db.cursor()
        # execute the INSERT statement
        cur.execute(sql, (content.replace('\x00', ''), url, embedding))
        # get the generated id back
        id = cur.fetchone()[0]
        # commit the changes to the database
        db.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return id

def search_embedding(db, embedding, limit=10, similarity_threshold=0.1):
    # <-> : L2 distance
    # <#> : inner product, returns the negative inner product since Postgres only supports ASC order index scans on operators
    # <=> : cosine distance
    #sql = "SELECT content, url FROM documents ORDER BY embedding <-> '{}' LIMIT {};".format(embedding, limit)
    sql = "SELECT * FROM match_documents('{}', {}, {});".format(embedding, similarity_threshold, limit)
    data = []
    try:
        cur = db.cursor()
        # execute the INSERT statement
        cur.execute(sql)
        # get the generated id back
        rows = cur.fetchall()
        for row in rows:
            data.append(row)
        # commit the changes to the database
        db.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return data

def gen_context(refs, max_token=3000):
    if refs == None or len(refs) == 0:
        return None

    context_text = ""
    token_count = 0
    for doc in refs:
        token_count = token_count + num_tokens_from_string(doc[1]);
        if token_count > max_token:
            break
        if doc[2] != "":
            context_text = "{}{}\nSOURCE: {}\n---\n".format(context_text, doc[1], doc[2])
        else:
            context_text = "{}{}\n---\n".format(context_text, doc[1])
    return context_text

@retry(stop_max_attempt_number=10)
@limiter.ratelimit('OPENAI_API', delay=True)
def get_answer(context, question):
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": user_message.format(context, question)}
    ]

    #print(json.dumps(messages))
    ret = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    return ret['choices'][0]['message']['content']

def gen_vector_from_file(db, filepath, chunk_size=2000):
    chunk_content = ""
    content_size = 0
    total_size = 0
    call_api_cnt = 0
    for line in open(filepath):
        str = re.sub(r'\s+', ' ', line.strip())
        size = num_tokens_from_string(str)
        total_size += size
        if content_size + size < chunk_size:
            content_size += size
            chunk_content += str
            continue

        # generate embedding
        # print(chunk_content)
        call_api_cnt += 1
        embedding = get_embedding(chunk_content)
        if len(embedding) == 0:
            print("empty embedding!")
            return

        id = create_embedding(db, chunk_content, "", embedding)
        if not id:
          print("failed to create embedding")
          return 
        content_size = size
        chunk_content = str
        # sleep 300ms
    if content_size > 0:
        call_api_cnt += 1
        #generate embedding
        #print(chunk_content)
        embedding = get_embedding(chunk_content)
        id = create_embedding(db, chunk_content, "", embedding)
        if not id:
            print("failed to create embedding")
            return 
    print("total token cost: %d\n call api count: %d"%(total_size, call_api_cnt))

def answer(db, question):
    embedding = get_embedding(question)
    data = search_embedding(db, embedding)
    context = gen_context(data)
    print(" > ##Context## %s"%context)
    return get_answer(context, question)

def _exit(sig, frame):
    if pg_con:
        pg_con.close()
    print("\nBye")
    exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./doc_gpt gen file_path or ./doc_gpt chat")
        exit(1)
    
    signal.signal(signal.SIGINT, _exit)
    signal.signal(signal.SIGTERM, _exit)

    cmd = sys.argv[1]
    if cmd == "gen":
        if len(sys.argv) < 3:
            print("Usage: ./doc_gpt gen file_path")
            exit(1)
        filepath = sys.argv[2]
        if not os.path.exists(filepath):
            print("file [%s] not exist"%filepath)
            exit(1)
        pg_con = connect(db="postgres", host="127.0.0.1", user="root", password="root123")
        gen_vector_from_file(pg_con, filepath)
        pg_con.close()
    elif cmd == "chat":
        pg_con = connect(db="postgres", host="127.0.0.1", user="root", password="root123")
        while True:
            q = input("Q# ")
            ans = answer(pg_con, q)
            print("A# %s"%ans)
