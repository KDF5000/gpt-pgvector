#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

import openai
import psycopg2
import tiktoken

openai.api_key = os.getenv("OPENAI_API_KEY")

system_content = """
 `You are a helpful assistant. When given CONTEXT you answer questions using only that information,
  and you always format your output in markdown. You include code snippets if relevant. If you are unsure and the answer
  is not explicitly written in the CONTEXT provided, you say "Sorry, I don't know how to help with that."  If the CONTEXT includes
  source URLs include them under a SOURCES heading at the end of your response. Always include all of the relevant source urls
  from the CONTEXT, but never list a URL more than once (ignore trailing forward slashes when comparing for uniqueness).
  Never include URLs that are not in the CONTEXT sections. Never make up URLs`;
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

def connect(db, user, password):
    con = None
    try:
        con = psycopg2.connect(database=db, user=user, password=password)
    except psycopg2.DatabaseError as e:
        print(f'Error {e}')
        sys.exit(1)
    return con

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   res = openai.Embedding.create(input = [text], model=model)
   #print(res)
   if res:
       return res['data'][0]['embedding']
   return []

def create_embedding(db, content, url, embedding):
    sql = """INSERT INTO documents(content, url, embedding) 
             VALUES(%s, %s, %s) RETURNING id;"""
    print(sql)
    id = None
    if not db:
        return None 
    try:
        cur = db.cursor()
        # execute the INSERT statement
        cur.execute(sql, (content, url, embedding))
        # get the generated id back
        id = cur.fetchone()[0]
        # commit the changes to the database
        db.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return id

def search_embedding(db, embedding, limit=10):
    # <-> : L2 distance
    # <#> : inner product, returns the negative inner product since Postgres only supports ASC order index scans on operators
    # <=> : cosine distance
    sql = "SELECT content, url FROM documents ORDER BY embedding <-> '{}' LIMIT {};".format(embedding, limit)
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

def gen_context(refs, max_token=1500):
    context_text = ""
    token_count = 0
    for doc in refs:
        token_count = token_count + num_tokens_from_string(doc[0]);
        if token_count > max_token:
            break
        context_text = "{}{}\n---\n".format(context_text, doc[0])
    return context_text

def get_answer(context, question):
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": user_message.format(context, question)}
    ]

    ret = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    return ret['choices'][0]['message']['content']

if __name__ == "__main__":
    con = connect(db="postgres", user="root", password="root123")
    text = '''
    In all-stop mode, whenever your program stops under GDB for any reason, all threads of execution stop, not just the current thread. This allows you to examine the overall state of the program, including switching between threads, without worrying that things may change underfoot.

Conversely, whenever you restart the program, all threads start executing. This is true even when single-stepping with commands like step or next.

In particular, GDB cannot single-step all threads in lockstep. Since thread scheduling is up to your debugging target’s operating system (not controlled by GDB), other threads may execute more than one statement while the current thread completes a single step. Moreover, in general other threads stop in the middle of a statement, rather than at a clean statement boundary, when the program stops.

You might even find your program stopped in another thread after continuing or even single-stepping. This happens whenever some other thread runs into a breakpoint, a signal, or an exception before the first thread completes whatever you requested.

Whenever GDB stops your program, due to a breakpoint or a signal, it automatically selects the thread where that breakpoint or signal happened. GDB alerts you to the context switch with a message such as ‘[Switching to Thread n]’ to identify the thread.
'''
    embedding = get_embedding(text)
    #print(embedding)
    #id = create_embedding(con, text, "", embedding)
    #print(id)
    data = search_embedding(con, embedding)
    #print(data)
    context = gen_context(data)
    #print(context)
    ans = get_answer(context, "what is gdb all-stop mode")
    print(ans)

    con.close()
