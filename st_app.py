import streamlit as st
import requests
import json
import pandas as pd
import os

fpath = os.getcwd()
file_path_split = fpath.split("\\")
level = ""
for _path in file_path_split:
    level+=_path+"\\"

def process_file(file_path):
    file_path = file_path.replace("/", "\\")
    df = pd.read_excel(f"{level}{file_path}", engine="openpyxl")
    return df.to_csv().encode("UTF-8")


def title():
    st.markdown("""
            <h1 style='text-align: left; font-size: 40px;'>
                Natural Language to SQL Converter
            </h1>
        """, unsafe_allow_html=True)


def process_nlp_text():
    st.header("Natural Language Query")
    # user_query = st.text_input("Enter your NLP text query." )
    user_query = st.text_area("""Enter your NLP text query.""" )
    generate_sql_button_clicked = st.button("Generate SQL", key="generate_sql_button_clicked")
    if generate_sql_button_clicked:
        if not user_query:
            st.error("Please enter NLP text query.")
        else:
            with st.spinner("Generating SQL. Please wait.."):
                response = requests.post(url="http://localhost:8000/process-user-query/",
                                         data=json.dumps({"query": user_query}))
                if response.status_code != 200:
                    st.error(response.text)
                elif response.status_code == 200:
                    resp_dict = json.loads(response.text)
                    st.success(f"{resp_dict['message']}.\nPlease review below GEN AI generated SQL query.")
                    return resp_dict


def process_sql_query(sql_query):
    st.header("Review generated SQL Query")
    sql_user_query = st.text_area("GEN AI generated SQL Query\n", f"{sql_query}")
    generate_sql_result = st.button("Generate Result", key = "generate_sql_result")
    if generate_sql_result:
        with st.spinner("Please wait. Processing sql query."):
            if sql_user_query:
                sql_qry_result_response = requests.post(url="http://localhost:8000/process-visy-sql-query-result/",
                                                        data=json.dumps({"query": sql_user_query}))
                if sql_qry_result_response.status_code != 200:
                    st.error(sql_qry_result_response.text)
                elif sql_qry_result_response.status_code == 200:
                    resp_qry_dict = json.loads(sql_qry_result_response.text)
                    df = pd.DataFrame(json.loads(resp_qry_dict['data']))
                    st.dataframe(df)
                    down_file = process_file(resp_qry_dict['file_path'])
                    st.download_button(label="Download data as csv", data=down_file, file_name="sample-bt.csv",
                                       mime="text/csv")
                    st.success("Result generated successfully.")
            else:
                st.error("Please provide SQL query to process.")



if __name__ == "__main__":
    title()
    data_dict = process_nlp_text()
    if data_dict:
        process_sql_query(data_dict['sql_query'])