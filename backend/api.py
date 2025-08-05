import json
import logging
from typing import List
from fastapi import FastAPI, UploadFile
from fastapi.routing import APIRouter
from fastapi.exceptions import HTTPException
from backend.models import QueryRequest, QueryResponse, UserQuery
from backend.utils import generate_result, create_embeddings,\
    similarity_search, create_prompt, generate_sql_using_gcp_llm_gemini_flash


logger = logging.getLogger(__name__)
app = FastAPI()

process_user_query = APIRouter(prefix="/process-user-query")


@process_user_query.post("/")
async def generate_latest_embedding_and_get_sql(request: UserQuery):
    user_query = request.query
    # user_query = "How many parcel  requests were raised by parcel_master"
    try:
        if user_query:
            with open("data/metadata.json", "r") as file_data:
                metadata_json = json.load(file_data)
            v_embedding = create_embeddings(metadata_json)
            relevant_table_names = similarity_search(user_query,v_embedding)
            relevant_tables = [table for table in metadata_json if table['table_name'] in relevant_table_names]
            prompt = create_prompt(user_query, relevant_tables)
            sql_query_dict = generate_sql_using_gcp_llm_gemini_flash(prompt)
            # sql_query_dict = get_llm_response(prompt)
            if sql_query_dict['success']:
                # return {"message": "SQL Query generated successfully.",
                #         "sql_query": sql_query_dict["data"]
                #         }
                sql_result_flag, file_path, data, sql_qry = generate_result(sql_query_dict["data"],"v.db")
                if sql_result_flag:
                    return {"message": "Result generated successfully.",
                            "data": data,
                            "file_saved_path": file_path,
                            "sql_query": sql_query_dict["data"]}
                else:
                    return {"message": "SQL expression generated successfully. Error while pulling data from table.",
                            "sql_query": sql_query_dict["data"],
                            "file_saved_path": "",
                            "data": {}
                            }
            else:
                raise HTTPException(status_code=500, detail=f"GEN-AI unable to convert given text data to SQL QUery. Error msg: {sql_query_dict['data']}")
        else:
            raise IndexError
    except IndexError:
        raise HTTPException(status_code=400,detail="No user query found. Please send user query and try again.")