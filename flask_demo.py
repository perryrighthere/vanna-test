from vanna.openai import OpenAI_Chat
from vanna.qdrant import Qdrant_VectorStore
from qdrant_client import QdrantClient
import httpx
from openai import OpenAI
import pandas as pd


client = OpenAI(
    base_url="https://aiapi.test.seres.cn/v1",
    api_key="sk-gdKNRLPALlfgJO2B10iv42fJC5NnEpPDw9MbseYCGCSYSgMC",
    http_client=httpx.Client(verify=False)
)

# Qdrant client setup
qdrant = QdrantClient(url="http://localhost:6333")

# Vanna class combining both backends
class MyVanna(Qdrant_VectorStore, OpenAI_Chat):
    def __init__(self, client=None, config=None):
        Qdrant_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, client=client, config=config)

# Initialize Vanna
vn = MyVanna(client=client, config={'client': qdrant, 'model': 'qwen2.5-instruct', 'language': 'zh'})

# Connect to MySQL database
vn.connect_to_mysql(
    host='srtest.seres.cn', 
    dbname='doris_mcp_poc', 
    user='doris_mcp_u', 
    password='zjjKtmu2iDlNCnS0', 
    port=9030
)

from vanna.flask import VannaFlaskApp
app = VannaFlaskApp(vn=vn, allow_llm_to_see_data=True, ask_results_correct=False)
app.run()