# #!/usr/bin/env python3
# """
# Demonstration of Using Trained Vanna Model

# This script shows how to use a trained Vanna model for:
# 1. Natural language to SQL generation
# 2. Query execution and result analysis
# 3. Follow-up question generation
# 4. Interactive querying workflow
# 5. Model performance evaluation
# """

# from vanna.openai import OpenAI_Chat
# from vanna.qdrant import Qdrant_VectorStore
# from qdrant_client import QdrantClient
# import httpx
# from openai import OpenAI
# import pandas as pd
# import time
# from typing import List, Tuple, Optional


# def setup_trained_vanna():
#     """Set up connection to already trained Vanna instance"""
#     try:
#         # OpenAI client setup
#         client = OpenAI(
#             base_url="http://aiapi-api.test.seres.cn/v1",
#             api_key="sk-gdKNRLPALlfgJO2B10iv42fJC5NnEpPDw9MbseYCGCSYSgMC",
#             http_client=httpx.Client(verify=False)
#         )
        
#         # Qdrant client setup (should contain trained data)
#         qdrant = QdrantClient(url="http://localhost:6333")
        
#         # Vanna class combining both backends
#         class MyVanna(Qdrant_VectorStore, OpenAI_Chat):
#             def __init__(self, client=None, config=None):
#                 Qdrant_VectorStore.__init__(self, config=config)
#                 OpenAI_Chat.__init__(self, client=client, config=config)
        
#         # Initialize Vanna with trained data
#         vn = MyVanna(client=client, config={'client': qdrant, 'model': 'qwen2.5-instruct'})
        
#         # Connect to MySQL database
#         vn.connect_to_mysql(
#             host='srtest.seres.cn', 
#             dbname='doris_mcp_poc', 
#             user='doris_mcp_u', 
#             password='zjjKtmu2iDlNCnS0', 
#             port=9030
#         )
        
#         print("✅ Connected to trained Vanna instance")
#         return vn
        
#     except Exception as e:
#         print(f"❌ Error connecting to trained Vanna: {e}")
#         return None


# def demonstrate_sql_generation(vn):
#     """Demonstrate natural language to SQL generation"""
#     print("\n🔧 SQL GENERATION FROM NATURAL LANGUAGE")
#     print("=" * 50)
    
#     # Test questions covering different query types
#     test_questions = [
#         # Basic counting and filtering
#         "How many employees are there in total?",
#         "How many active employees do we have?",
#         "How many key employees are in the system?",
        
#         # Grouping and aggregation
#         "What is the gender distribution of employees?",
#         "Show me the age distribution by company",
#         "Count employees by marital status",
        
#         # Filtering with Chinese text
#         "Find all employees from 重庆市",
#         "Show me employees from 四川省", 
#         "List all married employees",
        
#         # Complex analytical queries
#         "What is the average age by organization?",
#         "Find the youngest employee in each company",
#         "Show me employees with complete height and weight data",
        
#         # Management and hierarchy
#         "List all management positions (cadre_label = 1)",
#         "Find core position employees who are also key employees",
#         "Show nationality distribution"
#     ]
    
#     sql_results = []
    
#     for i, question in enumerate(test_questions, 1):
#         print(f"\n[{i:2d}] Question: {question}")
#         try:
#             start_time = time.time()
#             sql = vn.generate_sql(question)
#             generation_time = time.time() - start_time
            
#             print(f"     Generated SQL: {sql}")
#             print(f"     Generation time: {generation_time:.2f}s")
            
#             sql_results.append({
#                 'question': question,
#                 'sql': sql,
#                 'generation_time': generation_time,
#                 'status': 'success'
#             })
            
#         except Exception as e:
#             print(f"     ❌ Error: {e}")
#             sql_results.append({
#                 'question': question,
#                 'sql': None,
#                 'generation_time': 0,
#                 'status': 'failed',
#                 'error': str(e)
#             })
    
#     return sql_results


# def demonstrate_query_execution(vn, sql_results: List[dict]):
#     """Demonstrate executing generated SQL and analyzing results"""
#     print("\n📊 SQL EXECUTION AND RESULT ANALYSIS")
#     print("=" * 50)
    
#     executed_results = []
    
#     for result in sql_results[:8]:  # Execute first 8 successful queries
#         if result['status'] != 'success' or not result['sql']:
#             continue
            
#         print(f"\n🔍 Executing: {result['question']}")
#         print(f"   SQL: {result['sql']}")
        
#         try:
#             start_time = time.time()
#             df = vn.run_sql(result['sql'])
#             execution_time = time.time() - start_time
            
#             if df is not None and len(df) > 0:
#                 print(f"   ✅ Results: {len(df)} rows returned in {execution_time:.2f}s")
#                 print(f"   Preview:")
#                 print(df.head().to_string(index=False))
                
#                 executed_results.append({
#                     'question': result['question'],
#                     'sql': result['sql'],
#                     'row_count': len(df),
#                     'execution_time': execution_time,
#                     'data': df
#                 })
#             else:
#                 print("   ⚠️  No results returned")
                
#         except Exception as e:
#             print(f"   ❌ Execution error: {e}")
    
#     return executed_results


# def demonstrate_ask_workflow(vn):
#     """Demonstrate the complete ask() workflow"""
#     print("\n💬 COMPLETE ASK() WORKFLOW DEMONSTRATION")
#     print("=" * 50)
    
#     ask_questions = [
#         "How many employees are in each company?",
#         "What's the gender breakdown?",
#         "Show me key employees with their details",
#         "Find employees older than 35"
#     ]
    
#     for question in ask_questions:
#         print(f"\n🤖 Asking: '{question}'")
#         print("-" * 60)
        
#         try:
#             # Use the complete ask() method
#             start_time = time.time()
#             result = vn.ask(
#                 question=question,
#                 print_results=False,  # We'll handle printing ourselves
#                 auto_train=True,      # Automatically learn from successful queries
#                 visualize=False       # Skip visualization for this demo
#             )
#             total_time = time.time() - start_time
            
#             if result:
#                 sql, df, plot = result
#                 print(f"Generated SQL: {sql}")
#                 print(f"Total processing time: {total_time:.2f}s")
                
#                 if df is not None and len(df) > 0:
#                     print(f"Results ({len(df)} rows):")
#                     print(df.to_string(index=False))
                    
#                     # Generate follow-up questions
#                     try:
#                         followup_questions = vn.generate_followup_questions(
#                             question=question,
#                             sql=sql,
#                             df=df,
#                             n_questions=3
#                         )
#                         print(f"\nSuggested follow-up questions:")
#                         for i, followup in enumerate(followup_questions[:3], 1):
#                             if followup.strip():
#                                 print(f"  {i}. {followup.strip()}")
#                     except Exception as e:
#                         print(f"Could not generate follow-up questions: {e}")
#                 else:
#                     print("No results returned")
#             else:
#                 print("❌ Ask failed - no result returned")
                
#         except Exception as e:
#             print(f"❌ Error in ask workflow: {e}")


# def demonstrate_training_data_usage(vn):
#     """Show how training data influences query generation"""
#     print("\n📚 TRAINING DATA INFLUENCE DEMONSTRATION")
#     print("=" * 50)
    
#     # Check current training data
#     try:
#         training_data = vn.get_training_data()
#         print(f"Current training data: {len(training_data)} items")
#         if len(training_data) > 0:
#             print("Training data types:")
#             if 'content' in training_data.columns:
#                 for _, row in training_data.head(5).iterrows():
#                     content = str(row['content'])[:100] + "..." if len(str(row['content'])) > 100 else str(row['content'])
#                     print(f"  - {content}")
#     except Exception as e:
#         print(f"Could not retrieve training data: {e}")
    
#     # Test similar question retrieval
#     print(f"\n🔍 Testing context retrieval for questions:")
#     test_questions = [
#         "Show me employee count by company",
#         "What is the age distribution?", 
#         "Find key employees"
#     ]
    
#     for question in test_questions:
#         print(f"\nQuestion: {question}")
#         try:
#             # Get similar questions from training data
#             similar_questions = vn.get_similar_question_sql(question=question)
#             if len(similar_questions) > 0:
#                 print("Similar training examples found:")
#                 for _, row in similar_questions.head(2).iterrows():
#                     print(f"  - Question: {row.get('question', 'N/A')}")
#                     sql_preview = str(row.get('content', ''))[:80] + "..." if len(str(row.get('content', ''))) > 80 else str(row.get('content', ''))
#                     print(f"    SQL: {sql_preview}")
#             else:
#                 print("  No similar examples found")
#         except Exception as e:
#             print(f"  Error retrieving similar questions: {e}")


# def analyze_model_performance(sql_results: List[dict], executed_results: List[dict]):
#     """Analyze the performance of the trained model"""
#     print("\n📈 MODEL PERFORMANCE ANALYSIS")
#     print("=" * 50)
    
#     total_questions = len(sql_results)
#     successful_generation = len([r for r in sql_results if r['status'] == 'success'])
#     successful_execution = len(executed_results)
    
#     print(f"SQL Generation Performance:")
#     print(f"  Total questions: {total_questions}")
#     print(f"  Successful generations: {successful_generation} ({successful_generation/total_questions*100:.1f}%)")
#     print(f"  Successful executions: {successful_execution}")
    
#     if successful_generation > 0:
#         avg_generation_time = sum(r['generation_time'] for r in sql_results if r['status'] == 'success') / successful_generation
#         print(f"  Average generation time: {avg_generation_time:.2f}s")
    
#     if executed_results:
#         avg_execution_time = sum(r['execution_time'] for r in executed_results) / len(executed_results)
#         total_rows = sum(r['row_count'] for r in executed_results)
#         print(f"  Average execution time: {avg_execution_time:.2f}s")
#         print(f"  Total rows returned: {total_rows}")
    
#     print(f"\nQuery Type Analysis:")
#     query_types = {
#         'Count queries': len([r for r in sql_results if r['status'] == 'success' and 'COUNT' in str(r['sql']).upper()]),
#         'Group by queries': len([r for r in sql_results if r['status'] == 'success' and 'GROUP BY' in str(r['sql']).upper()]),
#         'Filter queries': len([r for r in sql_results if r['status'] == 'success' and 'WHERE' in str(r['sql']).upper()]),
#         'Complex queries': len([r for r in sql_results if r['status'] == 'success' and ('AVG' in str(r['sql']).upper() or 'MIN' in str(r['sql']).upper() or 'MAX' in str(r['sql']).upper())])
#     }
    
#     for query_type, count in query_types.items():
#         print(f"  {query_type}: {count}")


# def interactive_demo():
#     """Interactive demo where user can ask questions"""
#     print("\n💻 INTERACTIVE DEMO")
#     print("=" * 50)
#     print("Enter questions in natural language (type 'quit' to exit)")
#     print("Examples:")
#     print("  - How many employees are there?")
#     print("  - Show me the gender distribution")
#     print("  - Find key employees from 重庆市")
    
#     vn = setup_trained_vanna()
#     if not vn:
#         print("❌ Could not connect to trained Vanna")
#         return
    
#     while True:
#         try:
#             question = input("\n🤔 Your question: ").strip()
#             if question.lower() in ['quit', 'exit', 'q']:
#                 break
                
#             if not question:
#                 continue
                
#             print(f"\n🤖 Processing: '{question}'")
            
#             # Generate SQL
#             sql = vn.generate_sql(question)
#             print(f"Generated SQL: {sql}")
            
#             # Execute and show results
#             df = vn.run_sql(sql)
#             if df is not None and len(df) > 0:
#                 print(f"\nResults ({len(df)} rows):")
#                 if len(df) > 10:
#                     print(df.head(10).to_string(index=False))
#                     print(f"... and {len(df) - 10} more rows")
#                 else:
#                     print(df.to_string(index=False))
#             else:
#                 print("No results found")
                
#         except KeyboardInterrupt:
#             break
#         except Exception as e:
#             print(f"❌ Error: {e}")
    
#     print("\n👋 Thanks for trying the interactive demo!")


# if __name__ == "__main__":
#     print("🚀 TRAINED VANNA USAGE DEMONSTRATION")
#     print("=" * 60)
    
#     # Setup connection to trained model
#     vn = setup_trained_vanna()
#     training_data = vn.get_training_data()
#     print(f"Found {len(training_data)} training items")
    
#     if not vn:
#         print("❌ Cannot proceed without trained Vanna connection")
#         print("Please run investigate_vanna_training.py first to train the model")
#         exit(1)
    
#     try:
#         # 1. Demonstrate SQL generation capabilities
#         sql_results = demonstrate_sql_generation(vn)
        
#         # 2. Execute some queries and analyze results
#         executed_results = demonstrate_query_execution(vn, sql_results)
        
#         # 3. Show complete ask() workflow
#         demonstrate_ask_workflow(vn)
        
#         # 4. Show training data influence
#         demonstrate_training_data_usage(vn)
        
#         # 5. Analyze model performance
#         analyze_model_performance(sql_results, executed_results)
        
#         # 6. Interactive demo (optional - uncomment to try)
#         # interactive_demo()
        
#         print("\n\n✨ DEMONSTRATION COMPLETE!")
#         print("Key takeaways:")
#         print("✅ Trained Vanna can generate SQL from natural language")
#         print("✅ Training data significantly improves accuracy")
#         print("✅ Model handles Chinese text and business context")
#         print("✅ Complete workflow: question → SQL → results → insights")
#         print("✅ Auto-training improves model over time")
        
#     except Exception as e:
#         print(f"\n❌ Demo failed: {e}")
#         print("Make sure you have:")
#         print("  - Run the training script first")
#         print("  - Qdrant server running with training data")
#         print("  - Valid database connection")

#!/usr/bin/env python3
"""
Investigation of Vanna Training Capabilities with Employee Data

This script demonstrates how to train Vanna with different types of data using real connections:
1. DDL statements (schema definitions)
2. Documentation (business context)
3. Question-SQL pairs (examples)
4. Training data management operations
"""

from vanna.openai import OpenAI_Chat
from vanna.qdrant import Qdrant_VectorStore
from qdrant_client import QdrantClient
import httpx
from openai import OpenAI
import pandas as pd


def setup_vanna_connection():
    """Set up Vanna with OpenAI and Qdrant backends"""
    try:
        # OpenAI client setup
        client = OpenAI(
            base_url="https://aiapi.test.seres.cn/v1",
            api_key="sk-KzbWXgnY22sSBtkSNVSOscVH1SVIZCLTgoGzrqnnjKEQflqA",
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
        vn = MyVanna(client=client, config={'client': qdrant, 'model': 'DeepSeek-V3-0324'})
        
        # Connect to MySQL database
        vn.connect_to_mysql(
            host='srtest.seres.cn', 
            dbname='doris_mcp_poc', 
            user='doris_mcp_u', 
            password='zjjKtmu2iDlNCnS0', 
            port=9030
        )
        
        print("✅ Successfully connected to Vanna with OpenAI + Qdrant + MySQL")
        return vn
        
    except Exception as e:
        print(f"❌ Error setting up Vanna connection: {e}")
        print("Note: This requires running Qdrant locally and valid API keys")
        return None


def display_training_data_summary(vn):
    """Display current training data in Qdrant"""
    try:
        training_data = vn.get_training_data()
        if len(training_data) > 0:
            print(f"\n📊 Current training data: {len(training_data)} items")
            print(training_data.head(10))
        else:
            print("\n📊 No training data found")
        return training_data
    except Exception as e:
        print(f"❌ Error retrieving training data: {e}")
        return pd.DataFrame()


def demonstrate_employee_data_training(vn):
    """Demonstrate training Vanna with the employee data schema and related information"""
    
    print("🔍 INVESTIGATING VANNA TRAINING CAPABILITIES")
    print("=" * 50)
    
    # Show initial training data state
    display_training_data_summary(vn)
    
    print("\n1. TRAINING WITH DDL STATEMENTS")
    print("-" * 30)
    
    # BI dataset table DDL based on the actual schema
    bi_dataset_ddl = """
    CREATE TABLE bi_dataset (
        person_code VARCHAR(65533) NULL,
        position_code VARCHAR(65533) NOT NULL,
        position_name VARCHAR(65533) NOT NULL,
        organization_code VARCHAR(65533) NOT NULL,
        organization_name VARCHAR(65533) NOT NULL,
        empl_status VARCHAR(65533) NOT NULL,
        company_code VARCHAR(65533) NOT NULL,
        company_name VARCHAR(65533) NOT NULL,
        sex VARCHAR(65533) NOT NULL,
        height VARCHAR(65533) NOT NULL,
        weight VARCHAR(65533) NOT NULL,
        date_of_birth VARCHAR(65533) NOT NULL,
        age VARCHAR(65533) NOT NULL,
        key_employee VARCHAR(65533) NOT NULL,
        core_position VARCHAR(65533) NOT NULL,
        cadre_label VARCHAR(65533) NOT NULL,
        expert_label VARCHAR(65533) NOT NULL,
        native_place VARCHAR(65533) NOT NULL,
        nation VARCHAR(65533) NOT NULL,
        marital_status_name VARCHAR(65533) NOT NULL,
        PRIMARY KEY (person_code)
    );
    """
    
    # Marketing knowledge table DDL
    marketing_knowledge_ddl = """
    CREATE TABLE marketing_knowledge (
        id VARCHAR(50) NULL,
        primary_category VARCHAR(50) NOT NULL,
        secondary_category VARCHAR(50) NULL,
        main_name VARCHAR(100) NULL,
        question VARCHAR(65533) NOT NULL,
        answer VARCHAR(65533) NOT NULL,
        A VARCHAR(65533) NULL,
        B VARCHAR(65533) NULL,
        C VARCHAR(65533) NULL,
        D VARCHAR(65533) NULL,
        answer_choose VARCHAR(5) NULL
    );
    """
    
    try:
        bi_ddl_id = vn.train(ddl=bi_dataset_ddl)
        print(f"✅ Successfully added bi_dataset DDL training data: {bi_ddl_id}")
        
        mk_ddl_id = vn.train(ddl=marketing_knowledge_ddl)
        print(f"✅ Successfully added marketing_knowledge DDL training data: {mk_ddl_id}")
    except Exception as e:
        print(f"❌ Error adding DDL: {e}")
    
    print("\n2. TRAINING WITH DOCUMENTATION")
    print("-" * 30)
    
    # 基于实际数据的业务背景文档
    bi_dataset_documentation = [
        """
        BI数据集数据字典：
        - person_code: 员工唯一标识符（如：16385, 16388）
        - position_name: 岗位名称，以中文显示（如：岗位56026, 岗位74842）
        - organization_name: 部门名称，以中文显示（如：部门63750, 部门63616）
        - company_name: 公司/管控单位名称（如：管控单位2017, 管控单位2030）
        - empl_status: 员工状态，"在职"表示在职状态
        - sex: 性别 - "男"（男性）或"女"（女性）
        - height: 身高，以厘米为单位，可能为空
        - weight: 体重，以公斤为单位，可能为空
        - date_of_birth: 出生日期，格式为YYYY-MM-DD
        - age: 年龄，由出生日期计算得出
        - native_place: 籍贯（如：重庆市, 四川省）
        - nation: 民族（如：汉族, 土家族）
        - marital_status_name: 婚姻状况 - "已婚"或"未婚"
        """,
        """
        BI数据集业务规则：
        - key_employee: 关键员工标识，0=普通员工，1=关键员工
        - core_position: 核心岗位标识，0=非核心岗位，1=核心岗位
        - cadre_label: 干部标识，0=非管理岗，1=管理岗
        - expert_label: 专家标识，0=非专业技术人员，1=专业技术人员/专家
        - 身高和体重字段可能为空字符串
        - 年龄由出生日期计算并存储为整数字符串
        - 所有在职员工的empl_status都是"在职"
        """,
        """
        数据质量说明：
        - 部分记录的身高/体重字段为空
        - 所有在职员工的员工状态都是"在职"
        - 出生日期格式为YYYY-MM-DD
        - 岗位、组织和公司名称使用中文字符
        - 籍贯和民族字段在某些记录中可能为空
        - 数据来源于企业人力资源管理系统
        """
    ]
    
    # Marketing knowledge table documentation
    marketing_knowledge_documentation = [
        """
        营销知识数据集数据字典：
        - id: 知识条目唯一标识符（如：Brand_case20, Brand_case21）
        - primary_category: 主要类别（如：品牌知识）
        - secondary_category: 次要类别（如：企业介绍、品牌历史、品牌介绍）
        - main_name: 品牌名称（如：LEASY, NATROL, SONAVOX, TOMFORD汤姆福特, AEKYUNG爱敬化工）
        - question: 问题内容（中文描述的业务问题）
        - answer: 正确答案内容
        - A, B, C, D: 四个选择题选项
        - answer_choose: 正确答案选项（A, B, C, D中的一个）
        """,
        """
        营销知识业务规则：
        - 每个知识条目都有唯一的id标识
        - primary_category主要为"品牌知识"
        - secondary_category包括：企业介绍、品牌历史、品牌介绍、企业名称等
        - question字段包含具体的业务问题
        - answer字段包含标准答案
        - A、B、C、D四个字段为选择题的四个选项
        - answer_choose指向正确的选项（A/B/C/D）
        - 品牌名称包含中文和英文混合格式
        - 涉及成立时间、地址、创始人等企业基本信息
        """,
        """
        数据关联说明：
        - marketing_knowledge表和bi_dataset表没有直接的外键关联
        - 但可以通过业务逻辑进行关联查询
        - 例如：可以查询某公司员工同时了解该公司品牌知识的情况
        - 支持跨表的统计分析和知识管理应用
        - 数据来源于企业品牌知识管理系统
        """
    ]
    
    for i, doc in enumerate(bi_dataset_documentation):
        try:
            doc_id = vn.train(documentation=doc.strip())
            print(f"✅ Added bi_dataset documentation {i+1}: {doc_id}")
        except Exception as e:
            print(f"❌ Error adding bi_dataset documentation {i+1}: {e}")
    
    for i, doc in enumerate(marketing_knowledge_documentation):
        try:
            doc_id = vn.train(documentation=doc.strip())
            print(f"✅ Added marketing_knowledge documentation {i+1}: {doc_id}")
        except Exception as e:
            print(f"❌ Error adding marketing_knowledge documentation {i+1}: {e}")
    
    print("\n3. TRAINING WITH QUESTION-SQL PAIRS")
    print("-" * 30)
    
    # 基于实际bi_dataset结构的中文问答示例
    training_examples = [
        {
            "question": "目前有多少在职员工？",
            "sql": "SELECT COUNT(*) as active_employees FROM bi_dataset WHERE empl_status = '在职'"
        },
        {
            "question": "按员工数量显示前5个公司",
            "sql": """SELECT company_name, COUNT(*) as employee_count 
                     FROM bi_dataset 
                     GROUP BY company_name, company_code 
                     ORDER BY employee_count DESC 
                     LIMIT 5"""
        },
        {
            "question": "显示所有关键员工信息",
            "sql": """SELECT person_code, position_name, organization_name, company_name, sex, age
                     FROM bi_dataset 
                     WHERE key_employee = '1'
                     ORDER BY age DESC"""
        },
        {
            "question": "男女性别分布情况如何？",
            "sql": """SELECT sex as '性别', COUNT(*) as '人数'
                     FROM bi_dataset
                     WHERE sex != '' AND sex IS NOT NULL
                     GROUP BY sex
                     ORDER BY COUNT(*) DESC"""
        },
        {
            "question": "查找30岁以上在核心岗位的员工",
            "sql": """SELECT person_code, position_name, age, organization_name, company_name
                     FROM bi_dataset 
                     WHERE CAST(age AS UNSIGNED) > 30 
                     AND core_position = '1'
                     ORDER BY CAST(age AS UNSIGNED) DESC"""
        },
        {
            "question": "婚姻状况的分布情况",
            "sql": """SELECT marital_status_name as '婚姻状况', COUNT(*) as '人数'
                     FROM bi_dataset
                     WHERE marital_status_name != '' AND marital_status_name IS NOT NULL
                     GROUP BY marital_status_name
                     ORDER BY COUNT(*) DESC"""
        },
        {
            "question": "查找来自重庆市的员工",
            "sql": """SELECT person_code, position_name, native_place, organization_name, company_name, age
                     FROM bi_dataset
                     WHERE native_place = '重庆市'
                     ORDER BY CAST(age AS UNSIGNED) DESC"""
        },
        {
            "question": "统计各个部门的员工数量",
            "sql": """SELECT organization_name as '部门名称', COUNT(*) as '员工数量'
                     FROM bi_dataset
                     GROUP BY organization_name, organization_code
                     ORDER BY COUNT(*) DESC"""
        }
    ]
    
    # Multi-table query examples between bi_dataset and marketing_knowledge
    multi_table_examples = [
        {
            "question": "统计营销知识数据中有多少个不同的品牌，同时统计员工数据中有多少个不同的公司",
            "sql": """SELECT 
                        (SELECT COUNT(DISTINCT main_name) FROM marketing_knowledge WHERE main_name IS NOT NULL AND main_name != '') as '品牌数量',
                        (SELECT COUNT(DISTINCT company_name) FROM bi_dataset WHERE company_name IS NOT NULL AND company_name != '') as '公司数量'"""
        },
        {
            "question": "查询营销知识中的所有品牌名称和员工数据中的所有公司名称",
            "sql": """SELECT DISTINCT main_name as '名称', '品牌' as '类型' FROM marketing_knowledge WHERE main_name IS NOT NULL AND main_name != ''
                     UNION ALL
                     SELECT DISTINCT company_name as '名称', '公司' as '类型' FROM bi_dataset WHERE company_name IS NOT NULL AND company_name != ''
                     ORDER BY 类型, 名称"""
        },
        {
            "question": "统计营销知识按主要类别分组的数量和员工按性别分组的数量",
            "sql": """SELECT primary_category as '类别', COUNT(*) as '数量', '营销知识' as '数据源' FROM marketing_knowledge GROUP BY primary_category
                     UNION ALL
                     SELECT sex as '类别', COUNT(*) as '数量', '员工数据' as '数据源' FROM bi_dataset WHERE sex != '' AND sex IS NOT NULL GROUP BY sex
                     ORDER BY 数据源, 数量 DESC"""
        },
        {
            "question": "查询营销知识中关于企业介绍的品牌数量和员工中关键员工的数量",
            "sql": """SELECT 
                        (SELECT COUNT(DISTINCT main_name) FROM marketing_knowledge WHERE secondary_category = '企业介绍') as '企业介绍品牌数',
                        (SELECT COUNT(*) FROM bi_dataset WHERE key_employee = '1') as '关键员工数量'"""
        },
        {
            "question": "分别统计营销知识和员工数据的总记录数",
            "sql": """SELECT 
                        (SELECT COUNT(*) FROM marketing_knowledge) as '营销知识总数',
                        (SELECT COUNT(*) FROM bi_dataset) as '员工总数'"""
        },
        {
            "question": "查询营销知识中品牌历史类别的问题和员工数据中30岁以上员工的信息（分别查询）",
            "sql": """SELECT 'marketing' as source, main_name as name, question as info FROM marketing_knowledge WHERE secondary_category = '品牌历史' LIMIT 5
                     UNION ALL
                     SELECT 'employee' as source, company_name as name, CONCAT(position_name, '-年龄:', age) as info FROM bi_dataset WHERE CAST(age AS UNSIGNED) > 30 LIMIT 5"""
        }
    ]
    
    for i, example in enumerate(training_examples):
        try:
            q_id = vn.train(question=example["question"], sql=example["sql"])
            print(f"✅ Added single-table Q&A {i+1}: {q_id}")
            print(f"   Question: {example['question'][:60]}...")
        except Exception as e:
            print(f"❌ Error adding single-table Q&A {i+1}: {e}")
    
    # Add multi-table examples
    for i, example in enumerate(multi_table_examples):
        try:
            q_id = vn.train(question=example["question"], sql=example["sql"])
            print(f"✅ Added multi-table Q&A {i+1}: {q_id}")
            print(f"   Question: {example['question'][:60]}...")
        except Exception as e:
            print(f"❌ Error adding multi-table Q&A {i+1}: {e}")
    
    print("\n4. TRAINING DATA MANAGEMENT")
    print("-" * 30)
    
    # Show current training data
    training_data = display_training_data_summary(vn)
    
    # Test training data removal (if there's data to remove)
    if len(training_data) > 0:
        print(f"\nTesting training data removal...")
        first_id = training_data.iloc[0]['id'] if 'id' in training_data.columns else None
        if first_id:
            try:
                removed = vn.remove_training_data(first_id)
                print(f"✅ Removal result: {removed}")
            except Exception as e:
                print(f"❌ Error removing training data: {e}")
    else:
        print("No training data available for removal testing")
    
    print("\n5. ADVANCED TRAINING SCENARIOS")
    print("-" * 30)
    
    # 针对bi_dataset的复杂中文SQL查询示例
    advanced_examples = [
        {
            "question": "计算各部门平均年龄，只显示人数超过3人的部门",
            "sql": """SELECT organization_name as '部门名称', 
                            COUNT(*) as '员工数量',
                            ROUND(AVG(CAST(age AS UNSIGNED)), 1) as '平均年龄'
                     FROM bi_dataset 
                     WHERE age != '' AND age REGEXP '^[0-9]+$' AND empl_status = '在职'
                     GROUP BY organization_name, organization_code
                     HAVING COUNT(*) > 3
                     ORDER BY AVG(CAST(age AS UNSIGNED)) DESC"""
        },
        {
            "question": "查找每个公司最年轻和最年长的员工年龄及年龄跨度",
            "sql": """SELECT company_name as '公司名称',
                            COUNT(*) as '员工总数',
                            MIN(CAST(age AS UNSIGNED)) as '最小年龄',
                            MAX(CAST(age AS UNSIGNED)) as '最大年龄',
                            MAX(CAST(age AS UNSIGNED)) - MIN(CAST(age AS UNSIGNED)) as '年龄跨度'
                     FROM bi_dataset
                     WHERE age != '' AND age REGEXP '^[0-9]+$' AND empl_status = '在职'
                     GROUP BY company_name, company_code
                     HAVING COUNT(*) > 2
                     ORDER BY MAX(CAST(age AS UNSIGNED)) - MIN(CAST(age AS UNSIGNED)) DESC"""
        },
        {
            "question": "统计各民族员工数量分布情况",
            "sql": """SELECT nation as '民族', COUNT(*) as '人数',
                            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM bi_dataset WHERE nation != '' AND nation IS NOT NULL), 2) as '百分比'
                     FROM bi_dataset
                     WHERE nation != '' AND nation IS NOT NULL AND empl_status = '在职'
                     GROUP BY nation
                     ORDER BY COUNT(*) DESC"""
        },
        {
            "question": "查找具有完整身体数据（身高和体重）的员工信息",
            "sql": """SELECT person_code as '员工编号', position_name as '岗位名称', 
                            organization_name as '部门名称', sex as '性别', 
                            height as '身高(cm)', weight as '体重(kg)', age as '年龄'
                     FROM bi_dataset
                     WHERE height != '' AND height IS NOT NULL AND height REGEXP '^[0-9]+$'
                     AND weight != '' AND weight IS NOT NULL AND weight REGEXP '^[0-9]+$'
                     AND empl_status = '在职'
                     ORDER BY CAST(age AS UNSIGNED) DESC"""
        },
        {
            "question": "分析干部和专家在各公司的分布情况",
            "sql": """SELECT company_name as '公司名称',
                            COUNT(*) as '员工总数',
                            SUM(CASE WHEN cadre_label = '1' THEN 1 ELSE 0 END) as '干部数量',
                            SUM(CASE WHEN expert_label = '1' THEN 1 ELSE 0 END) as '专家数量',
                            ROUND(SUM(CASE WHEN cadre_label = '1' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as '干部比例%',
                            ROUND(SUM(CASE WHEN expert_label = '1' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as '专家比例%'
                     FROM bi_dataset
                     WHERE empl_status = '在职'
                     GROUP BY company_name, company_code
                     HAVING COUNT(*) > 5
                     ORDER BY COUNT(*) DESC"""
        }
    ]
    
    for i, example in enumerate(advanced_examples):
        try:
            adv_id = vn.train(question=example["question"], sql=example["sql"])
            print(f"✅ Added advanced Q&A {i+1}: {adv_id}")
        except Exception as e:
            print(f"❌ Error adding advanced Q&A {i+1}: {e}")
    
    print("\n6. TESTING SQL GENERATION")
    print("-" * 30)
    
    # 使用训练模型测试SQL生成（包括单表和多表查询）
    test_questions = [
        # Single table queries
        "有多少在职员工？",
        "按部门显示关键员工", 
        "男女性别分布情况",
        "查找来自四川省的员工",
        "显示已婚员工信息",
        "核心岗位有多少人？",
        "干部的平均年龄是多少？",
        
        # Marketing knowledge single table queries
        "营销知识库中有多少个品牌？",
        "查询所有品牌历史类的知识",
        "统计各个品牌的知识条目数量",
        
        # Multi-table queries
        "统计营销知识和员工数据的总数量",
        "查询营销知识中的品牌名称和员工数据中的公司名称",
        "比较营销知识按类别分布和员工按性别分布",
        "营销知识中有多少企业介绍，员工中有多少关键员工？"
    ]
    
    for question in test_questions:
        print(f"\n🤖 Testing question: '{question}'")
        try:
            sql = vn.generate_sql(question)
            print(f"Generated SQL: {sql}")
        except Exception as e:
            print(f"❌ Error generating SQL: {e}")
    
    print("\n7. FINAL TRAINING DATA SUMMARY")
    print("-" * 30)
    final_training_data = display_training_data_summary(vn)
    
    print(f"\n📊 TRAINING INSIGHTS:")
    print(f"   - Schema training provides structure understanding")
    print(f"   - Documentation adds business context and rules")
    print(f"   - Question-SQL pairs enable pattern learning")
    print(f"   - Training data is stored in Qdrant vector database")
    print(f"   - Generated SQL uses trained context for accuracy")
    
    return vn


def demonstrate_training_best_practices():
    """Demonstrate best practices for training Vanna"""
    
    print("\n\n🎯 VANNA TRAINING BEST PRACTICES")
    print("=" * 50)
    
    practices = [
        {
            "practice": "Start with DDL",
            "description": "Always begin training with CREATE TABLE statements to establish schema knowledge",
            "example": "Train with complete table definitions including constraints and relationships"
        },
        {
            "practice": "Add Business Context", 
            "description": "Include documentation about business rules, data meanings, and constraints",
            "example": "Document what 'key_employee' means in business terms"
        },
        {
            "practice": "Provide Diverse Examples",
            "description": "Train with various SQL patterns: aggregations, joins, filtering, sorting",
            "example": "Include simple SELECTs, GROUP BY queries, and complex analytical queries"
        },
        {
            "practice": "Use Representative Data",
            "description": "Ensure training examples reflect real-world query patterns",
            "example": "Include common business questions and their SQL solutions"
        },
        {
            "practice": "Iterative Training",
            "description": "Train incrementally and test, then add more examples based on gaps",
            "example": "Start with basic queries, then add complex analytical examples"
        },
        {
            "practice": "Training Data Hygiene",
            "description": "Regularly review and clean training data, remove obsolete examples",
            "example": "Remove outdated schema definitions when tables are restructured"
        }
    ]
    
    for i, practice in enumerate(practices, 1):
        print(f"\n{i}. {practice['practice'].upper()}")
        print(f"   Description: {practice['description']}")
        print(f"   Example: {practice['example']}")


def test_ask_functionality(vn):
    """Test the full ask() functionality including SQL execution"""
    print("\n8. TESTING FULL ASK() FUNCTIONALITY")
    print("-" * 30)
    
    # 测试完整ask()功能（包括单表和多表查询）
    ask_questions = [
        # Single table queries - bi_dataset
        "数据库中有多少员工？",
        "显示性别分布情况",
        "我们数据库中有哪些公司？",
        "查找所有关键员工",
        "显示来自重庆市的员工",
        
        # Single table queries - marketing_knowledge
        "营销知识库中有哪些品牌？",
        "查询企业介绍类的知识条目",
        
        # Multi-table queries
        "统计营销知识和员工数据的记录总数",
        "比较营销知识的品牌数量和员工数据的公司数量"
    ]
    
    for question in ask_questions:
        print(f"\n💬 Asking: '{question}'")
        try:
            # Use ask() which generates SQL and can execute it
            result = vn.ask(question, print_results=True, auto_train=True)
            if result:
                sql, df, plot = result
                print(f"   SQL: {sql}")
                if df is not None and len(df) > 0:
                    print(f"   Results: {len(df)} rows returned")
                else:
                    print("   No results returned")
        except Exception as e:
            print(f"   ❌ Error with ask(): {e}")


if __name__ == "__main__":
    print("🚀 Starting Vanna Training Investigation with Real Connections...")
    
    # Set up Vanna connection
    vn = setup_vanna_connection()
    
    if vn is None:
        print("❌ Could not establish Vanna connection. Please check:")
        print("   - Qdrant is running on localhost:6333")
        print("   - OpenAI API credentials are valid")
        print("   - MySQL database is accessible")
        exit(1)
    
    try:
        # Demonstrate training with employee data
        print("\n" + "="*60)
        trained_vanna = demonstrate_employee_data_training(vn)
        
        # Test full ask functionality
        test_ask_functionality(trained_vanna)
        
        # Show best practices
        demonstrate_training_best_practices()
        
        print("\n\n✨ INVESTIGATION COMPLETE!")
        print("This script demonstrates real Vanna training capabilities with:")
        print("✅ OpenAI LLM integration")
        print("✅ Qdrant vector database storage") 
        print("✅ MySQL database connection")
        print("✅ DDL, documentation, and Q&A training")
        print("✅ SQL generation and execution")
        
    except Exception as e:
        print(f"\n❌ Investigation failed: {e}")
        print("Check your connections and credentials")