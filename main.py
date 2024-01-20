import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from tools import SQLTools, repl_tool

load_dotenv()

# ollama_openhermes = Ollama(model="openhermes", verbose=True)
gpt4 = ChatOpenAI(model="gpt-4-1106-preview", verbose=True)

data_scientist = Agent(
    role="Data Scientist",
    goal="Write SQL code to extract insights from tabular data",
    backstory="""Act as a research assistant. I will provide some context. Your role is to excel at writing SQL code 
    to extract insights. When given a task, you should think through it step-by-step and not assume column names if 
    they are not explicitly given. Follow these guidelines to ensure accuracy and efficiency in your SQL 
    code:
    
    1. Determine column names: If column names are not explicitly provided, use the following SQL code template to 
    query the table and retrieve column names:
    SELECT * FROM {schema.table_name} LIMIT 0;
    This query will return the column names. 
    Take note of the column names that you will need to use in your SQL code.
    
    2. Validate filters: If you need to filter on specific values, use the following SQL code template to determine 
    the unique values in the column you are filtering on: 
    SELECT DISTINCT {column_name} FROM {schema.table_name};
    This query will provide you with a list of unique values in the column, ensuring that you 
    use correct filters.
    
    3. Determine time and geographic granularity: Before writing the final SQL prompt, 
    ensure that you are working with the correct time and geographic granularity. If needed, write an SQL query to 
    determine the time and geographic granularity.
    
    4. Write the SQL code: Once you have determined the column names, validated filters, and specified the time and 
    geographic granularity, you can confidently write the SQL code to extract the insights.""",
    verbose=True,
    allow_delegation=False,
    tools=[
        SQLTools().query_table,
        SQLTools().get_column_names,
        SQLTools().get_unique_values,
        repl_tool,
    ],
    llm=gpt4,
)

task1 = Task(
    description="""Use the table elephant.registration_ld_mhd to determine how many cars of different makes are 
    registered in the state of California every month. Think of interesting graphs that you can create with this 
    data. Use the Python REPL for this.""",
    verbose=True,
    agent=data_scientist,
)

crew = Crew(
    agents=[data_scientist],
    tasks=[task1],
    verbose=True,
    process=Process.sequential,
)

result = crew.kickoff()
