import pandas as pd
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from pyca.connect.redshift import Redshift
from typing import List

python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you "
    "want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)


class QueryInput(BaseModel):
    query: str = Field(description="should be an SQL search query")


class ColumnInput(BaseModel):
    table_name: str = Field(description="should be a Redshift table name")


class UniqueValuesInput(BaseModel):
    table_name: str = Field(description="should be a Redshift table name")
    column_name: str = Field(
        description="should be a column name in the Redshift table"
    )


class SQLTools:
    def __init__(self):
        self.redshift = Redshift()

    @staticmethod
    @tool(args_schema=QueryInput)
    def query_table(query: str) -> pd.DataFrame:
        """Look up things in Redshift SQL database."""
        return SQLTools().redshift.read_query(query)

    @staticmethod
    @tool(args_schema=ColumnInput)
    def get_column_names(table_name: str) -> List[str]:
        """Get column names from a Redshift table."""
        query = """SELECT * FROM {table_name} LIMIT 0""".format(table_name=table_name)
        return SQLTools.query_table(query).columns.tolist()

    @staticmethod
    @tool(args_schema=UniqueValuesInput)
    def get_unique_values(table_name: str, column_name: str) -> List[str]:
        """Get unique values in a column of a Redshift table."""
        query = """SELECT DISTINCT {column_name} FROM {table_name}""".format(
            column_name=column_name, table_name=table_name
        )
        return SQLTools.query_table(query).values.flatten().tolist()
