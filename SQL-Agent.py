from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection parameters
dbname = "dvdrrental"
user = "x"
password = "x"
host = "x"
port = "x"

# Set up the PostgreSQL connection URI
pg_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
db = SQLDatabase.from_uri(pg_uri)

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create memory for chat history
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
    human_prefix="User",
    ai_prefix="AI"
)

# Define the prompt template for SQL assistant
prompt = """
You are an advanced SQL Query Assistant for a mutual funds database. Your goal is to generate precise and efficient SQL queries.

Key Guidelines:
1. Analyze the user's query thoroughly
2. Select only relevant columns and tables
3. Use appropriate JOINs, WHERE clauses, and aggregations
4. Optimize query for performance
5. Ensure data accuracy and meaningful insights

Specific Considerations:
- Focus on mutual funds performance analysis
- Handle complex financial data relationships
- Provide clear, actionable query results

User Query: {input}
Previous Context: {agent_scratchpad}

Generate a SQL query and execute it that precisely answers the user's question while maintaining query efficiency and clarity.
"""

# Set up the prompt template for the agent
prompt_template = PromptTemplate(
    input_variables=["user_query", "agent_scratchpad"],
    template=prompt
)

# Create the SQL agent
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    prompt=prompt_template,
    memory=memory,  
    verbose=True,
    agent_type='openai-tools'
)

# Main function to interact with the assistant
def main():
    print("SQL Assistant Started. Type 'stop' to exit.")
    
    while True:
        question = input("\nEnter Your Query ('stop'): ").strip() 
        
        if question.lower() == 'stop':  
            print("Exiting SQL Assistant. Goodbye!")
            break
        
        try:
            # Invoke the agent with the input
            response = agent_executor.invoke({"input": question})
            
            # Get the model's response
            model_answer = response['output']
            
            # Save user query and model's answer in memory
            memory.chat_memory.add_user_message(question)  # User's message
            memory.chat_memory.add_ai_message(model_answer)  # Model's response
            
            # Print the response
            print("\nGenerated Response:\n", model_answer)
            
            # Print the current chat history
            print("\nCurrent Chat History:")
            for msg in memory.chat_memory.messages:
                if isinstance(msg, HumanMessage):
                    print(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"AI: {msg.content}")
        
        except Exception as e:
            print(f"An error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    main()
