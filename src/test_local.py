# Replace the old community import with this line:
from langchain_ollama import ChatOllama

def test_local_slm():
    print("Testing connection to Local Gemma 4...")
    llm = ChatOllama(model="gemma4:e4b", temperature=0)
    response = llm.invoke("What is the time complexity of binary search?")
    print(f"\nResponse from Gemma 4:\n{response.content}\n")

if __name__ == "__main__":
    test_local_slm()