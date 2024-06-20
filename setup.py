from setuptools import find_packages,setup

setup(
    name='EquiNews-LLM-AI-Powered-News-Research-Tool-for-Equity-Analysts',
    version='0.0.1',
    author='Anish Yadav',
    author_email='reach.anish.yadav@gmail.com',
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2","unstructured","faiss-cpu"],
    packages=find_packages()
)

