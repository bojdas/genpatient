# use ollama and langchain to create clinical notes
# before running this, run 'ollama serve' on local machine


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OllamaEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
from datetime import datetime
import sys, time

# selected_model="llama2:13b"
selected_model="mistral"
# selected_model="gemma:7b"
# selected_model="llama3"
clinical_notes = "patient_notes.txt"
colleague_info="colleague_notes.txt"
result = "result.txt"



def main():
    # the number of case notes to generate    
    num_cases = 1
    print(f" Creating Case Notes for {num_cases} patients")
    llm = Ollama(
    model=selected_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=1
    )
    # tbd: A general instruction
    # llm("""I shall ask you to generate clinnical notes. Make each note about an unique patient with a different name and id.
    # Wait for further instructions before generating responses""")


    prompt = """You are a clinician specializing in epilepsy patients. You will generate a Clinical Note for an epilepsy patient. 
    Each note should include the patient's name, id, date of visit, all current medications including dosage, chief complaint, 
    history of adverse reactions, family medical history, general clinical observations, assessment and a followup plan
        Use the following format for the note:
        Patient Name:
        Patient Id:
        Date of Visit:
        Current Medications:
        Chief Complaint:
        Advserse Reactions:
        Family Medical History:
        General Observations:
        Assessment:
        Follow-up Plan:

        Make sure that patient names and ids are unique for each note you generate. Uniqueness of patient identity is very important for your job.
        Make sure patients are from diverse ethnicities and diverse backgrounds""" 
    # prompt_without_drug = """You will generate a Clinical Note for an epilepsy patient who does not use drug X. Each note should include
        # the patient's name, id, date of visit, current medications, chief complaint, general observations, assessment and a followup plan"""


    notes_str = create_notes(llm, num_cases, prompt, clinical_notes)
    print()
    print(f"Created notes at the file {clinical_notes}")

    combo_file_prompt(llm, notes_str, colleague_info, result)
 


def create_notes(llm, num_cases, prompt, filename):
    # llm = Ollama(
    #     model=selected_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    # )
    notes_str = ""
    with open(filename, 'a') as f:
        sys.stdout = f
        print()
        stamp = datetime.now()
        print("===========================================Clinical Note====================================")
        for i in range(num_cases):
            # prompt = """You will generate a Clinical Note for an epilepsy patient based on the usage of drug X; the note shall
            # include an assessment of whether there was progress. Each note should include
            # the patient's name, id, date of visit, current medications, chief complaint, general observations, assessment and a followup plan"""        
            note = llm.invoke(prompt)
            notes_str += note
            # print(res)
            print()
            print("========================================================================================")
            time.sleep(5) # it seems like more time ==> better results
        f.flush()
        sys.stdout = sys.__stdout__

    return notes_str

def combo_file_prompt(llm, clin_str, colleague_info, result):
    coll_str = ""
    with open(colleague_info, 'r') as file:
        coll_str = file.read()

    combo_prompt = f"""Use the following information to provide an opinion about whether the patient should be prescribed the drug Cenbomate.
    First these are the clinical notes for applicable patient: {clin_str}. Also consider the opinion of experts in
    this field: {coll_str}"""

    ans = llm.invoke(combo_prompt)

    print("LLM ANS:", ans)
    

    pass


if __name__ == "__main__":
    main()
