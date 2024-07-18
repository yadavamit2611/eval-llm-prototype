from datasets import load_dataset

ds = load_dataset("LangChainDatasets/question-answering-state-of-the-union", split="train")
ds.to_csv('politics-dataset.csv')
print("dataset added")