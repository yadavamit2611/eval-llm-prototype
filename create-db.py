from pymongo import MongoClient

# Replace the URI with your MongoDB URI if you're using a cloud service
client = MongoClient("mongodb://localhost:27017/")
db = client["eval_db"]  # Replace 'food_db' with your database name
collection = db["dummy_data"]  # Replace 'questions_answers' with your collection name

# Data for the CSV file
data = [
    ["Sector", "Question", "Answer"],
    ["Computer Science", "What is an algorithm?", "An algorithm is a step-by-step procedure or formula for solving a problem or performing a task in computing."],
    ["Computer Science", "What is a programming language?", "A programming language is a formal language used to write instructions that a computer can execute."],
    ["Computer Science", "What is a data structure?", "A data structure is a way of organizing and storing data so it can be accessed and modified efficiently."],
    ["Computer Science", "What is object-oriented programming?", "Object-oriented programming (OOP) is a programming paradigm based on the concept of objects, which can contain data and methods."],
    ["Computer Science", "What is a compiler?", "A compiler is a program that translates source code written in a high-level programming language into machine code."],
    ["Computer Science", "What is a database?", "A database is an organized collection of structured information or data, typically stored electronically in a computer system."],
    ["Computer Science", "What is cloud computing?", "Cloud computing is the delivery of computing services over the internet, including storage, processing, and software."],
    ["Computer Science", "What is machine learning?", "Machine learning is a subset of artificial intelligence that involves training algorithms to learn patterns from data and make decisions."],
    ["Computer Science", "What is a computer network?", "A computer network is a group of interconnected computers that share resources and information."],
    ["Computer Science", "What is an operating system?", "An operating system (OS) is software that manages hardware and software resources on a computer, providing services for computer programs."],
    ["Computer Science", "What is a virtual machine?", "A virtual machine (VM) is an emulation of a computer system that runs on a host machine, allowing multiple OS instances on one hardware platform."],
    ["Computer Science", "What is cybersecurity?", "Cybersecurity is the practice of protecting systems, networks, and data from digital attacks, damage, or unauthorized access."],
]


# Convert list to dictionary format suitable for MongoDB
documents = []
for entry in data[1:]:
    document = {
        "Sector": entry[0],
        "Question": entry[1],
        "Answer": entry[2]
    }
    documents.append(document)

# Insert data into MongoDB collection
collection.insert_many(documents)

print("Data successfully inserted into MongoDB collection.")
