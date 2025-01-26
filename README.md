# **NLP Pipeline for Multi-Label Classification and Entity Extraction**

## **Overview**
This project implements an end-to-end Natural Language Processing (NLP) pipeline designed to perform:
1. **Multi-Label Text Classification**: Classifies snippets into categories like `Objection`, `Pricing Discussion`, `Security`, and `Competition`.
2. **Entity/Keyword Extraction**: Extracts domain-specific entities (e.g., competitors, product features) using a dictionary-based lookup and advanced regex-based methods.
3. **REST API**: A containerized API built with FastAPI to accept text snippets and return:
   - Predicted labels.
   - Extracted entities.
   - Summary.

This pipeline is fully Dockerized, making it easy to deploy and scale.

---

## **Key Features**
- Multi-label classification using Logistic Regression, Support Vector Machines (SVM), and ensemble methods.
- Dictionary-based and regex-based entity extraction.
- REST API with endpoints for inference.
- Containerized using Docker for easy deployment.

---

## **Setup and Usage**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/NLP-Pipeline.git
cd NLP-Pipeline
```

### **2. Install Dependencies Locally (Optional)**
If you want to run the project without Docker:
```bash
pip install -r requirements.txt
```

### **3. Using Docker**
#### **3.1 Build the Docker Image**
Make sure Docker is installed and running. Then, build the Docker image:
```bash
docker build -t nlp-pipeline-service .
```

#### **3.2 Run the Docker Container**
Run the container and expose it on port 8000:
```bash
docker run -p 8000:8000 nlp-pipeline-service
```

#### **3.3 Test the API**
Use curl or Postman to test the `/process` endpoint. Example:
```bash
curl -X POST "http://localhost:8000/process" \
-H "Content-Type: application/json" \
-d '{"snippet": "Data security is paramount, especially with increasing regulatory pressures."}'
```

**Expected Output:**
```json
{
  "predicted_labels": ["security", "features"],
  "extracted_entities": {
    "competitors": [],
    "features": ["data", "security"],
    "pricing_keywords": []
  },
  "summary": "The snippet mentions no competitors and focuses on data, security."
}

## **Colab Notebook**
You can explore and run the NLP Pipeline in this [Google Colab Notebook](https://colab.research.google.com/drive/1RN4l3awJCu8o9awNN4JzAzuPT21LrqCt?usp=sharing).
