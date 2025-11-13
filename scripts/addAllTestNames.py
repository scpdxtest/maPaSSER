from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import datetime

app = Flask(__name__)
CORS(app)

@app.route('/getnames', methods=['GET'])
def get_test_names():
    # Connect to MongoDB
    from pymongo import MongoClient
    mongo_client = MongoClient('mongodb://your-mongodb-server-ip:port/')  # Change URL to your MongoDB connection string
    db = mongo_client['myDB']  # Database name
    collection_name = 'unique_test_names'
    # Check if collection exists
    collection_exists = collection_name in db.list_collection_names()
    if not collection_exists:
        print(f"Collection {collection_name} does not exist.")
        return jsonify({'error': 'Collection does not exist.'}), 404
    else:
        print(f"Collection {collection_name} exists.")
    # Get the collection
    metrics_collection = db[collection_name]
    # Retrieve all documents from the collection
    documents = list(metrics_collection.find({}, {'_id': 0}))  # Exclude the '_id' field
    # Close the MongoDB connection
    mongo_client.close()
    # Return the documents as JSON
    return jsonify(documents), 200

@app.route('/tests', methods=['POST'])
def metrics():
    data = request.json
    if not data:
        return jsonify({'error': 'JSON data is missing.'}), 400

    userID = data.get('userID')
    if not userID:
        return jsonify({'error': 'userID parameter is missing.'}), 400
    testID = data.get('testID')
    if not testID:
        return jsonify({'error': 'testID parameter is missing.'}), 400
    
    print("userID ---> ", userID)
    print("testID ---> ", testID)

    rouge_scores = []
    laplace_perplexity = 0.0
    lidstone_perplexity = 0.0
    cosine_similarity = 0.0
    pearson_corr = 0.0
    f1 = 0.0
    meteor_score = 0.0
    blue_score = 0.0
    bert1_score = {}
    bert_rt_score = {}  
    # Assuming these variables are calculated somewhere in your code

    res = [meteor_score, blue_score, 
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        laplace_perplexity, lidstone_perplexity, 
        0.0,
        pearson_corr, f1, 
        0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0]

    # MongoDB connection setup
    mongo_client = MongoClient('mongodb://your-mongodb-server-ip:port/')  # Change URL to your MongoDB connection string
    db = mongo_client['myDB']  # Database name
    
    # Check if collection exists, create it if it doesn't
    collection_name = 'unique_test_names'
    collection_exists = collection_name in db.list_collection_names()
    
    if not collection_exists:
        print(f"Creating new collection: {collection_name}")
        # You can specify collection options here if needed
        db.create_collection(collection_name)
        # Optional: Create indexes for better query performance
        db[collection_name].create_index([("testid", 1)])
        db[collection_name].create_index([("userid", 1)])
        print(f"Collection {collection_name} created successfully")
    else:
        print(f"Collection {collection_name} already exists")
    
    # Now you can use the collection
    metrics_collection = db[collection_name]

    # Check if a document with the given testid already exists
    existing_doc = metrics_collection.find_one({"testid": testID})
    
    if not existing_doc:
        # Structure the document with testid, creator, and timestamp
        current_datetime = datetime.datetime.utcnow()
        
        # Create the document with the requested structure
        mongo_document = {
            "testid": testID,
            "creator": userID,
            "created_at": current_datetime,
            "results": res,
            "description": '',
            "reference": '',
            "candidate": ''
        }
        
        # Insert the document into MongoDB
        try:
            result = metrics_collection.insert_one(mongo_document)
            print(f"MongoDB document inserted with ID: {result.inserted_id}")
        except Exception as e:
            print(f"Error inserting document into MongoDB: {e}")
    else:
        print(f"Document with testid '{testID}' already exists in MongoDB. Skipping insertion.")

    # Close the MongoDB connection
    mongo_client.close()
    return jsonify({'message': 'Data processed successfully.'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8088)
    # app.run(debug=True, host='195.230.127.227', port=8302)
