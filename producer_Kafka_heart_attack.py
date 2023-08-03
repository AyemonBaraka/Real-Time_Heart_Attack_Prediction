from kafka import KafkaProducer
import csv
from time import sleep
import json

# Kafka broker address
kafka_broker = "localhost:9092"
# Kafka topic to produce messages to
topic = "my_json_topic"
# Kafka broker address
bootstrap_servers = '10.18.17.153:9092'
# Kafka topics to send the data
csv_file_path = "/home/ayemon/KafkaProjects/kafkaspark07_90/heart_20.csv"

# Function to read the CSV file and send rows as messages to Kafka
# Function to read the CSV file and send rows as JSON messages to Kafka
def produce_json_messages(producer, csv_file):

   with open(csv_file, 'r') as file:
       csv_reader = csv.reader(file)
       header = next(csv_reader)  # Get the header
       for row in csv_reader:
           # Assuming each row is a list of numerical values
           message = dict(zip(header, row))
           producer.send(topic, value=message)
           print(f"Sending data to topic '{topic}': {message}")
           sleep(10)

if __name__ == "__main__":
    # Create a Kafka producer instance
    # Create a Kafka producer
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                             value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    # Call the function to produce messages from the CSV file
    produce_json_messages(producer, csv_file_path)

    # Close the producer after sending all messages
    producer.close()