from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

#stocks = ["AAPL", "GOOG", "TSLA"]
stocks = ["APPLE", "GOOGLE", "TESLA", "MICROSOFT", "AMAZON", "NETFLIX", "META"]
#rices = {"AAPL": 175.0, "GOOG": 140.0, "TSLA": 250.0}
prices = {
    "APPLE": 175.0,
    "GOOGLE": 140.0,
    "TESLA": 250.0,
    "MICROSOFT": 320.0,
    "AMAZON": 130.0,
    "NETFLIX": 400.0,
    "META": 300.0
}

print("Producer started. Sending stock data to Kafka...")

while True:
    stock = random.choice(stocks)
    change = random.uniform(-6.5, 8.5)
    prices[stock] = max(50, prices[stock] + change)

    data = {
        "stock": stock,
        "price": round(prices[stock], 2),
        "timestamp": time.time()
    }

    producer.send("stock-data", data)
    print(f"Sent: {data}")
    time.sleep(0.8)