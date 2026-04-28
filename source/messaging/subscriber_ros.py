from is_wire.core import Channel, Subscription, Message
import time

channel = Channel("amqp://10.20.5.2:30000")
subscription = Subscription(channel, name="TestSubscriber")
subscription.subscribe(topic="ros.object_footprint")

print("Listening on 'ros.object_footprint'...")

while True:
    try:
        msg = channel.consume(timeout=1.0)
        if msg:
            print(f"Received: {msg}")
            print(f"Body: {msg.body}")
    except Exception as e:
        print(f"Waiting... ({e})")