from tensorflow.python.summary.summary_iterator import summary_iterator

# Specify the path to the event file
event_file = r'C:\Users\alyey\Git\fl-energy\logs\events.out.tfevents.1728772451.AlyEAhmad'

# Iterate through the event records
for event in summary_iterator(event_file):
    for value in event.summary.value:
        if value.HasField('simple_value'):
            print(f"Step: {event.step}, Value: {value.simple_value}, Tag: {value.tag}")
