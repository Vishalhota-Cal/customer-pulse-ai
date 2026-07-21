from brain.classifier import OpenAIClient, classify_feedback
from domain.feedback import FeedbackItem

client = OpenAIClient()
item = FeedbackItem(id='test1', text='The app crashes every time I try to upload a photo')
result = classify_feedback(item, client)
print(result)
