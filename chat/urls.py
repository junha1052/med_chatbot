from django.urls import path
from .view import ChatBotAPIView

urlpatterns = [
    path('bot/', ChatBotAPIView.as_view(), name='chat-bot'),
]