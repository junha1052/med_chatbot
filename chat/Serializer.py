from rest_framework import serializers

class ChatRequestSerializer(serializers.Serializer):
    query = serializers.CharField()

class ChatResponseSerializer(serializers.Serializer):
    answer = serializers.CharField()