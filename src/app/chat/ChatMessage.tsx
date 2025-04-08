'use client';
import React from 'react';
import { ChatMessage as ChatMessageType } from './type';
import { User, Bot } from 'lucide-react';

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  return (
    <div className={`flex items-start gap-4 ${message.isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
        message.isUser ? 'bg-blue-500' : 'bg-gray-500'
      }`}>
        {message.isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : (
          <Bot className="w-5 h-5 text-white" />
        )}
      </div>
      <div className={`flex-1 rounded-lg p-4 ${
        message.isUser ? 'bg-blue-500 text-white' : 'bg-gray-100'
      }`}>
        <p className="text-sm">{message.content}</p>
        <span className="text-xs opacity-75 block mt-2">
          {new Date(message.timestamp).toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
}