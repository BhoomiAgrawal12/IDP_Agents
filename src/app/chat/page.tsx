"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";

type FileType = "image" | "document" | "audio" | "video" | "code" | "other";

type AttachedFile = {
  file: File;
  type: FileType;
  preview?: string;
  id: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachments?: AttachedFile[];
  timestamp: Date;
};

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<AttachedFile[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<AttachedFile | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [input]);

  const getFileType = (file: File): FileType => {
    const imageTypes = ["image/jpeg", "image/png", "image/gif", "image/webp"];
    const documentTypes = ["application/pdf", "text/plain"];
    const audioTypes = ["audio/mpeg", "audio/wav"];
    const videoTypes = ["video/mp4", "video/webm"];
    const codeTypes = ["text/javascript", "application/json"];

    if (imageTypes.includes(file.type)) return "image";
    if (documentTypes.includes(file.type)) return "document";
    if (audioTypes.includes(file.type)) return "audio";
    if (videoTypes.includes(file.type)) return "video";
    if (codeTypes.includes(file.type)) return "code";
    return "other";
  };

  const createFilePreview = async (file: File): Promise<AttachedFile> => {
    const fileType = getFileType(file);
    const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    if (fileType === "image" || fileType === "video" || fileType === "audio") {
      return { file, type: fileType, preview: URL.createObjectURL(file), id };
    }
    return { file, type: fileType, id };
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      const filePromises = newFiles.map(createFilePreview);
      const processedFiles = await Promise.all(filePromises);

      setFiles((prev) => {
        const existingFiles = new Set(
          prev.map((f) => `${f.file.name}-${f.file.size}`)
        );
        const uniqueNewFiles = processedFiles.filter(
          (f) => !existingFiles.has(`${f.file.name}-${f.file.size}`)
        );
        return [...prev, ...uniqueNewFiles];
      });

      e.target.value = "";
    }
  };

  const sendToBackend = async (
    query: string,
    attachedFiles: AttachedFile[]
  ) => {
    try {
      const formData = new FormData();
      formData.append("query", query);

      attachedFiles.forEach((attachedFile) => {
        formData.append("files", attachedFile.file);
      });

      const response = await axios.post(
        "http://localhost:8000/process",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      return response.data.structured_output;
    } catch (error) {
      console.error("Error sending data to backend:", error);
      throw error;
    }
  };

  const handleSendMessage = async () => {
    if (input.trim() === "" && files.length === 0) return;

    const newUserMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      attachments: files.length > 0 ? [...files] : undefined,
      timestamp: new Date(),
    };

    setMessages([...messages, newUserMessage]);
    setInput("");
    setIsLoading(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      // Only keep the files array for sending to backend, clear UI state
      const filesToSend = [...files];
      setFiles([]);

      // Send data to backend
      const backendResponse = await sendToBackend(input, filesToSend);

      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: backendResponse || "I processed your request.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, aiResponse]);
    } catch (error) {
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "Sorry, there was an error processing your request. Please try again.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  const removeFile = (id: string) => {
    setFiles((prev) => {
      const fileToRemove = prev.find((f) => f.id === id);
      if (fileToRemove?.preview) URL.revokeObjectURL(fileToRemove.preview);
      return prev.filter((f) => f.id !== id);
    });
  };

  const openFileSelector = () => {
    fileInputRef.current?.click();
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const getFileIcon = (type: FileType) => {
    switch (type) {
      case "image":
        return (
          <svg
            className="h-5 w-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
        );
      case "document":
        return (
          <svg
            className="h-5 w-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
          </svg>
        );
      case "audio":
        return (
          <svg
            className="h-5 w-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <path d="M9 18V5l12-2v13" />
            <circle cx="6" cy="18" r="3" />
            <circle cx="18" cy="16" r="3" />
          </svg>
        );
      case "video":
        return (
          <svg
            className="h-5 w-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <rect x="2" y="2" width="20" height="20" rx="2.18" />
            <polygon points="7 2 7 22 17 22 17 2" />
          </svg>
        );
      case "code":
        return (
          <svg
            className="h-5 w-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <polyline points="16 18 22 12 16 6" />
            <polyline points="8 6 2 12 8 18" />
          </svg>
        );
      default:
        return (
          <svg
            className="h-5 w-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
          >
            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
            <polyline points="13 2 13 9 20 9" />
          </svg>
        );
    }
  };

  const renderFilePreview = () => {
    if (!selectedFile) return null;

    return (
      <div
        className="fixed inset-0 bg-black/80 flex items-center justify-center z-50"
        onClick={() => setSelectedFile(null)}
      >
        <div
          className="bg-white dark:bg-gray-800 rounded-lg p-4 max-w-3xl max-h-[90vh] overflow-auto"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white truncate max-w-[80%]">
              {selectedFile.file.name}
            </h3>
            <button
              onClick={() => setSelectedFile(null)}
              className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            >
              <svg
                className="h-6 w-6"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          {selectedFile.type === "image" && selectedFile.preview && (
            <img
              src={selectedFile.preview}
              alt={selectedFile.file.name}
              className="max-w-full h-auto rounded"
            />
          )}

          {selectedFile.type === "video" && selectedFile.preview && (
            <video controls className="max-w-full h-auto rounded">
              <source
                src={selectedFile.preview}
                type={selectedFile.file.type}
              />
            </video>
          )}

          {selectedFile.type === "audio" && selectedFile.preview && (
            <audio controls className="w-full">
              <source
                src={selectedFile.preview}
                type={selectedFile.file.type}
              />
            </audio>
          )}

          {(selectedFile.type === "document" ||
            selectedFile.type === "code") && (
            <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded text-sm overflow-auto max-h-[60vh]">
              <p className="text-gray-500 dark:text-gray-400">
                Preview not available for this file type
              </p>
              <p className="mt-2">
                Size: {(selectedFile.file.size / 1024).toFixed(2)} KB
              </p>
              <p>Type: {selectedFile.file.type || "Unknown"}</p>
            </div>
          )}

          {selectedFile.type === "other" && (
            <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded text-sm">
              <p className="text-gray-500 dark:text-gray-400">
                No preview available
              </p>
              <p className="mt-2">
                Size: {(selectedFile.file.size / 1024).toFixed(2)} KB
              </p>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                <svg
                  className="h-6 w-6 text-white"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                >
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <div>
                <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
                  AI Assistant
                </h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {isLoading ? "Typing..." : "Online"}
                </p>
              </div>
            </div>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center text-gray-500 dark:text-gray-400">
              <div className="text-center">
                <svg
                  className="h-16 w-16 mx-auto mb-4"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                >
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
                <p>Start by sending a message or attaching files</p>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[70%] p-3 rounded-lg ${
                    message.role === "user"
                      ? "bg-gradient-to-r from-purple-500 to-pink-500 text-white"
                      : "bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700"
                  }`}
                >
                  <p className="text-sm">{message.content}</p>
                  {message.attachments && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      {message.attachments.map((attachment) => (
                        <div
                          key={attachment.id}
                          className="relative cursor-pointer group"
                          onClick={() => setSelectedFile(attachment)}
                        >
                          {attachment.type === "image" && attachment.preview ? (
                            <div className="w-20 h-20 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-600">
                              <img
                                src={attachment.preview}
                                alt={attachment.file.name}
                                className="w-full h-full object-cover"
                              />
                              <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                <svg
                                  className="h-6 w-6 text-white"
                                  viewBox="0 0 24 24"
                                  fill="none"
                                  stroke="currentColor"
                                >
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth="2"
                                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                  />
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth="2"
                                    d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                                  />
                                </svg>
                              </div>
                            </div>
                          ) : (
                            <div className="flex items-center gap-1 bg-gray-100 dark:bg-gray-700 p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors">
                              {getFileIcon(attachment.type)}
                              <span className="text-xs truncate max-w-[120px]">
                                {attachment.file.name}
                              </span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                  <p
                    className={`text-xs mt-1 ${
                      message.role === "user"
                        ? "text-white/80 text-right"
                        : "text-gray-500"
                    }`}
                  >
                    {formatTime(message.timestamp)}
                  </p>
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white dark:bg-gray-800 rounded-lg p-3 flex items-center gap-1">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
                <div className="w-2 h-2 bg-pink-500 rounded-full animate-pulse delay-150"></div>
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse delay-300"></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* File Preview */}
        {files.length > 0 && (
          <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-3">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-300">
                Attachments ({files.length})
              </span>
              <button
                onClick={() => {
                  files.forEach(
                    (file) => file.preview && URL.revokeObjectURL(file.preview)
                  );
                  setFiles([]);
                }}
                className="text-sm text-red-500 hover:text-red-600"
              >
                Clear All
              </button>
            </div>
            <div className="flex overflow-x-auto gap-3 pb-2">
              {files.map((file) => (
                <div key={file.id} className="relative flex-shrink-0 group">
                  {file.type === "image" && file.preview ? (
                    <div
                      className="w-20 h-20 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-600 cursor-pointer"
                      onClick={() => setSelectedFile(file)}
                    >
                      <img
                        src={file.preview}
                        alt={file.file.name}
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                        <svg
                          className="h-6 w-6 text-white"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                          />
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                          />
                        </svg>
                      </div>
                    </div>
                  ) : (
                    <div
                      className="w-20 h-20 bg-gray-100 dark:bg-gray-700 rounded-lg flex flex-col items-center justify-center p-2 cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                      onClick={() => setSelectedFile(file)}
                    >
                      {getFileIcon(file.type)}
                      <span className="text-xs text-center truncate w-full mt-1">
                        {file.file.name}
                      </span>
                    </div>
                  )}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(file.id);
                    }}
                    className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-5 h-5 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-end gap-3">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                className="w-full bg-gray-100 dark:bg-gray-700 rounded-lg p-3 pr-10 border-0 focus:ring-2 focus:ring-purple-500 resize-none text-sm"
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                rows={1}
              />
              <button
                onClick={openFileSelector}
                className="absolute right-3 bottom-3 text-gray-500 hover:text-purple-500"
              >
                <svg
                  className="h-5 w-5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
                  />
                </svg>
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                multiple
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={input.trim() === "" && files.length === 0}
              className="bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg p-3 disabled:opacity-50 hover:from-purple-600 hover:to-pink-600 transition-colors"
            >
              <svg
                className="h-5 w-5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
              >
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            Shift + Enter for new line
          </p>
        </div>

        {/* File Preview Modal */}
        {renderFilePreview()}
      </div>
    </div>
  );
}
