export type FileType = "image" | "document" | "audio" | "video" | "code" | "other";

export interface ChatMessage {
  content: string;
  isUser: boolean;
  timestamp: Date;
}

export interface FileUpload {
  file: File;
  type: FileType;
  preview?: string;
}