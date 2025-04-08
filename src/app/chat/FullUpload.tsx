'use client';
import React, { useCallback } from 'react';
import { FileType } from './type';
import { Upload } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File, type: FileType) => void;
}

export function FileUpload({ onFileSelect }: FileUploadProps) {
  const determineFileType = (file: File): FileType => {
    const type = file.type.split('/')[0];
    switch (type) {
      case 'image':
        return 'image';
      case 'audio':
        return 'audio';
      case 'video':
        return 'video';
      case 'text':
        return file.name.endsWith('.json') || 
               file.name.endsWith('.js') || 
               file.name.endsWith('.ts') || 
               file.name.endsWith('.py') ? 'code' : 'document';
      default:
        return 'other';
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      const fileType = determineFileType(file);
      onFileSelect(file, fileType);
    }
  }, [onFileSelect]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const fileType = determineFileType(file);
      onFileSelect(file, fileType);
    }
  }, [onFileSelect]);

  return (
    <div
      className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-blue-500 transition-colors"
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
    >
      <input
        type="file"
        className="hidden"
        onChange={handleChange}
        id="file-upload"
      />
      <label htmlFor="file-upload" className="cursor-pointer">
        <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
        <p className="text-gray-600">
          Drag and drop a file here, or click to select
        </p>
        <p className="text-sm text-gray-500 mt-2">
          Supports images, documents, audio, video, and code files
        </p>
      </label>
    </div>
  );
}