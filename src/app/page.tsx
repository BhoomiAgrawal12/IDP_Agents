'use client';
import { useState, ChangeEvent, FormEvent } from 'react';
import axios, { AxiosResponse } from 'axios';

interface ProcessResponse {
  structured_output: string;
}

export default function Home() {
  const [query, setQuery] = useState<string>('');
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<string>('');

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!query && files.length === 0) {
      alert('Please enter a query or upload files.');
      return;
    }

    const formData = new FormData();
    formData.append('query', query);
    files.forEach((file) => {
      formData.append('files', file);
    });

    try {
      const res: AxiosResponse<ProcessResponse> = await axios.post(
        'http://localhost:8000/process',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      setResult(res.data.structured_output);
    } catch (error) {
      console.error('Error during processing:', error);
      setResult('An error occurred while processing your request.');
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Autonomous Document Intelligence</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter your query..."
          className="border p-2 w-full mb-2"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <input
          type="file"
          multiple
          className="mb-2"
          onChange={handleFileChange}
          accept=".pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.jpg,.jpeg,.png,.txt" // optional
        />
        <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded">
          Submit
        </button>
      </form>
      <div className="mt-4 whitespace-pre-wrap">{result}</div>
    </div>
  );
}
