export interface SecurityReport {
    anonymizedFields: string[];
    totalAnonymized: number;
  }
  
  export interface ProcessResult {
    dataset: { files: File[]; text: string };
    qualityIssues?: string[];
    securityReport?: SecurityReport;
  }
  
  export interface ProcessRequest {
    files: File[];
    query: string;
  }