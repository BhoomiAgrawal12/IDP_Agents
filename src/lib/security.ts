import { createHash } from "crypto";

// Define types for security output
interface SecurityReport {
  anonymizedFields: string[];
  totalAnonymized: number;
}

interface AnonymizedResult {
  text: string;
  securityReport: SecurityReport;
}

// PII patterns
const piiPatterns: Record<string, RegExp> = {
  names: /\b[A-Z][a-z]+ [A-Z][a-z]+\b/g,
  emails: /\b[\w\.-]+@[\w\.-]+\b/g,
  phone: /\b\d{3}-\d{3}-\d{4}\b/g,
};

export function detectAndAnonymizeText(text: string, policy: string = "names"): AnonymizedResult {
  const report: SecurityReport = { anonymizedFields: [], totalAnonymized: 0 };
  let anonymizedText = text;

  // Determine fields to anonymize
  const fieldsToAnonymize: string[] = [];
  if (policy.includes("all")) {
    fieldsToAnonymize.push(...Object.keys(piiPatterns));
  } else {
    if (policy.includes("names")) fieldsToAnonymize.push("names");
    if (policy.includes("emails")) fieldsToAnonymize.push("emails");
    if (policy.includes("phone")) fieldsToAnonymize.push("phone");
  }
  if (fieldsToAnonymize.length === 0) fieldsToAnonymize.push("names"); // Default policy

  // Anonymize each field
  fieldsToAnonymize.forEach((field) => {
    const matches = text.match(piiPatterns[field]) || [];
    if (matches.length > 0) {
      report.anonymizedFields.push(field);
      report.totalAnonymized += matches.length;
      anonymizedText = anonymizedText.replace(
        piiPatterns[field],
        (match) =>
          `[${field.toUpperCase()}_${createHash("sha256")
            .update(match)
            .digest("hex")
            .slice(0, 8)}]`
      );
    }
  });

  return { text: anonymizedText, securityReport: report };
}