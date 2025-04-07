import { getPresignedUrl } from "@/lib/s3";
import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const { filename, fileType } = await req.json();
  const url = await getPresignedUrl(filename, fileType);
  return NextResponse.json({ url });
}
