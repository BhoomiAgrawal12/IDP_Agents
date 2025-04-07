import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import bcrypt from "bcrypt";
import { limiter } from "@/lib/ratelimiter";
import { encrypt } from "@/lib/encryption";

export async function POST(req: Request) {
  // 🔁 1. Get IP for rate limiting
  const ip = req.headers.get("x-forwarded-for") || "anonymous";

  const { success } = await limiter.limit(ip);
  if (!success) {
    return NextResponse.json({ message: "Too many requests" }, { status: 429 });
  }

 
  const { name, email, password, token } = await req.json();

  const verifyRes = await fetch("https://www.google.com/recaptcha/api/siteverify", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `secret=${process.env.RECAPTCHA_SECRET_KEY}&response=${token}`,
  });


  const { success: captchaSuccess } = await verifyRes.json();
  if (!captchaSuccess) {
    return NextResponse.json({ message: "reCAPTCHA failed" }, { status: 403 });
  }

  const existing = await prisma.user.findUnique({ where: { email } });
  if (existing) {
    return NextResponse.json({ message: "User already exists" }, { status: 400 });
  }

  const encryptedPhone = encrypt(email); 

  console.log('encryptedPhone:', encryptedPhone);
  const hashedPassword = await bcrypt.hash(password, 12);

  
  await prisma.user.create({
    data: {
      name,
      email,
      password: hashedPassword,
    },
  });

  return NextResponse.json({ message: "User registered successfully" }, { status: 201 });
}
