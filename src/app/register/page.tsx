"use client";

import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import ReCAPTCHA from "react-google-recaptcha";
import { useRef } from "react";

const formSchema = z.object({
  name: z.string().min(2, "Name must be at least 2 characters"),
  email: z.string().email("Invalid email"),
  password: z.string().min(6, "Password must be at least 6 characters"),
});

type FormData = z.infer<typeof formSchema>;

export default function RegisterPage() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<FormData>({
    resolver: zodResolver(formSchema),
  });
  const recaptchaRef = useRef<ReCAPTCHA>(null);

  const onSubmit = async (data: FormData) => {
    const token = await recaptchaRef.current?.executeAsync();
    if (!token) return alert("Please verify reCAPTCHA");

    const res = await fetch("/api/register", {
      method: "POST",
      body: JSON.stringify({ ...data, token }),
    });
    if (!res.ok) {
      const { message } = await res.json();
      return alert(message);
    }
    console.log("Form submitted: ", res);
  };

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="max-w-md mx-auto mt-20 space-y-4">
      <input
        {...register("name")}
        placeholder="Name"
        className="border p-2 w-full"
      />
      {errors.name && (
        <p className="text-red-500 text-sm">{errors.name.message}</p>
      )}

      <input
        {...register("email")}
        type="email"
        placeholder="Email"
        className="border p-2 w-full"
      />
      {errors.email && (
        <p className="text-red-500 text-sm">{errors.email.message}</p>
      )}

      <input
        {...register("password")}
        type="password"
        placeholder="Password"
        className="border p-2 w-full"
      />
      {errors.password && (
        <p className="text-red-500 text-sm">{errors.password.message}</p>
      )}

      <button type="submit" className="bg-black text-white px-4 py-2">
        Register
      </button>
      <ReCAPTCHA
        ref={recaptchaRef}
        sitekey={process.env.NEXT_PUBLIC_RECAPTCHA_SITE_KEY!}
      />
    </form>
  );
}
