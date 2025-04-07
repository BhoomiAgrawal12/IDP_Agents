import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

const redis = Redis.fromEnv();

export const limiter = new Ratelimit({
  redis,
  limiter: Ratelimit.slidingWindow(3, "10 s"), 
  analytics: true,
});
