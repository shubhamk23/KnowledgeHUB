import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  experimental: {
    serverActions: {
      allowedOrigins: ["localhost:3000"],
    },
  },
  // Allow fetching from the FastAPI backend during SSR
  async rewrites() {
    return [];
  },
};

export default nextConfig;
