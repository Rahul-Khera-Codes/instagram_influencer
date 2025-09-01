import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**.cdninstagram.com", // covers all Instagram CDN subdomains
      },
    ],
  },
};

export default nextConfig;
