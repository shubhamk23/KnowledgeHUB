import type { Metadata } from "next";
import { ThemeProvider } from "next-themes";
import { Toaster } from "sonner";
import "./globals.css";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { getSections } from "@/lib/api";
import type { Section } from "@/types";

export const metadata: Metadata = {
  title: {
    default: "Knowledge Hub",
    template: "%s | Knowledge Hub",
  },
  description: "A personal AI/ML knowledge hub — notes, research reviews, and blog articles.",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  let sections: Section[] = [];
  try {
    sections = await getSections();
  } catch {
    // Backend not running — show empty nav
  }

  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <Navbar sections={sections} />
          <main style={{ minHeight: "calc(100vh - 56px - 80px)" }}>{children}</main>
          <Footer />
          <Toaster position="bottom-right" richColors />
        </ThemeProvider>
      </body>
    </html>
  );
}
