"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AuthState {
  token: string | null;
  username: string | null;
  isAuthenticated: boolean;
  setToken: (token: string, username: string) => void;
  clearToken: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      token: null,
      username: null,
      isAuthenticated: false,
      setToken: (token, username) => {
        localStorage.setItem("admin_token", token);
        set({ token, username, isAuthenticated: true });
      },
      clearToken: () => {
        localStorage.removeItem("admin_token");
        set({ token: null, username: null, isAuthenticated: false });
      },
    }),
    {
      name: "admin-auth",
    }
  )
);
