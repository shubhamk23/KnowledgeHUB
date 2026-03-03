"use client";

import { useEffect, useRef } from "react";

interface Props {
  html: string;
}

export function NoteRenderer({ html }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;

    // Add copy button to every pre block
    const preBlocks = ref.current.querySelectorAll("pre");
    preBlocks.forEach((pre) => {
      if (pre.parentElement?.classList.contains("code-block-wrapper")) return;

      const wrapper = document.createElement("div");
      wrapper.className = "code-block-wrapper";
      pre.parentNode?.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);

      const btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.textContent = "Copy";
      btn.addEventListener("click", async () => {
        const code = pre.querySelector("code")?.innerText ?? pre.innerText;
        await navigator.clipboard.writeText(code);
        btn.textContent = "Copied!";
        setTimeout(() => (btn.textContent = "Copy"), 2000);
      });
      wrapper.appendChild(btn);
    });
  }, [html]);

  return (
    <div
      ref={ref}
      className="prose"
      style={{ width: "100%" }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
