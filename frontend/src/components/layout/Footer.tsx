export function Footer() {
  return (
    <footer
      style={{
        background: "var(--nav-bg)",
        color: "rgba(241, 245, 249, 0.5)",
        padding: "1.5rem",
        textAlign: "center",
        fontSize: "0.8rem",
        marginTop: "4rem",
      }}
    >
      <p>
        Knowledge Hub &mdash; Built with FastAPI + Next.js &mdash;{" "}
        {new Date().getFullYear()}
      </p>
    </footer>
  );
}
