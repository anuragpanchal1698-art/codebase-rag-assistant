import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const STORAGE_KEY = "codebase-rag-chat-history";

function loadHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

const EXAMPLES = [
  "Where is RecursiveCharacterTextSplitter defined?",
  "How does ConversationBufferMemory work?",
  "What does the README say about installation?",
];

function MarkdownContent({ text }) {
  return (
    <div className="prose prose-invert prose-sm max-w-none
      prose-p:my-1.5 prose-headings:mt-3 prose-headings:mb-1.5
      prose-table:my-2 prose-th:px-3 prose-th:py-1.5 prose-td:px-3 prose-td:py-1.5
      prose-th:bg-white/[0.06] prose-td:border-white/[0.08] prose-th:border-white/[0.08]
      prose-code:text-indigo-300 prose-code:bg-black/30 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:before:content-none prose-code:after:content-none
      prose-strong:text-neutral-100
      prose-ul:my-1.5 prose-li:my-0.5
    ">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState(loadHistory);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedPdfs, setUploadedPdfs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  async function sendMessage(overrideText) {
    const text = (overrideText ?? input).trim();
    if (!text || loading) return;

    const userMsg = { role: "user", content: text };
    const nextMessages = [...messages, userMsg];
    setMessages([...nextMessages, { role: "assistant", content: "" }]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history: messages }),
      });

      if (!res.ok || !res.body) throw new Error(`Server error: ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        accumulated += decoder.decode(value, { stream: true });

        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: accumulated };
          return updated;
        });
      }

      if (!accumulated.trim()) {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "assistant",
            content: "(No response text was generated. Try rephrasing your question.)",
          };
          return updated;
        });
      }
    } catch (err) {
      setError(
        err.message.includes("Failed to fetch")
          ? "Can't reach the backend. Make sure `uvicorn api.main:app --reload --port 8000` is running."
          : err.message
      );
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  }

  async function handlePdfUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported.");
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const data = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener("progress", (event) => {
          if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            setUploadProgress(percent);
          }
        });

        xhr.addEventListener("load", () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve(JSON.parse(xhr.responseText));
            } catch {
              reject(new Error("Invalid response from server."));
            }
          } else {
            reject(new Error(`Server error: ${xhr.status}`));
          }
        });

        xhr.addEventListener("error", () => reject(new Error("Upload failed.")));

        xhr.open("POST", `${API_URL}/upload-pdf`);
        xhr.send(formData);
      });

      if (data.error) {
        setError(data.error);
      } else {
        setUploadedPdfs((prev) => [...prev, data.filename]);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `📄 Uploaded and indexed **${data.filename}** (${data.chunks_indexed} chunks). You can now ask questions about it.`,
          },
        ]);
      }
    } catch (err) {
      setError("Failed to upload PDF. Check that the backend is running.");
    } finally {
      setUploading(false);
      setUploadProgress(0);
      e.target.value = "";
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function clearChat() {
    setMessages([]);
    localStorage.removeItem(STORAGE_KEY);
  }

  return (
    <div className="flex flex-col h-screen bg-[#0a0a0f] text-neutral-100 overflow-hidden">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 left-1/4 w-[600px] h-[600px] bg-indigo-600/10 rounded-full blur-[120px]" />
        <div className="absolute top-1/3 -right-40 w-[500px] h-[500px] bg-purple-600/10 rounded-full blur-[120px]" />
      </div>

      <header className="relative z-10 flex items-center justify-between px-6 md:px-10 py-4 border-b border-white/[0.06] backdrop-blur-xl bg-black/20">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-lg shadow-lg shadow-indigo-500/20">
            🤖
          </div>
          <div className="text-left">
            <h1 className="text-[15px] font-semibold tracking-tight leading-none">
              Codebase RAG Assistant
            </h1>
            <p className="text-[12px] text-neutral-500 mt-1 leading-none">
              Agentic RAG · Hybrid Search · Live GitHub
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <input
            type="file"
            accept=".pdf"
            ref={fileInputRef}
            onChange={handlePdfUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="relative text-xs px-3.5 py-1.5 rounded-full border border-white/10 text-neutral-400 hover:text-neutral-100 hover:border-white/20 hover:bg-white/5 transition-all duration-200 disabled:opacity-70 overflow-hidden"
          >
            {uploading && (
              <div
                className="absolute inset-0 bg-indigo-500/30 transition-all duration-150"
                style={{ width: `${uploadProgress}%` }}
              />
            )}
            <span className="relative z-10">
              {uploading ? `Uploading ${uploadProgress}%` : "📄 Upload PDF"}
            </span>
          </button>
          <button
            onClick={clearChat}
            className="text-xs px-3.5 py-1.5 rounded-full border border-white/10 text-neutral-400 hover:text-neutral-100 hover:border-white/20 hover:bg-white/5 transition-all duration-200"
          >
            Clear chat
          </button>
        </div>
      </header>

      <main className="relative z-10 flex-1 overflow-y-auto">
        <div className="max-w-3xl w-full mx-auto px-6 md:px-10 py-8 space-y-5">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center text-center pt-20 pb-10">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-purple-600/20 border border-white/10 flex items-center justify-center text-2xl mb-5">
                💬
              </div>
              <h2 className="text-xl font-medium text-neutral-200 mb-2">
                Ask about the LangChain codebase
              </h2>
              <p className="text-sm text-neutral-500 mb-8 max-w-sm">
                Grounded answers with real file paths, powered by hybrid semantic + keyword search.
              </p>
              <div className="flex flex-col gap-2 w-full max-w-md">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => sendMessage(ex)}
                    className="text-left text-sm px-4 py-3 rounded-xl border border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/[0.15] transition-all duration-200 text-neutral-300"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => {
            const isStreamingEmpty =
              loading && i === messages.length - 1 && m.role === "assistant" && m.content === "";
            return (
              <div
                key={i}
                className={`msg-in flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {m.role === "assistant" && (
                  <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-xs mr-2.5 mt-0.5 flex-shrink-0 shadow-md shadow-indigo-500/20">
                    🤖
                  </div>
                )}
                <div
                  className={`max-w-[78%] rounded-2xl px-4 py-3 text-[14px] whitespace-pre-wrap leading-relaxed shadow-sm ${
                    m.role === "user"
                      ? "bg-gradient-to-br from-indigo-600 to-indigo-700 text-white rounded-tr-sm"
                      : "bg-white/[0.04] border border-white/[0.06] text-neutral-200 rounded-tl-sm"
                  }`}
                >
                  {isStreamingEmpty ? (
                    <div className="flex items-center gap-1.5 px-1 py-0.5">
                      <span className="dot w-1.5 h-1.5 rounded-full bg-neutral-400" style={{ animationDelay: "0ms" }} />
                      <span className="dot w-1.5 h-1.5 rounded-full bg-neutral-400" style={{ animationDelay: "160ms" }} />
                      <span className="dot w-1.5 h-1.5 rounded-full bg-neutral-400" style={{ animationDelay: "320ms" }} />
                    </div>
                  ) : (
                    <MarkdownContent text={m.content} />
                  )}
                </div>
              </div>
            );
          })}

          {error && (
            <div className="msg-in bg-red-500/10 border border-red-500/20 text-red-300 rounded-xl px-4 py-3 text-sm">
              {error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="relative z-10 border-t border-white/[0.06] backdrop-blur-xl bg-black/20 px-6 md:px-10 py-5">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-2 rounded-2xl border border-white/[0.08] bg-white/[0.03] focus-within:border-indigo-500/50 focus-within:bg-white/[0.05] transition-all duration-200 px-2 py-2">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about the codebase..."
              rows={1}
              className="flex-1 resize-none bg-transparent px-3 py-2 text-[14px] text-neutral-100 placeholder-neutral-500 focus:outline-none max-h-40"
            />
            <button
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              className="flex-shrink-0 w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white flex items-center justify-center hover:shadow-lg hover:shadow-indigo-500/30 disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none transition-all duration-200"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M22 2 11 13" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M22 2 15 22 11 13 2 9 22 2Z" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
          <p className="text-[11px] text-neutral-600 text-center mt-2.5">
            Enter to send · Shift+Enter for new line
          </p>
        </div>
      </footer>
    </div>
  );
}