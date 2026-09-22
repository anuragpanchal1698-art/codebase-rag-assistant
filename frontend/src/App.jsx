import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

/* ============================================================
   MARKDOWN
============================================================ */

function MarkdownContent({ text }) {
  return (
    <div
      className="
        prose prose-invert prose-sm max-w-none
        prose-p:my-1.5
        prose-headings:mt-3 prose-headings:mb-1.5
        prose-ul:my-1.5 prose-ol:my-1.5
        prose-li:my-0.5
        prose-strong:text-white
        prose-code:text-indigo-300
        prose-code:bg-black/30
        prose-code:px-1.5
        prose-code:py-0.5
        prose-code:rounded-md
        prose-code:before:content-none
        prose-code:after:content-none
        prose-pre:bg-[#080910]
        prose-pre:border
        prose-pre:border-white/[0.08]
        prose-table:my-2
        prose-th:px-3 prose-th:py-1.5
        prose-td:px-3 prose-td:py-1.5
        prose-th:bg-white/[0.06]
        prose-th:border-white/[0.08]
        prose-td:border-white/[0.08]
      "
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

/* ============================================================
   ICONS
============================================================ */

function Icon({ name, size = 18 }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round",
    strokeLinejoin: "round",
  };

  const paths = {
    menu: (
      <>
        <path d="M4 6h16" />
        <path d="M4 12h16" />
        <path d="M4 18h16" />
      </>
    ),

    plus: (
      <>
        <path d="M12 5v14" />
        <path d="M5 12h14" />
      </>
    ),

    message: (
      <>
        <path d="M20 11.5a7.5 7.5 0 0 1-8 7.5 8.5 8.5 0 0 1-3.7-.85L4 20l1.8-3.55A7.25 7.25 0 0 1 4 11.5 7.5 7.5 0 0 1 12 4a7.5 7.5 0 0 1 8 7.5Z" />
      </>
    ),

    trash: (
      <>
        <path d="M4 7h16" />
        <path d="M10 11v6" />
        <path d="M14 11v6" />
        <path d="M6 7l1 13h10l1-13" />
        <path d="M9 7V4h6v3" />
      </>
    ),

    upload: (
      <>
        <path d="M12 16V4" />
        <path d="m7 9 5-5 5 5" />
        <path d="M5 20h14" />
      </>
    ),

    send: (
      <>
        <path d="M22 2 11 13" />
        <path d="m22 2-7 20-4-9-9-4 20-7Z" />
      </>
    ),

    file: (
      <>
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" />
        <path d="M14 2v6h6" />
        <path d="M8 13h8" />
        <path d="M8 17h6" />
      </>
    ),

    chevron: (
      <path d="m9 18 6-6-6-6" />
    ),
  };

  return <svg {...common}>{paths[name]}</svg>;
}

/* ============================================================
   APP
============================================================ */

export default function App() {
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);

  const [messages, setMessages] = useState([]);
  const [uploadedPdfs, setUploadedPdfs] = useState([]);

  const [input, setInput] = useState("");

  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const [error, setError] = useState(null);

  const [sidebarOpen, setSidebarOpen] = useState(true);

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  /* ==========================================================
     LOAD CHATS
  ========================================================== */

  useEffect(() => {
    loadChats();
  }, []);

  async function loadChats() {
    try {
      const response = await fetch(`${API_URL}/chats`);

      if (!response.ok) {
        throw new Error("Failed to load chats.");
      }

      const data = await response.json();

      const chatList = data.chats || [];

      setChats(chatList);

      if (chatList.length > 0) {
        await selectChat(chatList[0].id);
      } else {
        await createNewChat();
      }
    } catch (err) {
      setError(
        "Unable to connect to the backend. Make sure FastAPI is running."
      );
    }
  }

  /* ==========================================================
     CREATE CHAT
  ========================================================== */

  async function createNewChat() {
    if (loading || uploading) return;

    try {
      setError(null);

      const response = await fetch(`${API_URL}/chats`, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error("Failed to create chat.");
      }

      const chat = await response.json();

      setChats((prev) => [chat, ...prev]);
      setCurrentChatId(chat.id);
      setMessages([]);
      setUploadedPdfs([]);
      setInput("");

      if (window.innerWidth < 900) {
        setSidebarOpen(false);
      }
    } catch (err) {
      setError("Unable to create a new chat.");
    }
  }

  /* ==========================================================
     SELECT CHAT
  ========================================================== */

  async function selectChat(chatId) {
    if (!chatId || loading || uploading) return;

    try {
      setError(null);

      const response = await fetch(
        `${API_URL}/chats/${chatId}`
      );

      if (!response.ok) {
        throw new Error("Failed to load chat.");
      }

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setCurrentChatId(chatId);

      setMessages(
        (data.messages || []).map((message) => ({
          role: message.role,
          content: message.content,
        }))
      );

      await loadPdfs(chatId);

      if (window.innerWidth < 900) {
        setSidebarOpen(false);
      }
    } catch (err) {
      setError(err.message || "Unable to load chat.");
    }
  }

  /* ==========================================================
     LOAD PDF LIST
  ========================================================== */

  async function loadPdfs(chatId) {
    if (!chatId) return;

    try {
      const response = await fetch(
        `${API_URL}/pdfs?chat_id=${encodeURIComponent(chatId)}`
      );

      if (!response.ok) return;

      const data = await response.json();

      setUploadedPdfs(data.pdfs || []);
    } catch {
      setUploadedPdfs([]);
    }
  }

  /* ==========================================================
     DELETE CHAT
  ========================================================== */

  async function deleteCurrentChat(chatId, event) {
    event?.stopPropagation();

    if (!chatId || loading || uploading) return;

    const confirmed = window.confirm(
      "Delete this chat and all PDFs uploaded to it?"
    );

    if (!confirmed) return;

    try {
      setError(null);

      const response = await fetch(
        `${API_URL}/chats/${chatId}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error("Failed to delete chat.");
      }

      const remaining = chats.filter(
        (chat) => chat.id !== chatId
      );

      setChats(remaining);

      if (currentChatId === chatId) {
        if (remaining.length > 0) {
          await selectChat(remaining[0].id);
        } else {
          await createNewChat();
        }
      }
    } catch (err) {
      setError(
        err.message || "Unable to delete the chat."
      );
    }
  }

  /* ==========================================================
     SEND MESSAGE
  ========================================================== */

  async function sendMessage(overrideText) {
    const text = (
      overrideText !== undefined
        ? overrideText
        : input
    ).trim();

    if (
      !text ||
      loading ||
      uploading ||
      !currentChatId
    ) {
      return;
    }

    const previousMessages = [...messages];

    const userMessage = {
      role: "user",
      content: text,
    };

    setMessages([
      ...previousMessages,
      userMessage,
      {
        role: "assistant",
        content: "",
      },
    ]);

    setInput("");
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_URL}/chat/stream`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            chat_id: currentChatId,
            message: text,
            history: previousMessages,
          }),
        }
      );

      if (!response.ok || !response.body) {
        throw new Error(
          `Server error: ${response.status}`
        );
      }

      const reader =
        response.body.getReader();

      const decoder =
        new TextDecoder();

      let accumulated = "";

      while (true) {
        const { done, value } =
          await reader.read();

        if (done) break;

        accumulated += decoder.decode(
          value,
          { stream: true }
        );

        setMessages((prev) => {
          const updated = [...prev];

          updated[
            updated.length - 1
          ] = {
            role: "assistant",
            content: accumulated,
          };

          return updated;
        });
      }

      accumulated += decoder.decode();

      if (!accumulated.trim()) {
        setMessages((prev) => {
          const updated = [...prev];

          updated[
            updated.length - 1
          ] = {
            role: "assistant",
            content:
              "No response was generated. Try asking the question again.",
          };

          return updated;
        });
      }

      /*
       * Refresh sidebar so the automatically generated
       * chat title and updated timestamp appear.
       */
      await refreshChats();
    } catch (err) {
      setMessages(previousMessages);

      setError(
        err.message.includes(
          "Failed to fetch"
        )
          ? "Can't reach the backend. Make sure FastAPI is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  /* ==========================================================
     REFRESH SIDEBAR
  ========================================================== */

  async function refreshChats() {
    try {
      const response = await fetch(
        `${API_URL}/chats`
      );

      if (!response.ok) return;

      const data = await response.json();

      setChats(data.chats || []);
    } catch {
      // Sidebar refresh failure should not break chat.
    }
  }

  /* ==========================================================
     PDF UPLOAD
  ========================================================== */

  async function handlePdfUpload(event) {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!currentChatId) {
      setError("Create a chat before uploading a PDF.");
      event.target.value = "";
      return;
    }

    if (
      !file.name
        .toLowerCase()
        .endsWith(".pdf")
    ) {
      setError("Only PDF files are supported.");
      event.target.value = "";
      return;
    }

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    const formData = new FormData();

    formData.append(
      "file",
      file
    );

    formData.append(
      "chat_id",
      currentChatId
    );

    try {
      const data = await new Promise(
        (resolve, reject) => {
          const xhr =
            new XMLHttpRequest();

          xhr.upload.addEventListener(
            "progress",
            (event) => {
              if (
                event.lengthComputable
              ) {
                const percent =
                  Math.round(
                    (event.loaded /
                      event.total) *
                      100
                  );

                setUploadProgress(
                  percent
                );
              }
            }
          );

          xhr.addEventListener(
            "load",
            () => {
              if (
                xhr.status >= 200 &&
                xhr.status < 300
              ) {
                try {
                  resolve(
                    JSON.parse(
                      xhr.responseText
                    )
                  );
                } catch {
                  reject(
                    new Error(
                      "Invalid response from server."
                    )
                  );
                }
              } else {
                reject(
                  new Error(
                    `Server error: ${xhr.status}`
                  )
                );
              }
            }
          );

          xhr.addEventListener(
            "error",
            () =>
              reject(
                new Error(
                  "Upload failed."
                )
              )
          );

          xhr.open(
            "POST",
            `${API_URL}/upload-pdf`
          );

          xhr.send(formData);
        }
      );

      if (data.error) {
        setError(data.error);
        return;
      }

      await loadPdfs(currentChatId);

      /*
       * This message is only visual feedback.
       * The actual user/assistant conversation history
       * remains controlled by the backend.
       */
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            `📄 **${data.filename}** uploaded and indexed successfully.\n\n` +
            `${data.chunks_indexed} chunks are now available for questions in this chat.`,
        },
      ]);
    } catch (err) {
      setError(
        err.message ||
          "Failed to upload PDF."
      );
    } finally {
      setUploading(false);
      setUploadProgress(0);
      event.target.value = "";
    }
  }

  /* ==========================================================
     KEYBOARD
  ========================================================== */

  function handleKeyDown(event) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();
      sendMessage();
    }
  }

  /* ==========================================================
     AUTO RESIZE TEXTAREA
  ========================================================== */

  useEffect(() => {
    if (!textareaRef.current) return;

    textareaRef.current.style.height =
      "auto";

    textareaRef.current.style.height =
      `${Math.min(
        textareaRef.current.scrollHeight,
        180
      )}px`;
  }, [input]);

  /* ==========================================================
     AUTO SCROLL
  ========================================================== */

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  /* ==========================================================
     CURRENT CHAT
  ========================================================== */

  const currentChat =
    chats.find(
      (chat) =>
        chat.id === currentChatId
    );

  /* ==========================================================
     RENDER
  ========================================================== */

  return (
    <div className="relative flex h-screen overflow-hidden bg-[#07080d] text-neutral-100">

      {/* ======================================================
          BACKGROUND GLOW
      ====================================================== */}

      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div
          className="
            absolute -top-64 left-[18%]
            h-[650px] w-[650px]
            rounded-full
            bg-indigo-600/[0.07]
            blur-[140px]
          "
        />

        <div
          className="
            absolute top-[35%] -right-64
            h-[600px] w-[600px]
            rounded-full
            bg-blue-500/[0.045]
            blur-[140px]
          "
        />

        <div
          className="
            absolute -bottom-72 left-[35%]
            h-[550px] w-[550px]
            rounded-full
            bg-violet-600/[0.035]
            blur-[130px]
          "
        />
      </div>

      {/* ======================================================
          MOBILE OVERLAY
      ====================================================== */}

      {sidebarOpen && (
        <div
          className="
            fixed inset-0 z-30
            bg-black/60
            backdrop-blur-sm
            lg:hidden
          "
          onClick={() =>
            setSidebarOpen(false)
          }
        />
      )}

      {/* ======================================================
          SIDEBAR
      ====================================================== */}

      <aside
        className={`
          fixed lg:relative z-40
          flex h-full flex-col
          border-r border-white/[0.07]
          bg-[#090a10]/95
          backdrop-blur-2xl
          transition-all duration-300
          ${sidebarOpen
            ? "w-[285px] translate-x-0"
            : "w-0 -translate-x-full lg:w-[72px] lg:translate-x-0"
          }
        `}
      >
        {/* SIDEBAR HEADER */}

        <div className="flex h-[72px] items-center gap-3 border-b border-white/[0.06] px-4">

          <div
            className="
              flex h-9 w-9 shrink-0
              items-center justify-center
              rounded-xl
              border border-indigo-400/20
              bg-gradient-to-br
              from-indigo-500
              to-blue-600
              text-sm
              shadow-lg shadow-indigo-600/20
            "
          >
            ✦
          </div>

          {sidebarOpen && (
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold tracking-tight">
                RAG Assistant
              </div>

              <div className="mt-0.5 text-[10px] uppercase tracking-[0.16em] text-neutral-600">
                PDF Intelligence
              </div>
            </div>
          )}
        </div>

        {/* NEW CHAT */}

        <div className="p-3">
          <button
            onClick={createNewChat}
            disabled={loading || uploading}
            className="
              group flex w-full items-center gap-3
              rounded-xl
              border border-white/[0.08]
              bg-white/[0.035]
              px-3 py-2.5
              text-sm text-neutral-300
              transition-all duration-200
              hover:border-indigo-400/20
              hover:bg-indigo-500/[0.07]
              hover:text-white
              disabled:cursor-not-allowed
              disabled:opacity-50
            "
          >
            <span className="
              flex h-7 w-7 shrink-0 items-center justify-center
              rounded-lg bg-white/[0.06]
              text-neutral-400
              transition-colors
              group-hover:bg-indigo-500/20
              group-hover:text-indigo-300
            ">
              <Icon name="plus" size={16} />
            </span>

            {sidebarOpen && (
              <span className="font-medium">
                New chat
              </span>
            )}
          </button>
        </div>

        {/* CHAT LIST */}

        {sidebarOpen && (
          <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">

            <div className="px-2 pb-2 pt-2 text-[10px] font-semibold uppercase tracking-[0.15em] text-neutral-600">
              Conversations
            </div>

            <div className="space-y-1">
              {chats.map((chat) => {
                const active =
                  chat.id ===
                  currentChatId;

                return (
                  <button
                    key={chat.id}
                    onClick={() =>
                      selectChat(
                        chat.id
                      )
                    }
                    className={`
                      group relative flex w-full
                      items-center gap-2
                      rounded-xl px-3 py-2.5
                      text-left
                      transition-all duration-200
                      ${
                        active
                          ? "bg-indigo-500/[0.10] text-neutral-100"
                          : "text-neutral-500 hover:bg-white/[0.035] hover:text-neutral-300"
                      }
                    `}
                  >
                    {active && (
                      <span
                        className="
                          absolute left-0
                          h-6 w-[2px]
                          rounded-full
                          bg-indigo-400
                          shadow-[0_0_12px_rgba(129,140,248,0.8)]
                        "
                      />
                    )}

                    <span
                      className={`
                        shrink-0
                        ${
                          active
                            ? "text-indigo-400"
                            : "text-neutral-600"
                        }
                      `}
                    >
                      <Icon
                        name="message"
                        size={16}
                      />
                    </span>

                    <span className="min-w-0 flex-1 truncate text-[13px]">
                      {chat.title ||
                        "New Chat"}
                    </span>

                    <span
                      onClick={(event) =>
                        deleteCurrentChat(
                          chat.id,
                          event
                        )
                      }
                      className="
                        hidden shrink-0
                        rounded-md p-1
                        text-neutral-700
                        hover:bg-red-500/10
                        hover:text-red-400
                        group-hover:block
                      "
                    >
                      <Icon
                        name="trash"
                        size={14}
                      />
                    </span>
                  </button>
                );
              })}
            </div>

            {chats.length === 0 && (
              <div className="px-3 py-8 text-center text-xs text-neutral-600">
                No conversations yet.
              </div>
            )}
          </div>
        )}

        {/* SIDEBAR FOOTER */}

        {sidebarOpen && (
          <div className="border-t border-white/[0.06] p-3">
            <div className="
              rounded-xl
              border border-white/[0.05]
              bg-white/[0.02]
              px-3 py-2.5
            ">
              <div className="flex items-center gap-2">
                <span className="
                  h-1.5 w-1.5 rounded-full
                  bg-emerald-400
                  shadow-[0_0_8px_rgba(52,211,153,0.8)]
                " />

                <span className="text-[11px] text-neutral-500">
                  PDF-only knowledge mode
                </span>
              </div>
            </div>
          </div>
        )}
      </aside>

      {/* ======================================================
          MAIN AREA
      ====================================================== */}

      <section className="relative z-10 flex min-w-0 flex-1 flex-col">

        {/* ====================================================
            TOP BAR
        ==================================================== */}

        <header
          className="
            flex h-[72px] shrink-0
            items-center justify-between
            border-b border-white/[0.06]
            bg-[#08090e]/70
            px-4 sm:px-6
            backdrop-blur-2xl
          "
        >
          <div className="flex min-w-0 items-center gap-3">

            <button
              onClick={() =>
                setSidebarOpen(
                  !sidebarOpen
                )
              }
              className="
                flex h-9 w-9 shrink-0
                items-center justify-center
                rounded-lg
                border border-white/[0.07]
                bg-white/[0.025]
                text-neutral-500
                transition-all
                hover:bg-white/[0.06]
                hover:text-neutral-200
              "
            >
              <Icon
                name="menu"
                size={18}
              />
            </button>

            <div className="min-w-0">
              <div className="truncate text-sm font-semibold text-neutral-200">
                {currentChat?.title ||
                  "New Chat"}
              </div>

              <div className="mt-0.5 flex items-center gap-2 text-[10px] text-neutral-600">
                <span className="
                  h-1.5 w-1.5 rounded-full
                  bg-emerald-400
                " />

                <span>
                  PDF grounded
                </span>

                {uploadedPdfs.length >
                  0 && (
                  <>
                    <span>
                      ·
                    </span>

                    <span>
                      {
                        uploadedPdfs.length
                      }{" "}
                      PDF
                      {uploadedPdfs.length !==
                      1
                        ? "s"
                        : ""}
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* HEADER ACTIONS */}

          <div className="flex shrink-0 items-center gap-2">

            <input
              type="file"
              accept=".pdf,application/pdf"
              ref={fileInputRef}
              onChange={
                handlePdfUpload
              }
              className="hidden"
            />

            <button
              onClick={() =>
                fileInputRef.current?.click()
              }
              disabled={
                uploading ||
                loading ||
                !currentChatId
              }
              className="
                relative flex items-center gap-2
                overflow-hidden
                rounded-xl
                border border-white/[0.08]
                bg-white/[0.035]
                px-3 py-2
                text-xs font-medium
                text-neutral-400
                transition-all duration-200
                hover:border-indigo-400/20
                hover:bg-indigo-500/[0.07]
                hover:text-indigo-200
                disabled:cursor-not-allowed
                disabled:opacity-50
              "
            >
              {uploading && (
                <span
                  className="
                    absolute inset-y-0 left-0
                    bg-indigo-500/10
                    transition-all
                  "
                  style={{
                    width: `${uploadProgress}%`,
                  }}
                />
              )}

              <span className="relative flex items-center gap-2">
                <Icon
                  name="upload"
                  size={15}
                />

                <span className="hidden sm:inline">
                  {uploading
                    ? `${uploadProgress}%`
                    : "Upload PDF"}
                </span>
              </span>
            </button>
          </div>
        </header>

        {/* ====================================================
            PDF STRIP
        ==================================================== */}

        {uploadedPdfs.length > 0 && (
          <div
            className="
              flex shrink-0 items-center gap-2
              overflow-x-auto
              border-b border-white/[0.04]
              bg-white/[0.012]
              px-4 sm:px-6
              py-2
            "
          >
            <span className="mr-1 shrink-0 text-[10px] uppercase tracking-[0.12em] text-neutral-700">
              Sources
            </span>

            {uploadedPdfs.map(
              (pdf) => (
                <div
                  key={pdf}
                  className="
                    flex shrink-0
                    items-center gap-1.5
                    rounded-lg
                    border border-white/[0.06]
                    bg-white/[0.025]
                    px-2.5 py-1.5
                    text-[11px]
                    text-neutral-500
                  "
                >
                  <Icon
                    name="file"
                    size={13}
                  />

                  <span className="max-w-[180px] truncate">
                    {pdf}
                  </span>
                </div>
              )
            )}
          </div>
        )}

        {/* ====================================================
            MESSAGES
        ==================================================== */}

        <main className="min-h-0 flex-1 overflow-y-auto">

          <div className="mx-auto w-full max-w-4xl px-4 py-8 sm:px-8 sm:py-10">

            {messages.length === 0 ? (
              <EmptyState
                onExample={sendMessage}
              />
            ) : (
              <div className="space-y-6">

                {messages.map(
                  (message, index) => {
                    const isUser =
                      message.role ===
                      "user";

                    const isEmptyStreaming =
                      loading &&
                      index ===
                        messages.length -
                          1 &&
                      message.role ===
                        "assistant" &&
                      message.content ===
                        "";

                    return (
                      <div
                        key={`${index}-${message.role}`}
                        className={`
                          flex
                          ${
                            isUser
                              ? "justify-end"
                              : "justify-start"
                          }
                        `}
                      >
                        {!isUser && (
                          <div className="mr-3 mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-indigo-400/15 bg-gradient-to-br from-indigo-500/20 to-blue-500/10 text-indigo-300 shadow-lg shadow-indigo-500/[0.06]">
                            ✦
                          </div>
                        )}

                        <div
                          className={`
                            ${
                              isUser
                                ? "max-w-[82%] sm:max-w-[75%]"
                                : "max-w-[88%] sm:max-w-[78%]"
                            }
                          `}
                        >
                          <div
                            className={`
                              rounded-2xl px-4 py-3.5
                              text-[14px]
                              leading-relaxed
                              ${
                                isUser
                                  ? `
                                    rounded-tr-md
                                    bg-gradient-to-br
                                    from-indigo-600
                                    to-indigo-700
                                    text-white
                                    shadow-lg
                                    shadow-indigo-900/20
                                  `
                                  : `
                                    rounded-tl-md
                                    border
                                    border-white/[0.065]
                                    bg-white/[0.032]
                                    text-neutral-200
                                    shadow-sm
                                  `
                              }
                            `}
                          >
                            {isEmptyStreaming ? (
                              <TypingIndicator />
                            ) : isUser ? (
                              <div className="whitespace-pre-wrap">
                                {message.content}
                              </div>
                            ) : (
                              <MarkdownContent
                                text={
                                  message.content
                                }
                              />
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  }
                )}

                <div
                  ref={bottomRef}
                />
              </div>
            )}

            {error && (
              <div className="
                mt-5
                rounded-xl
                border border-red-500/15
                bg-red-500/[0.06]
                px-4 py-3
                text-sm text-red-300
              ">
                {error}
              </div>
            )}
          </div>
        </main>

        {/* ====================================================
            COMPOSER
        ==================================================== */}

        <footer
          className="
            shrink-0
            border-t border-white/[0.06]
            bg-[#08090e]/80
            px-4 pb-4 pt-3
            backdrop-blur-2xl
            sm:px-6
          "
        >
          <div className="mx-auto max-w-4xl">

            <div
              className="
                relative
                flex items-end gap-2
                rounded-2xl
                border border-white/[0.08]
                bg-white/[0.035]
                p-2
                shadow-2xl
                shadow-black/20
                transition-all duration-200
                focus-within:border-indigo-400/25
                focus-within:bg-white/[0.045]
                focus-within:shadow-indigo-950/10
              "
            >

              {/* ATTACH */}

              <button
                onClick={() =>
                  fileInputRef.current?.click()
                }
                disabled={
                  loading ||
                  uploading ||
                  !currentChatId
                }
                title="Upload PDF"
                className="
                  mb-0.5
                  flex h-9 w-9
                  shrink-0
                  items-center justify-center
                  rounded-xl
                  text-neutral-600
                  transition-all
                  hover:bg-white/[0.06]
                  hover:text-indigo-300
                  disabled:opacity-30
                "
              >
                <Icon
                  name="upload"
                  size={17}
                />
              </button>

              {/* TEXTAREA */}

              <textarea
                ref={textareaRef}
                value={input}
                onChange={(event) =>
                  setInput(
                    event.target.value
                  )
                }
                onKeyDown={
                  handleKeyDown
                }
                disabled={
                  loading ||
                  !currentChatId
                }
                rows={1}
                placeholder={
                  currentChatId
                    ? uploadedPdfs.length
                      ? "Ask anything about your PDF..."
                      : "Upload a PDF, then ask a question..."
                    : "Create a new chat..."
                }
                className="
                  max-h-[180px]
                  min-h-[40px]
                  flex-1
                  resize-none
                  bg-transparent
                  px-2 py-2
                  text-sm
                  text-neutral-100
                  outline-none
                  placeholder:text-neutral-600
                  disabled:cursor-not-allowed
                "
              />

              {/* SEND */}

              <button
                onClick={() =>
                  sendMessage()
                }
                disabled={
                  loading ||
                  uploading ||
                  !input.trim() ||
                  !currentChatId
                }
                className="
                  mb-0.5
                  flex h-9 w-9
                  shrink-0
                  items-center justify-center
                  rounded-xl
                  bg-gradient-to-br
                  from-indigo-500
                  to-blue-600
                  text-white
                  shadow-lg
                  shadow-indigo-600/20
                  transition-all duration-200
                  hover:scale-[1.03]
                  hover:shadow-indigo-500/30
                  disabled:scale-100
                  disabled:cursor-not-allowed
                  disabled:opacity-25
                  disabled:shadow-none
                "
              >
                <Icon
                  name="send"
                  size={16}
                />
              </button>
            </div>

            <div className="mt-2 flex items-center justify-center gap-2 text-[10px] text-neutral-700">
              <span>
                Enter to send
              </span>

              <span>·</span>

              <span>
                Shift + Enter for new line
              </span>

              <span>·</span>

              <span>
                Answers grounded in your PDF
              </span>
            </div>
          </div>
        </footer>
      </section>
    </div>
  );
}

/* ============================================================
   EMPTY STATE
============================================================ */

function EmptyState({ onExample }) {
  const examples = [
    "What is this document about?",
    "Summarize the main points.",
    "Explain the most important concept in the PDF.",
  ];

  return (
    <div className="flex min-h-[55vh] flex-col items-center justify-center text-center">

      <div
        className="
          relative mb-7
          flex h-16 w-16
          items-center justify-center
          rounded-2xl
          border border-indigo-400/15
          bg-gradient-to-br
          from-indigo-500/10
          to-blue-500/[0.04]
          text-2xl text-indigo-300
          shadow-2xl
          shadow-indigo-950/20
        "
      >
        <span className="relative z-10">
          ✦
        </span>

        <span
          className="
            absolute inset-0
            rounded-2xl
            bg-indigo-500/10
            blur-xl
          "
        />
      </div>

      <h1 className="text-2xl font-semibold tracking-tight text-neutral-100 sm:text-3xl">
        Ask your documents.
      </h1>

      <p className="mt-3 max-w-md text-sm leading-6 text-neutral-500">
        Upload a PDF and ask questions about
        its content. Every conversation has
        its own isolated document context.
      </p>

      <div className="mt-8 grid w-full max-w-xl gap-2 sm:grid-cols-3">
        {examples.map(
          (example) => (
            <button
              key={example}
              onClick={() =>
                onExample(example)
              }
              className="
                rounded-xl
                border border-white/[0.07]
                bg-white/[0.025]
                px-3 py-3
                text-left text-xs
                leading-5 text-neutral-500
                transition-all duration-200
                hover:border-indigo-400/15
                hover:bg-indigo-500/[0.05]
                hover:text-neutral-300
              "
            >
              {example}
            </button>
          )
        )}
      </div>
    </div>
  );
}

/* ============================================================
   TYPING INDICATOR
============================================================ */

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-1 py-1">
      <span
        className="
          h-1.5 w-1.5 rounded-full
          bg-indigo-400/70
          animate-pulse
        "
      />

      <span
        className="
          h-1.5 w-1.5 rounded-full
          bg-indigo-400/70
          animate-pulse
          [animation-delay:150ms]
        "
      />

      <span
        className="
          h-1.5 w-1.5 rounded-full
          bg-indigo-400/70
          animate-pulse
          [animation-delay:300ms]
        "
      />
    </div>
  );
}