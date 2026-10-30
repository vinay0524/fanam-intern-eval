import React, { useState, useRef, useEffect } from 'react';
import { 
  MessageSquare, FileText, UploadCloud, MoreVertical, 
  Send, Sun, Moon, ChevronDown, User, Bot, Loader2, Trash2,
  Eye, X
} from 'lucide-react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const ALLOWED_EXTENSIONS = ['docx', 'pdf', 'csv', 'txt', 'md', 'xlsx', 'xls'];

function parseInlineMarkdown(str) {
  if (!str) return '';
  return str
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}

function FormattedAnswer({ text }) {
  if (!text) return null;
  const blocks = text.split(/\n\n+/);

  return (
    <div className="formatted-answer">
      {blocks.map((block, bIdx) => {
        const trimmed = block.trim();
        if (!trimmed) return null;

        const lines = trimmed.split('\n').map(l => l.trim()).filter(Boolean);
        const isList = lines.length > 0 && lines.every(l => /^[-*•]|\d+\.\s*/.test(l));

        if (isList) {
          return (
            <ul key={bIdx} className="formatted-list">
              {lines.map((line, lIdx) => {
                const cleaned = line.replace(/^[-*•]|\d+\.\s*/, '').trim();
                return (
                  <li 
                    key={lIdx} 
                    dangerouslySetInnerHTML={{ __html: parseInlineMarkdown(cleaned) }} 
                  />
                );
              })}
            </ul>
          );
        }

        return (
          <p 
            key={bIdx} 
            className="formatted-paragraph" 
            dangerouslySetInnerHTML={{ __html: parseInlineMarkdown(trimmed.replace(/\n/g, '<br/>')) }} 
          />
        );
      })}
    </div>
  );
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('app-theme') || 'light');
  const [dragActive, setDragActive] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [inputText, setInputText] = useState('');
  const [activeMenuDocId, setActiveMenuDocId] = useState(null);
  const [viewingDoc, setViewingDoc] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Close document menu dropdown when clicking anywhere outside
  useEffect(() => {
    const handleCloseMenu = () => setActiveMenuDocId(null);
    window.addEventListener('click', handleCloseMenu);
    return () => window.removeEventListener('click', handleCloseMenu);
  }, []);

  // Apply theme to document element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('app-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  // Real uploaded documents list (starts empty)
  const [documents, setDocuments] = useState([]);

  // Real chat messages stream (starts empty)
  const [messages, setMessages] = useState([]);

  // Real retrieved sources from Pinecone (starts empty)
  const [sources, setSources] = useState([]);

  // Auto scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Drag and drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      uploadFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      uploadFiles(Array.from(e.target.files));
    }
  };

  const uploadFiles = async (fileList) => {
    const validFiles = fileList.filter(f => {
      const ext = f.name.split('.').pop()?.toLowerCase();
      return ALLOWED_EXTENSIONS.includes(ext);
    });

    if (validFiles.length === 0) {
      alert(`No supported files selected. Supported formats: ${ALLOWED_EXTENSIONS.map(e => '.' + e).join(', ')}`);
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    validFiles.forEach(file => {
      formData.append('files', file);
    });

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      // Add all newly processed files to state
      const processedDocs = (data.files || []).map(f => {
        const sizeStr = f.file_size < 1024 * 1024 
          ? `${Math.max(1, Math.round(f.file_size / 1024))} KB` 
          : `${(f.file_size / (1024 * 1024)).toFixed(1)} MB`;

        return {
          id: `doc-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
          name: f.filename,
          size: sizeStr,
          date: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
          type: f.file_type || f.filename.split('.').pop()?.toUpperCase(),
          chunks: f.chunks_created,
          totalChars: f.total_characters,
          sampleExcerpt: f.sample_text || ''
        };
      });

      setDocuments(prev => [...processedDocs, ...prev]);

      // Informative assistant message
      const timeNow = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const fileNamesList = (data.files || []).map(f => f.filename).join(', ');

      setMessages(prev => [
        ...prev,
        {
          id: `sys-${Date.now()}`,
          sender: 'assistant',
          mainText: `Successfully indexed **${data.processed_files_count} file(s)** (${fileNamesList}) into Pinecone using Google Gemini & BM25 hybrid search!`,
          details: (data.files || []).map(f => `${f.filename} → ${f.chunks_created} chunks (${f.total_characters.toLocaleString()} chars)`),
          sources: [],
          time: timeNow
        }
      ]);
    } catch (err) {
      alert(`Error uploading documents: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isQuerying) return;

    const userQuery = inputText.trim();
    setInputText('');
    const timeNow = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // 1. Add user message
    const userMsgId = `usr-${Date.now()}`;
    setMessages(prev => [
      ...prev,
      {
        id: userMsgId,
        sender: 'user',
        text: userQuery,
        time: timeNow
      }
    ]);

    setIsQuerying(true);

    try {
      // 2. Query FastAPI Backend with hybrid search
      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userQuery,
          top_k: 5,
          alpha: 0.5
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Query failed');
      }

      // 3. Format sources
      const retrievedSources = (data.sources || []).map((s, i) => ({
        id: `ret-${Date.now()}-${i}`,
        doc: s.source || 'Document.docx',
        page: `Chunk #${s.chunk_index + 1}`,
        excerpt: `“${s.text}”`
      }));

      // Update right panel sources
      if (retrievedSources.length > 0) {
        setSources(retrievedSources);
      }

      // Add assistant response message with natural explanation
      setMessages(prev => [
        ...prev,
        {
          id: `asst-${Date.now()}`,
          sender: 'assistant',
          text: data.answer,
          sources: retrievedSources.slice(0, 2),
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);

    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          sender: 'assistant',
          mainText: `Error querying documents: ${err.message}`,
          sources: [],
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setSources([]);
  };

  return (
    <div className="app-wrapper">
      {/* 1. Top Navbar */}
      <header className="top-navbar">
        <div className="nav-brand">
          <div className="brand-icon">
            <MessageSquare size={17} />
          </div>
          <span className="brand-title">Financial Research Assistant</span>
        </div>

        <div className="nav-actions">
          <button 
            className="theme-toggle-btn" 
            onClick={toggleTheme} 
            title={theme === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode'}
          >
            {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
          </button>
          <div className="user-profile-badge">
            <div className="user-avatar">U</div>
            <span className="user-name">User</span>
            <ChevronDown size={14} className="user-chevron" />
          </div>
        </div>
      </header>

      {/* 2. Main Body with Sidebar + 3 Panels */}
      <div className="main-body">
        
        {/* Left Dark Sidebar - Chat Module Only */}
        <aside className="left-sidebar">
          <button className="sidebar-tab-btn active">
            <MessageSquare size={16} />
            <span>Chat</span>
          </button>
        </aside>

        {/* 3 Main Content Panels */}
        <div className="panels-container">

          {/* Left Panel: Documents */}
          <section className="ui-panel documents-panel">
            <div className="panel-header-simple">
              <FileText size={17} color="#0f172a" />
              <span>Documents</span>
            </div>

            <div className="documents-scrollable">
              {/* Upload Document Section */}
              <div>
                <h3 className="section-label">Upload Document</h3>
                <div 
                  className={`docx-dropzone ${dragActive ? 'dragging' : ''}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input 
                    type="file" 
                    multiple
                    ref={fileInputRef} 
                    onChange={handleFileInput} 
                    accept=".docx,.pdf,.csv,.txt,.xlsx,.md,.xls" 
                    style={{ display: 'none' }} 
                  />
                  <UploadCloud className="dropzone-cloud-icon" />
                  <p className="dropzone-text-primary">Drag and drop files here</p>
                  <span className="dropzone-text-or">or</span>
                  <button type="button" className="select-file-btn" disabled={isUploading}>
                    {isUploading ? 'Uploading...' : 'Select File'}
                  </button>
                  <p className="dropzone-text-hint">Supports .docx, .pdf, .csv, .txt, .xlsx</p>
                </div>
              </div>

              {/* Uploaded Documents List */}
              <div>
                <h3 className="section-label">Uploaded Documents ({documents.length})</h3>
                <div className="uploaded-docs-list">
                  {documents.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '1.25rem 0.5rem', color: '#94a3b8', fontSize: '0.78rem', lineHeight: 1.4, background: '#f8fafc', borderRadius: '8px', border: '1px dashed #e2e8f0' }}>
                      No documents uploaded yet.<br/>Upload a file above to begin.
                    </div>
                  ) : (
                    documents.map((doc) => (
                      <div key={doc.id} className="doc-list-item">
                        <div className="doc-item-left">
                          <div className="doc-item-icon-box">
                            <FileText size={16} />
                          </div>
                          <div className="doc-item-info">
                            <span className="doc-item-name" title={doc.name}>{doc.name}</span>
                            <span className="doc-item-meta">{doc.size} · {doc.date}</span>
                          </div>
                        </div>
                        <div className="doc-item-menu-container">
                          <button 
                            className={`doc-item-menu-btn ${activeMenuDocId === doc.id ? 'active' : ''}`} 
                            title="Document Options"
                            onClick={(e) => {
                              e.stopPropagation();
                              setActiveMenuDocId(activeMenuDocId === doc.id ? null : doc.id);
                            }}
                          >
                            <MoreVertical size={15} />
                          </button>

                          {activeMenuDocId === doc.id && (
                            <div className="doc-dropdown-menu" onClick={(e) => e.stopPropagation()}>
                              <button 
                                className="doc-dropdown-item view"
                                onClick={() => {
                                  setViewingDoc(doc);
                                  setActiveMenuDocId(null);
                                }}
                              >
                                <Eye size={13} />
                                <span>View</span>
                              </button>
                              <button 
                                className="doc-dropdown-item delete"
                                onClick={() => {
                                  setDocuments(prev => prev.filter(d => d.id !== doc.id));
                                  setActiveMenuDocId(null);
                                }}
                              >
                                <Trash2 size={13} />
                                <span>Delete</span>
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </section>

          {/* Center Panel: Chat Interface */}
          <section className="ui-panel chat-panel">
            <div className="chat-header">
              <div className="chat-header-left">
                <h2 className="chat-title">Chat with your documents</h2>
                <p className="chat-subtitle">Ask questions about your financial documents and get accurate answers with sources.</p>
              </div>
              {messages.length > 0 && (
                <button 
                  className="clear-chat-btn" 
                  onClick={handleClearChat} 
                  title="Clear all chat messages and sources"
                >
                  <Trash2 size={13} />
                  <span>Clear Chat</span>
                </button>
              )}
            </div>

            {/* Chat Stream */}
            <div className="chat-messages-container">
              {messages.length === 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8', textAlign: 'center', gap: '0.65rem', padding: '2rem' }}>
                  <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: '#eff6ff', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#2563eb' }}>
                    <MessageSquare size={24} />
                  </div>
                  <p style={{ fontWeight: 600, color: '#1e293b', fontSize: '0.95rem' }}>No messages yet</p>
                  <p style={{ fontSize: '0.82rem', maxWidth: '320px', lineHeight: 1.45, color: '#64748b' }}>
                    Upload your financial documents (.docx, .pdf, .csv, .txt) on the left and ask questions to get accurate AI answers with cited sources.
                  </p>
                </div>
              ) : (
                messages.map((m) => {
                if (m.sender === 'user') {
                  return (
                    <div key={m.id} className="message-row-user">
                      <div className="message-content-user-wrapper">
                        <div className="user-bubble">{m.text}</div>
                        <div className="user-avatar-bubble">
                          <User size={15} />
                        </div>
                      </div>
                      <span className="message-timestamp">{m.time}</span>
                    </div>
                  );
                }

                return (
                  <div key={m.id} className="message-row-assistant">
                    <div className="message-content-assistant-wrapper">
                      <div className="assistant-avatar-bubble">
                        <Bot size={15} />
                      </div>
                      <div className="assistant-bubble">
                        <FormattedAnswer text={m.text || m.mainText} />

                        {m.sources && m.sources.length > 0 && (
                          <div className="bubble-sources-section">
                            <p className="bubble-sources-title">Sources:</p>
                            <div className="bubble-source-chips">
                              {m.sources.map((src, idx) => (
                                <div key={idx} className="source-chip">
                                  <FileText size={12} color="#2563eb" />
                                  <span>{src.doc} ({src.page})</span>
                                </div>
                              ))}
                            </div>
                            {m.sources[0]?.excerpt && (
                              <p className="source-chip-excerpt">{m.sources[0].excerpt}</p>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                    <span className="message-timestamp" style={{ marginLeft: '2.5rem' }}>{m.time}</span>
                  </div>
                );
              }))}

              {isQuerying && (
                <div className="message-row-assistant">
                  <div className="message-content-assistant-wrapper">
                    <div className="assistant-avatar-bubble">
                      <Bot size={15} />
                    </div>
                    <div className="assistant-bubble" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#64748b' }}>
                      <Loader2 size={16} className="animate-spin" />
                      <span>Retrieving chunks & generating answer...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Chat Input Bar */}
            <div className="chat-input-bar-container">
              <form onSubmit={handleSendMessage} className="chat-input-form">
                <input
                  type="text"
                  className="chat-text-input"
                  placeholder="Ask a question about your documents..."
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                />
                <button
                  type="submit"
                  className="chat-send-btn"
                  disabled={!inputText.trim() || isQuerying}
                  title="Send Question"
                >
                  <Send size={15} />
                </button>
              </form>
            </div>
          </section>

          {/* Right Panel: Sources */}
          <section className="ui-panel sources-panel">
            <div className="sources-header">
              <h3 className="sources-title">
                <FileText size={16} color="#0f172a" />
                <span>Sources</span>
              </h3>
              <p className="sources-subtitle">View the document excerpts used to generate the answer.</p>
            </div>

            <div className="sources-content-scrollable">
              {sources.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '2.5rem 1rem', color: '#94a3b8', fontSize: '0.8rem', lineHeight: 1.5 }}>
                  No retrieved sources yet.<br/>Relevant excerpts from your documents will appear here when you ask a question.
                </div>
              ) : (
                sources.map((src) => (
                  <div key={src.id} className="source-card">
                    <div className="source-card-header">
                      <div className="source-card-doc-info">
                        <div className="source-card-icon">
                          <FileText size={14} />
                        </div>
                        <div className="source-card-names">
                          <span className="source-card-title">{src.doc}</span>
                          <span className="source-card-page">{src.page}</span>
                        </div>
                      </div>
                      <button className="doc-item-menu-btn" title="Options">
                        <MoreVertical size={14} />
                      </button>
                    </div>
                    <p className="source-card-body-quote">{src.excerpt}</p>
                  </div>
                ))
              )}
            </div>
          </section>

        </div>
      </div>

      {/* Document View Modal */}
      {viewingDoc && (
        <div className="modal-backdrop" onClick={() => setViewingDoc(null)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-box">
                <div className="modal-doc-icon-box">
                  <FileText size={18} />
                </div>
                <div>
                  <h3 className="modal-filename">{viewingDoc.name}</h3>
                  <span className="modal-sub">{viewingDoc.size} · Uploaded {viewingDoc.date}</span>
                </div>
              </div>
              <button className="modal-close-btn" onClick={() => setViewingDoc(null)} title="Close">
                <X size={16} />
              </button>
            </div>

            <div className="modal-body">
              <div className="modal-meta-grid">
                <div className="modal-meta-cell">
                  <span className="meta-label">Format</span>
                  <span className="meta-val">{viewingDoc.type || viewingDoc.name.split('.').pop()?.toUpperCase()}</span>
                </div>
                <div className="modal-meta-cell">
                  <span className="meta-label">File Size</span>
                  <span className="meta-val">{viewingDoc.size}</span>
                </div>
                <div className="modal-meta-cell">
                  <span className="meta-label">Chunks Created</span>
                  <span className="meta-val">{viewingDoc.chunks || 'Indexed'}</span>
                </div>
                <div className="modal-meta-cell">
                  <span className="meta-label">Index Status</span>
                  <span className="meta-val text-green">Indexed in Pinecone</span>
                </div>
              </div>

              {viewingDoc.sampleExcerpt && (
                <div className="modal-preview-box">
                  <span className="meta-label">Content Preview / Extracted Data:</span>
                  <p className="modal-preview-text">{viewingDoc.sampleExcerpt}</p>
                </div>
              )}
            </div>

            <div className="modal-footer">
              <button className="modal-close-action-btn" onClick={() => setViewingDoc(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
