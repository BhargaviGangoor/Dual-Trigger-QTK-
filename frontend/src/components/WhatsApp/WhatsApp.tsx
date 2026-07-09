import React, { useState } from "react";
import { Search, MoreVertical, Paperclip, Smile, Send, Check, CheckCheck, Laptop, Phone, Plus, X, Shield, QrCode } from "lucide-react";
import type { SimulatedDevice, ChatMessage } from "../../App";

interface WhatsAppViewProps {
  devices: SimulatedDevice[];
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  onLinkDeviceQR: () => void;
}

export default function WhatsAppView({ devices, messages, onSendMessage, onLinkDeviceQR }: WhatsAppViewProps) {
  const [inputText, setInputText] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [showQRModal, setShowQRModal] = useState(false);
  const [qrStep, setQrStep] = useState(0); // 0: show QR, 1: linking, 2: success

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    onSendMessage(inputText);
    setInputText("");
  };

  const simulateQRScan = () => {
    setQrStep(1);
    setTimeout(() => {
      setQrStep(2);
      setTimeout(() => {
        setShowQRModal(false);
        setQrStep(0);
        onLinkDeviceQR(); // resets/notifies to simulate pairing
      }, 1500);
    }, 1500);
  };

  return (
    <div className="flex h-full w-full bg-[#111b21] overflow-hidden">
      {/* 1. Sidebar Left */}
      <div className="w-[30%] h-full border-r border-[#222e35] flex flex-col bg-[#111b21]">
        {/* Sidebar Header */}
        <div className="h-[60px] bg-[#202c33] px-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img
              src="https://api.dicebear.com/7.x/bottts/svg?seed=user_primary"
              className="h-10 w-10 rounded-full bg-[#374248] border border-[#4f5d64]"
              alt="Avatar"
            />
            <span className="font-semibold text-sm">Primary Session</span>
          </div>
          
          <div className="flex items-center gap-4 text-[#aebac1]">
            <button 
              onClick={() => setShowSettings(!showSettings)} 
              className={`hover:bg-[#374248] p-2 rounded-full transition-colors ${showSettings ? "text-wa-green" : ""}`}
              title="Linked Devices Settings"
            >
              <Laptop size={20} />
            </button>
            <button className="hover:bg-[#374248] p-2 rounded-full transition-colors">
              <MoreVertical size={20} />
            </button>
          </div>
        </div>

        {/* Search Bar */}
        <div className="px-3 py-2 bg-[#111b21]">
          <div className="bg-[#202c33] rounded-lg flex items-center px-3 py-1.5 gap-3">
            <Search size={16} className="text-[#8696a0]" />
            <input
              type="text"
              placeholder="Search or start new chat"
              className="bg-transparent text-sm text-[#e9edef] border-none outline-none w-full placeholder:text-[#8696a0]"
            />
          </div>
        </div>

        {/* Chats List */}
        <div className="flex-1 overflow-y-auto">
          {/* Active research chat */}
          <div className="flex items-center gap-3 px-3 py-3 bg-[#2a3942] border-b border-[#222e35] cursor-pointer">
            <img
              src="https://api.dicebear.com/7.x/adventurer/svg?seed=Alice"
              className="h-12 w-12 rounded-full bg-[#374248]"
              alt="Alice"
            />
            <div className="flex-1 flex flex-col justify-center min-w-0">
              <div className="flex justify-between items-baseline">
                <span className="font-medium text-[#e9edef] truncate text-base">Alice (Researcher)</span>
                <span className="text-xs text-wa-green font-medium">Online</span>
              </div>
              <p className="text-xs text-[#8696a0] truncate mt-0.5">
                {messages.length > 0 ? messages[messages.length - 1].text : "No messages"}
              </p>
            </div>
          </div>

          {/* Group chats mock */}
          <div className="flex items-center gap-3 px-3 py-3 hover:bg-[#202c33]/50 border-b border-[#222e35] cursor-pointer">
            <div className="h-12 w-12 rounded-full bg-wa-green/10 flex items-center justify-center text-wa-green font-bold">
              G
            </div>
            <div className="flex-1 flex flex-col justify-center min-w-0">
              <div className="flex justify-between items-baseline">
                <span className="font-medium text-[#e9edef] truncate text-base">E2EE Protocol Review Group</span>
                <span className="text-xs text-[#8696a0]">Yesterday</span>
              </div>
              <p className="text-xs text-[#8696a0] truncate mt-0.5">Bob: Did we merge the session sync fixes?</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 px-3 py-3 hover:bg-[#202c33]/50 border-b border-[#222e35] cursor-pointer">
            <div className="h-12 w-12 rounded-full bg-[#374248] flex items-center justify-center text-[#8696a0] font-bold">
              S
            </div>
            <div className="flex-1 flex flex-col justify-center min-w-0">
              <div className="flex justify-between items-baseline">
                <span className="font-medium text-[#e9edef] truncate text-base">Security Bulletins</span>
                <span className="text-xs text-[#8696a0]">07/08/2026</span>
              </div>
              <p className="text-xs text-[#8696a0] truncate mt-0.5">[Alert] Anomaly scores increased for Web Pairings.</p>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Chat Area or Settings Middle */}
      <div className="flex-1 h-full flex flex-col bg-[#0b141a]">
        {showSettings ? (
          /* Linked Devices Panel */
          <div className="flex-1 flex flex-col bg-[#111b21] p-6 overflow-y-auto">
            <div className="flex justify-between items-center pb-4 border-b border-[#222e35]">
              <div>
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Laptop className="text-wa-green" /> Linked Devices Settings
                </h2>
                <p className="text-xs text-[#8696a0] mt-1">Manage secondary E2EE synchronization sessions linked to this account.</p>
              </div>
              <button 
                onClick={() => setShowQRModal(true)} 
                className="bg-wa-green text-white hover:bg-wa-green/90 px-4 py-2 rounded-lg flex items-center gap-2 text-sm font-semibold transition-colors"
              >
                <Plus size={16} /> Link a Device
              </button>
            </div>

            <div className="mt-6 flex flex-col gap-4">
              {devices.map((device) => {
                const isTrusted = device.state === "Trusted";
                const isRevoked = device.state === "Revoked";
                return (
                  <div key={device.id} className={`p-4 rounded-xl border flex justify-between items-start ${
                    isRevoked ? "border-red-500/20 bg-red-950/10" :
                    isTrusted ? "border-[#222e35] bg-[#202c33]/40" : "border-yellow-500/20 bg-yellow-950/10"
                  }`}>
                    <div className="flex gap-4">
                      <div className={`p-3 rounded-lg ${isRevoked ? "bg-red-500/10 text-red-500" : isTrusted ? "bg-wa-green/10 text-wa-green" : "bg-yellow-500/10 text-yellow-500"}`}>
                        {device.type === "primary" ? <Phone size={24} /> : <Laptop size={24} />}
                      </div>
                      <div>
                        <h3 className="font-semibold text-[#e9edef] flex items-center gap-2">
                          {device.name}
                          <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-bold ${
                            isRevoked ? "bg-red-500/20 text-red-400" :
                            isTrusted ? "bg-wa-green/20 text-wa-green" : "bg-yellow-500/20 text-yellow-400"
                          }`}>
                            {device.state}
                          </span>
                        </h3>
                        <p className="text-xs text-[#8696a0] mt-1">Fingerprint: <code className="font-mono text-gray-400 text-[10px]">{device.id}</code></p>
                        <p className="text-xs text-[#8696a0]">Network IP: {device.ip} ({device.network})</p>
                        <p className="text-xs text-[#8696a0]">Location: {device.country} / {device.timezone}</p>
                      </div>
                    </div>

                    <div className="text-right">
                      <span className="text-xs text-[#8696a0]">Trust score:</span>
                      <div className={`text-xl font-mono font-extrabold mt-0.5 ${
                        device.trust_score > 0.8 ? "text-wa-green" :
                        device.trust_score > 0.5 ? "text-yellow-500" : "text-red-500"
                      }`}>
                        {device.trust_score.toFixed(2)}
                      </div>
                      {!isRevoked && device.type !== "primary" && (
                        <button className="text-xs text-red-400 hover:underline mt-4 block">Revoke Device</button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          /* Alice Conversation Chat Pane */
          <>
            {/* Chat Header */}
            <div className="h-[60px] bg-[#202c33] px-4 flex items-center justify-between border-l border-[#374248]">
              <div className="flex items-center gap-3">
                <img
                  src="https://api.dicebear.com/7.x/adventurer/svg?seed=Alice"
                  className="h-10 w-10 rounded-full bg-[#374248]"
                  alt="Alice"
                />
                <div>
                  <h3 className="font-medium text-[#e9edef] text-sm">Alice (Researcher)</h3>
                  <p className="text-xs text-wa-green">Online | typing...</p>
                </div>
              </div>
              <div className="flex items-center gap-4 text-[#aebac1]">
                <button className="hover:bg-[#374248] p-2 rounded-full"><Search size={20} /></button>
                <button className="hover:bg-[#374248] p-2 rounded-full"><MoreVertical size={20} /></button>
              </div>
            </div>

            {/* Chat Messages Log */}
            <div className="flex-1 overflow-y-auto bg-[#0b141a] px-8 py-4 flex flex-col gap-3" style={{ backgroundImage: "url('https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png')", backgroundBlendMode: "overlay" }}>
              <div className="self-center bg-[#182229] text-[#8696a0] text-xs px-3 py-1 rounded-lg flex items-center gap-1.5 font-medium border border-[#222e35]">
                <Shield size={12} className="text-wa-green" /> Messages are end-to-end encrypted. No one outside of this chat can read them.
              </div>

              {messages.map((m) => (
                <div key={m.id} className={`max-w-[60%] p-2.5 rounded-lg text-sm relative flex flex-col ${
                  m.isIncoming ? "bg-[#202c33] self-start rounded-tl-none text-[#e9edef]" : "bg-[#005c4b] self-end rounded-tr-none text-[#e9edef]"
                }`}>
                  <p className="leading-5">{m.text}</p>
                  <div className="self-end flex items-center gap-1 mt-1">
                    <span className="text-[9px] text-[#8696a0]">{m.timestamp}</span>
                    {!m.isIncoming && (
                      m.status === "read" ? <CheckCheck size={14} className="text-[#53bdeb]" /> :
                      m.status === "delivered" ? <CheckCheck size={14} className="text-[#8696a0]" /> : <Check size={14} className="text-[#8696a0]" />
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Chat Input Bar */}
            <form onSubmit={handleSend} className="h-[62px] bg-[#202c33] px-4 flex items-center gap-3">
              <div className="flex items-center gap-3 text-[#aebac1]">
                <button type="button" className="hover:text-white transition-colors"><Smile size={22} /></button>
                <button type="button" className="hover:text-white transition-colors"><Paperclip size={22} /></button>
              </div>
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Type a message"
                className="flex-1 bg-[#2a3942] text-[#e9edef] placeholder:text-[#8696a0] rounded-lg px-4 py-2 border-none outline-none text-sm"
              />
              <button type="submit" className="bg-wa-green hover:bg-wa-green/80 text-white rounded-full p-2.5 flex items-center justify-center transition-colors">
                <Send size={18} />
              </button>
            </form>
          </>
        )}
      </div>

      {/* 3. Link Device QR Simulation Modal */}
      {showQRModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-[#222e35] rounded-2xl max-w-md w-full border border-[#374248] p-6 relative flex flex-col items-center">
            <button onClick={() => setShowQRModal(false)} className="absolute right-4 top-4 text-[#8696a0] hover:text-white">
              <X size={20} />
            </button>
            
            <h3 className="text-lg font-bold text-white mb-2">Link with QR Code</h3>
            <p className="text-xs text-[#8696a0] text-center mb-6">Scan this QR code from your primary device's WhatsApp Settings to authorize a new linked session.</p>

            {qrStep === 0 && (
              <div className="bg-white p-6 rounded-xl border-4 border-wa-green flex flex-col items-center justify-center cursor-pointer hover:opacity-90 transition-opacity" onClick={simulateQRScan}>
                <QrCode size={200} className="text-black" />
                <span className="text-[10px] text-black font-semibold mt-4 bg-gray-100 px-3 py-1 rounded-full uppercase tracking-wider">Click to Scan (Simulated)</span>
              </div>
            )}

            {qrStep === 1 && (
              <div className="h-[256px] w-[256px] flex flex-col items-center justify-center gap-4">
                <div className="h-12 w-12 rounded-full border-4 border-wa-green border-t-transparent animate-spin"></div>
                <span className="text-sm font-semibold text-[#e9edef]">Establishing secure pairing key epochs...</span>
              </div>
            )}

            {qrStep === 2 && (
              <div className="h-[256px] w-[256px] flex flex-col items-center justify-center gap-4 text-center">
                <div className="h-16 w-16 bg-wa-green/10 text-wa-green rounded-full flex items-center justify-center">
                  <CheckCheck size={36} />
                </div>
                <div>
                  <h4 className="font-bold text-white text-base">Device Synced!</h4>
                  <p className="text-xs text-[#8696a0] mt-1">Starting initial message and offline receipt syncs.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
