import { useState, useEffect, useRef } from "react";
import { MessageSquare, ShieldAlert, Wifi, RotateCcw } from "lucide-react";
import WhatsAppView from "./components/WhatsApp/WhatsApp";
import DashboardView from "./components/Dashboard/Dashboard";

export interface SimulatedDevice {
  id: string;
  name: string;
  type: string;
  trust_score: number;
  state: string;
  ip: string;
  network: string;
  country: string;
  timezone: string;
  battery: number;
  pairing_time: string;
  os_version: string;
}

export interface SimulationEvent {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  score_after: number | null;
  state_after: string | null;
}

export interface ChatMessage {
  id: string;
  sender: string;
  text: string;
  timestamp: string;
  status: "sent" | "delivered" | "read";
  isIncoming: boolean;
}

export interface ExperimentMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  fp: number;
  fn: number;
}

function App() {
  const [activeTab, setActiveTab] = useState<"whatsapp" | "dashboard">("whatsapp");
  
  // Simulation configurations
  const [profile, setProfile] = useState<string>("Student");
  const [attackType, setAttackType] = useState<string>("Ghost Pairing");
  const [attackDay, setAttackDay] = useState<number>(10);
  const [alpha, setAlpha] = useState<number>(0.8);
  const [threshold, setThreshold] = useState<number>(0.65);
  const [noise, setNoise] = useState<number>(0.05);
  const [simSpeed, setSimSpeed] = useState<number>(1); // seconds per simulated day

  // Simulation run states
  const [simRunning, setSimRunning] = useState<boolean>(false);
  const [currentDay, setCurrentDay] = useState<number>(0);
  const [devices, setDevices] = useState<SimulatedDevice[]>([]);
  const [events, setEvents] = useState<SimulationEvent[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  
  // Charts tracking history
  const [trustHistory, setTrustHistory] = useState<{ day: number; [key: string]: number }[]>([]);
  
  // WebSocket setup
  const [backendConnected, setBackendConnected] = useState<boolean>(false);
  const ws = useRef<WebSocket | null>(null);

  // Initialize DB reset and sample devices on start
  useEffect(() => {
    resetSimulation();
    connectWebSocket();
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  const connectWebSocket = () => {
    try {
      ws.current = new WebSocket("ws://localhost:8000/ws/simulation");
      
      ws.current.onopen = () => {
        setBackendConnected(true);
      };
      
      ws.current.onclose = () => {
        setBackendConnected(false);
      };
      
      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.status === "update") {
          setCurrentDay(data.day);
          setDevices(data.devices);
          setEvents(data.events);
          
          // Append trust scores to history
          const stepUpdate: any = { day: data.day };
          data.devices.forEach((d: SimulatedDevice) => {
            stepUpdate[d.name] = d.trust_score;
          });
          setTrustHistory(prev => [...prev, stepUpdate]);
        }
      };
    } catch (e) {
      setBackendConnected(false);
    }
  };

  // Re-run fallback client simulation loop when active and FastAPI is not available
  useEffect(() => {
    if (!simRunning || backendConnected) return;

    const interval = setInterval(() => {
      setCurrentDay(prevDay => {
        const nextDay = prevDay + 1;
        if (nextDay >= 30) {
          setSimRunning(false);
          clearInterval(interval);
          return 30;
        }
        runClientSimTick(nextDay);
        return nextDay;
      });
    }, 1200 / simSpeed);

    return () => clearInterval(interval);
  }, [simRunning, simSpeed, backendConnected, profile, attackType, attackDay, alpha, threshold]);

  const resetSimulation = () => {
    setSimRunning(false);
    setCurrentDay(0);
    setTrustHistory([]);
    
    // Default initial devices (Primary Phone + Chrome Web Session)
    const initialDevices: SimulatedDevice[] = [
      {
        id: "pri-01",
        name: "Primary Phone (Android)",
        type: "primary",
        trust_score: 1.0,
        state: "Trusted",
        ip: "192.168.1.45",
        network: "WiFi",
        country: "United States",
        timezone: "America/New_York",
        battery: 98,
        pairing_time: new Date().toISOString(),
        os_version: "Android 14"
      },
      {
        id: "web-02",
        name: "Chrome Browser (Windows)",
        type: "linked",
        trust_score: 1.0,
        state: "Trusted",
        ip: "192.168.1.45",
        network: "WiFi",
        country: "United States",
        timezone: "America/New_York",
        battery: 100,
        pairing_time: new Date().toISOString(),
        os_version: "Windows 11"
      }
    ];
    setDevices(initialDevices);
    
    setEvents([
      {
        id: "ev-001",
        type: "pair_device",
        description: "Primary Phone registered as trust root.",
        timestamp: new Date().toLocaleTimeString(),
        score_after: 1.0,
        state_after: "Trusted"
      },
      {
        id: "ev-002",
        type: "pair_device",
        description: "Linked device Chrome Browser successfully paired.",
        timestamp: new Date().toLocaleTimeString(),
        score_after: 1.0,
        state_after: "Trusted"
      }
    ]);

    setMessages([
      { id: "m1", sender: "Alice", text: "Hey! Are you working on the trust research paper?", timestamp: "09:42 AM", status: "read", isIncoming: true },
      { id: "m2", sender: "Me", text: "Yes, building the ghost pairing detection simulation platform.", timestamp: "09:43 AM", status: "read", isIncoming: false },
      { id: "m3", sender: "Alice", text: "Perfect! Does it support adaptive decay modeling?", timestamp: "09:45 AM", status: "read", isIncoming: true }
    ]);
  };

  const handleMessageSent = (text: string) => {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const newMsg: ChatMessage = {
      id: Math.random().toString(),
      sender: "Me",
      text,
      timestamp: time,
      status: "read",
      isIncoming: false
    };
    setMessages(prev => [...prev, newMsg]);

    // Simple reply logic
    setTimeout(() => {
      const responses = [
        "Sounds interesting. Let's run a trust validation test.",
        "Simulating sync verification protocols now.",
        "Did we verify the HMM state output?",
        "E2EE sync complete."
      ];
      const replyMsg: ChatMessage = {
        id: Math.random().toString(),
        sender: "Alice",
        text: responses[Math.floor(randomVal() * responses.length)],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        status: "read",
        isIncoming: true
      };
      setMessages(prev => [...prev, replyMsg]);
    }, 1500);
  };

  const randomVal = () => {
    return Math.random();
  };

  // Run a client-side backup simulation tick when local python is not available
  const runClientSimTick = (day: number) => {
    setDevices(prevDevices => {
      let activeList = [...prevDevices];
      let newEvents: SimulationEvent[] = [];

      // 1. Check if attack triggers
      if (day === attackDay && attackType !== "None") {
        if (attackType === "Ghost Pairing") {
          const ghost: SimulatedDevice = {
            id: `ghost-${Math.random().toString(36).substr(2, 4)}`,
            name: "Chrome Browser (Linux)",
            type: "linked",
            trust_score: 0.50,
            state: "Suspicious",
            ip: "185.220.101.4",  // Tor exit node
            network: "VPN",
            country: "Romania",
            timezone: "Europe/Bucharest",
            battery: 100,
            pairing_time: new Date().toISOString(),
            os_version: "Linux x86_64"
          };
          activeList.push(ghost);
          newEvents.push({
            id: Math.random().toString(),
            type: "attack_trigger",
            description: `Rogue device paired silently (Ghost Pairing). Injected at ${ghost.country} (IP: ${ghost.ip}).`,
            timestamp: new Date().toLocaleTimeString(),
            score_after: 0.50,
            state_after: "Suspicious"
          });
        } else {
          // Anomaly on existing linked device
          activeList = activeList.map(d => {
            if (d.type === "linked") {
              return {
                ...d,
                ip: "45.89.230.12",
                network: "VPN",
                country: "Netherlands",
                timezone: "Europe/Amsterdam",
                trust_score: 0.70,
                state: "Suspicious"
              };
            }
            return d;
          });
          newEvents.push({
            id: Math.random().toString(),
            type: "attack_trigger",
            description: `Session hijacking anomaly detected on Chrome Web session. Remote IP relocated.`,
            timestamp: new Date().toLocaleTimeString(),
            score_after: 0.70,
            state_after: "Suspicious"
          });
        }
      }

      // 2. Apply behavior models and trust decay
      activeList = activeList.map(d => {
        if (d.state === "Revoked") return d;

        // Calculate behavioral evidence
        let evidence = 1.0;
        
        // Under attack condition
        const isAttackerDevice = d.id.startsWith("ghost") || (attackType !== "None" && d.type === "linked" && day >= attackDay);
        
        if (isAttackerDevice) {
          // Attacker exhibits anomalous signatures (VPN, location changes, weird hours)
          evidence = 0.15 + (randomVal() * 0.15); // severe anomaly
        } else {
          // Normal user with occasional minor noise
          evidence = 0.95 - (randomVal() * noise);
        }

        // Apply trust decay formula: score = alpha * d.trust_score + (1-alpha) * evidence
        const newScore = alpha * d.trust_score + (1.0 - alpha) * evidence;
        
        // Determine state transition
        let nextState = d.state;
        let transitionReason = "";
        let lastKeyEpoch = (d as any).last_key_update_epoch || 0;
        let qatEpoch = (d as any).quarantined_at_epoch || null;
        let shares = (d as any).qtk_shares || null;
        
        const isDeviceActive = !isAttackerDevice;
        if (isDeviceActive && d.state !== "Quarantined" && d.state !== "Revoked") {
          lastKeyEpoch = day;
        }
        
        const epochGap = day - lastKeyEpoch;
        const R_dt = isAttackerDevice ? 0.75 : 0.15;
        const shouldQuarantine = (epochGap >= 5) || (R_dt >= threshold);
        
        if (shouldQuarantine && d.state !== "Quarantined" && d.state !== "Revoked") {
          nextState = "Quarantined";
          transitionReason = `Quarantined-TreeKEM (QTK) quarantine triggered (Epoch Gap: ${epochGap}, Risk: ${R_dt.toFixed(2)}).`;
          qatEpoch = day;
          shares = { "pri-01": [1, 54321], "web-02": [2, 98765] };
        } else if (d.state === "Quarantined") {
          if (isDeviceActive && R_dt < threshold) {
            nextState = "Trusted";
            transitionReason = "QTK key reconstructed from active members' secret shares. Quarantine lifted.";
            qatEpoch = null;
            shares = null;
            lastKeyEpoch = day;
          } else {
            const qDuration = day - (qatEpoch || day);
            if (qDuration >= 5) {
              nextState = "Revoked";
              transitionReason = "Quarantine timeout expired. Device expelled from continuous group key agreement.";
            }
          }
        } else {
          // Standard FSM
          if (newScore < 0.20) {
            nextState = "Revoked";
            transitionReason = "Trust score fell below critical threshold.";
          } else if (newScore < 0.50) {
            nextState = "Verification Required";
            transitionReason = "Severe trust drop detected.";
          } else if (newScore < threshold) {
            nextState = "Suspicious";
            transitionReason = "Moderate behavioral deviations.";
          } else if (newScore >= 0.80 && d.state === "Suspicious") {
            nextState = "Trusted";
            transitionReason = "Consistent standard behavior restored.";
          }
        }

        if (nextState !== d.state) {
          newEvents.push({
            id: Math.random().toString(),
            type: newScore < d.trust_score ? "trust_decay" : "trust_recovery",
            description: `State changed for ${d.name} from ${d.state} to ${nextState} (Score: ${newScore.toFixed(2)}). Reason: ${transitionReason}`,
            timestamp: new Date().toLocaleTimeString(),
            score_after: newScore,
            state_after: nextState
          });
        }

        return {
          ...d,
          trust_score: parseFloat(newScore.toFixed(4)),
          state: nextState,
          last_key_update_epoch: lastKeyEpoch,
          quarantined_at_epoch: qatEpoch,
          qtk_shares: shares
        } as SimulatedDevice;
      });

      // Update events timeline
      if (newEvents.length > 0) {
        setEvents(prev => [...newEvents, ...prev]);
      }

      // Record history
      const stepUpdate: any = { day };
      activeList.forEach(d => {
        stepUpdate[d.name] = d.trust_score;
      });
      setTrustHistory(prev => [...prev, stepUpdate]);

      return activeList;
    });
  };

  return (
    <div className="flex h-screen w-screen bg-[#0b141a] text-[#e9edef] overflow-hidden select-none">
      {/* Platform Navigation Sidebar */}
      <div className="w-16 flex flex-col items-center justify-between border-r border-[#222e35] bg-[#202c33] py-4">
        <div className="flex flex-col gap-6">
          <div className="flex items-center justify-center h-12 w-12 rounded-xl bg-wa-green/20 text-wa-green cursor-pointer hover:scale-105 transition-transform" title="E2EE Trust Simulator">
            <span className="font-extrabold text-xl font-mono">Ag</span>
          </div>
          
          <button
            onClick={() => setActiveTab("whatsapp")}
            className={`flex items-center justify-center h-10 w-10 rounded-lg transition-colors ${activeTab === "whatsapp" ? "bg-[#374248] text-wa-green" : "text-[#8696a0] hover:bg-[#374248] hover:text-[#e9edef]"}`}
            title="WhatsApp Web Replica"
          >
            <MessageSquare size={22} />
          </button>
          
          <button
            onClick={() => setActiveTab("dashboard")}
            className={`flex items-center justify-center h-10 w-10 rounded-lg transition-colors ${activeTab === "dashboard" ? "bg-[#374248] text-[#53bdeb]" : "text-[#8696a0] hover:bg-[#374248] hover:text-[#e9edef]"}`}
            title="Research Dashboard"
          >
            <ShieldAlert size={22} />
          </button>
        </div>

        <div className="flex flex-col gap-4 items-center">
          {/* Connection status indicator */}
          <div className="flex flex-col items-center gap-1">
            <Wifi size={16} className={backendConnected ? "text-wa-green" : "text-amber-500"} />
            <span className="text-[9px] text-[#8696a0]">{backendConnected ? "Server" : "Local"}</span>
          </div>
          <button onClick={resetSimulation} className="text-[#8696a0] hover:text-white transition-colors" title="Reset Session">
            <RotateCcw size={18} />
          </button>
        </div>
      </div>

      {/* Main View Area */}
      <div className="flex-1 h-full overflow-hidden">
        {activeTab === "whatsapp" ? (
          <WhatsAppView
            devices={devices}
            messages={messages}
            onSendMessage={handleMessageSent}
            onLinkDeviceQR={resetSimulation}
          />
        ) : (
          <DashboardView
            profile={profile}
            setProfile={setProfile}
            attackType={attackType}
            setAttackType={setAttackType}
            attackDay={attackDay}
            setAttackDay={setAttackDay}
            alpha={alpha}
            setAlpha={setAlpha}
            threshold={threshold}
            setThreshold={setThreshold}
            noise={noise}
            setNoise={setNoise}
            simSpeed={simSpeed}
            setSimSpeed={setSimSpeed}
            simRunning={simRunning}
            setSimRunning={setSimRunning}
            currentDay={currentDay}
            devices={devices}
            events={events}
            trustHistory={trustHistory}
            resetSim={resetSimulation}
          />
        )}
      </div>
    </div>
  );
}

export default App;
