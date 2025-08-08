import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

/* ---------- helpers ---------- */

const todayUTC = () => new Date().toISOString().slice(0, 10);
const shiftDay = (iso, d) => {
  const t = new Date(iso + "T00:00:00Z");
  t.setUTCDate(t.getUTCDate() + d);
  return t.toISOString().slice(0, 10);
};
const longDate = (iso) =>
  new Date(iso + "T00:00:00Z").toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
const makeDummy = () => {
  const cls = ["A", "B", "C", "M", "X"];
  return Array.from({ length: 24 }, (_, i) => ({
    time: `${(i % 12 || 12)}${i < 12 ? "AM" : "PM"}`,
    level: `${cls[Math.floor(Math.random() * 5)]}-Class`,
    classIndex: Math.floor(Math.random() * 5),
  }));
};

/* ---------- main ---------- */

const App = () => {
  const [day, setDay]        = useState(todayUTC());
  const [flareData, setData] = useState(makeDummy());
  const [dark, setDark]      = useState(true);          // start in dark-mode
  const [now, setNow]        = useState(new Date());    // live clock

  /* refresh dummy data when the chosen day changes */
  useEffect(() => setData(makeDummy()), [day]);

  /* tick the clock every 30 s */
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 30_000);
    return () => clearInterval(id);
  }, []);

  const currentHour   = now.getHours();                       // 0 … 23
  const currentTime   = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  /* theme palette */
  const bgGrad   = dark ? "from-slate-800 to-slate-900" : "from-sky-100 to-sky-300";
  const panel    = dark ? "bg-slate-700" : "bg-white/40 backdrop-blur";
  const textMain = dark ? "text-white"   : "text-slate-800";
  const textSub  = dark ? "text-slate-300" : "text-slate-600";
  const arrowClr = dark ? "text-blue-300"  : "text-blue-700";
  const hiBg     = dark ? "bg-cyan-500 text-white" : "bg-blue-600 text-white";

  /* ---------- JSX ---------- */

  return (
    <div className={`bg-gradient-to-b ${bgGrad} ${textMain} min-h-screen p-4 sm:p-6 font-sans transition-colors duration-300`}>
      {/* title bar */}
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-3xl sm:text-4xl font-extrabold">Solar&nbsp;Sense</h1>
        <button onClick={() => setDark((x) => !x)}>
          <img
            src={
              dark
                ? "https://img.icons8.com/emoji/96/sun-emoji.png"
                : "https://img.icons8.com/emoji/96/crescent-moon-emoji.png"
            }
            alt="mode"
            className="w-10 h-10"
          />
        </button>
      </header>

      {/* date + headline */}
      <section className="flex items-center justify-center gap-4 mb-6">
        <button onClick={() => setDay((d) => shiftDay(d, -1))} className={`px-3 text-2xl font-bold select-none ${arrowClr}`}>&lt;</button>

        <div className="text-center">
          <p className={`text-lg font-semibold ${textSub}`}>{longDate(day)}</p>
          <p className={`text-sm mb-1 ${textSub}`}>{currentTime}</p>
          <p className="text-3xl font-bold">{flareData[0].level}</p>
        </div>

        <button onClick={() => setDay((d) => shiftDay(d, 1))} className={`px-3 text-2xl font-bold select-none ${arrowClr}`}>&gt;</button>
      </section>

      {/* hourly strip */}
      <section className="mb-6">
        {/* desktop */}
        <div className={`hidden sm:flex overflow-x-auto gap-3 ${panel} rounded-xl p-3`}>
          {flareData.map((h, i) => {
            const hi = i === currentHour ? hiBg : "";
            return (
              <div key={i} className="text-center min-w-[64px]">
                <p className={`text-sm font-semibold rounded ${hi}`}>{h.time}</p>
                <p className={`text-xs rounded ${hi}`}>{h.level}</p>
              </div>
            );
          })}
        </div>

        {/* mobile */}
        <div className={`sm:hidden overflow-y-auto max-h-64 flex flex-col gap-2 ${panel} rounded-xl p-3`}>
          {flareData.map((h, i) => {
            const hi = i === currentHour ? hiBg : "bg-white/20 sm:bg-transparent";
            return (
              <div key={i} className={`flex justify-between rounded-lg px-3 py-1 ${hi}`}>
                <span className="text-sm font-medium">{h.time}</span>
                <span className="text-xs">{h.level}</span>
              </div>
            );
          })}
        </div>
      </section>

      {/* chart */}
      <section className={`${panel} p-4 rounded-xl mb-6`}>
        <h3 className={`text-sm mb-2 ${textMain}`}>Predicted Flare Classes</h3>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={flareData}>
            <XAxis dataKey="time" tick={{ fill: dark ? "#94a3b8" : "#334155", fontSize: 11 }} />
            <YAxis
              type="number"
              domain={[0, 4]}
              ticks={[0, 1, 2, 3, 4]}
              tickFormatter={(v) => ["A", "B", "C", "M", "X"][v]}
              width={30}
              tick={{ fill: dark ? "#94a3b8" : "#334155", fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{ background: dark ? "#1e293b" : "#f1f5f9", border: "none" }}
              labelStyle={{ color: dark ? "#f8fafc" : "#0f172a" }}
              formatter={(v) => [`${["A", "B", "C", "M", "X"][v]}-Class`, "Class"]}
            />
            <Line type="monotone" dataKey="classIndex" stroke="#38bdf8" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <footer className={`text-center text-xs ${textSub}`}>
        Updated {now.toLocaleTimeString()} · Dynamic Simulated Data
      </footer>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
