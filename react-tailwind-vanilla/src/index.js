import React from "react";
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

// Simulated 24-hour data
const flareData = Array.from({ length: 24 }, (_, i) => {
  const hour = (new Date().getHours() + i) % 24;
  const label = `${hour % 12 === 0 ? 12 : hour % 12}${hour < 12 ? "AM" : "PM"}`;
  const flux = (Math.random() * 3 + 1).toFixed(2);
  const level = flux > 3.5 ? "C-Class" : flux > 2 ? "B-Class" : "A-Class";
  return { time: label, flux: parseFloat(flux), level };
});

const App = () => {
  return (
    <div className="bg-gradient-to-b from-slate-800 to-slate-900 text-white min-h-screen p-6 font-sans">
      {/* Header */}
      <div className="text-center mb-6">
        <h1 className="text-2xl font-bold">Solar Flare Tracker</h1>
        <p className="text-sm text-slate-300">Next 24 Hours of Flare Activity</p>
      </div>

      {/* Current Panel */}
      <div className="flex items-center justify-center gap-6 mb-8">
        <img src="https://img.icons8.com/emoji/96/sun-emoji.png" alt="flare" className="w-16 h-16" />
        <div>
          <h2 className="text-3xl font-bold">{flareData[0].level}</h2>
          <p className="text-sm text-slate-400">Flux: {flareData[0].flux} × 10⁻⁶ W/m²</p>
          <p className="text-xs text-slate-500">Now</p>
        </div>
      </div>

      {/* 24-Hour Scroll */}
      <div className="overflow-x-auto whitespace-nowrap mb-6">
        <div className="flex gap-4 bg-slate-700 rounded-xl p-4 w-max">
          {flareData.map((h, i) => (
            <div key={i} className="text-center min-w-[60px]">
              <p className="text-sm font-semibold">{h.time}</p>
              <p className="text-xs">{h.level}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Flux Chart */}
      <div className="bg-slate-700 p-4 rounded-xl text-sm text-slate-300 mb-6">
        <h3 className="text-white text-sm mb-2">Predicted Flux Levels</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={flareData}>
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="flux" stroke="#38bdf8" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Footer */}
      <div className="text-center text-xs text-slate-500">
        Updated {new Date().toLocaleTimeString()} · Static Simulated Data
      </div>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
