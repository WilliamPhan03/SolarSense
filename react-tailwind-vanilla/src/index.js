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
  Legend,
} from "recharts";

/**
 * Solar Sense - Solar Flare Prediction Dashboard
 * This is the only frontend file, which fetches data from the backend
 * and displays it in a responsive chart and hourly strip.
 * The chart shows the predicted solar flare classes for each hour of the day.
 * The hourly strip shows the time and class for each hour.
 * The app starts in dark mode, but can be toggled to light mode.
 * The app also shows the current time and date.
 * 
 * The app frontend currently does sklearn pipeline not pytorch pipeline.
 * A C
 */

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
    timeZone: "UTC",
  });

const makeDummy = () => {
  const cls = ["A", "B", "C", "M", "X"];
  return Array.from({ length: 24 }, (_, i) => ({
    time: `${i % 12 || 12}${i < 12 ? "AM" : "PM"}`,
    level: `${cls[Math.floor(Math.random() * 5)]}-Class`,
    classIndex: Math.floor(Math.random() * 5),
  }));
};

const sci = new Intl.NumberFormat("en", {
  notation: "scientific",
  maximumFractionDigits: 0,
});

/* --- API helper ----------------------------------------- */
// Expecting backend to return:
// { hourly_pred:[{hour,flux,class}…],
//   minute_pred:[{timestamp,long_flux_pred}…],
//   minute_actual:[{timestamp,long_flux}…] }

const localDayToUTCISO = (isoLocal /* 'YYYY-MM-DD' in local time */) => {
  // interpret local midnight, then shift to UTC and format back to YYYY-MM-DD
  const d = new Date(isoLocal + "T00:00:00");    // local midnight
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 10);
};

/* --- API helper ----------------------------------------- */
const getForecast = async (isoLocalDay) => {
  const dateUTC = localDayToUTCISO(isoLocalDay);
  const r = await fetch(`http://localhost:8000/forecast/${dateUTC}`);
  if (!r.ok) return null;
  return await r.json();
};

/* ---------- main ---------- */

const App = () => {
  const [day, setDay] = useState(todayUTC());
  const [flareData, setData] = useState(makeDummy()); // hourly classes (strip stays)
  const [chartData, setChartData] = useState([]); // minute-by-minute flux for chart
  const [dark, setDark] = useState(true);
  const [now, setNow] = useState(new Date());

  // Fetch whenever the selected day changes
  useEffect(() => {
    (async () => {
      const data = await getForecast(day);

      // Hourly strip (unchanged)
      if (data && data.hourly_pred?.length === 24) {
        const hours = data.hourly_pred.map((h) => ({
          time: `${h.hour % 12 || 12}${h.hour < 12 ? "AM" : "PM"}`,
          level: `${h.class}-Class`,
          classIndex: ["A", "B", "C", "M", "X"].indexOf(h.class),
        }));
        setData(hours);
      } else {
        setData(makeDummy());
      }

      // Minute-by-minute series for the chart (Predicted + Actual)
      // Merge by timestamp so both series line up.
      const pred = (data?.minute_pred || []).map((d) => [
        new Date(d.timestamp).getTime(),
        Number(d.long_flux_pred),
      ]);
      const act = (data?.minute_actual || []).map((d) => [
        new Date(d.timestamp).getTime(),
        Number(d.long_flux),
      ]);

      if (pred.length || act.length) {
        const map = new Map(); // ts -> { t: Date, pred?, actual? }
        for (const [ts, v] of pred) {
          map.set(ts, { t: new Date(ts), pred: v });
        }
        for (const [ts, v] of act) {
          const row = map.get(ts) || { t: new Date(ts) };
          row.actual = v;
          map.set(ts, row);
        }
        const merged = Array.from(map.entries())
          .sort((a, b) => a[0] - b[0])
          .map(([, v]) => v);

        setChartData(merged);
      } else {
        setChartData([]); // falls back to empty chart if backend didn't include minute series
      }
    })();
  }, [day]);

  // live clock (only for display, not logic)
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 30_000);
    return () => clearInterval(id);
  }, []);

 const currentHour = now.getUTCHours();
 const currentTime = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", timeZone: "UTC" });

  /* theme palette */
  const bgGrad = dark ? "from-slate-800 to-slate-900" : "from-sky-100 to-sky-300";
  const panel = dark ? "bg-slate-700" : "bg-white/40 backdrop-blur";
  const textMain = dark ? "text-white" : "text-slate-800";
  const textSub = dark ? "text-slate-300" : "text-slate-600";
  const arrowClr = dark ? "text-blue-300" : "text-blue-700";
  const hiBg = dark ? "bg-cyan-500 text-white" : "bg-blue-600 text-white";

  // Dynamic log-scale bounds (safe defaults if empty)
  const allVals = chartData.flatMap((d) =>
    [d.pred, d.actual].filter((x) => x && x > 0)
  );
  const dMin = allVals.length ? Math.min(...allVals) : 1e-7;
  const dMax = allVals.length ? Math.max(...allVals) : 1e-5;
  const yMin = Math.max(1e-8, dMin * 0.8);
  const yMax = Math.min(1e-3, dMax * 1.2);

  return (
    <div
      className={`bg-gradient-to-b ${bgGrad} ${textMain} min-h-screen p-4 sm:p-6 font-sans transition-colors duration-300`}
    >
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
        <button
          onClick={() => setDay((d) => shiftDay(d, -1))}
          className={`px-3 text-2xl font-bold select-none ${arrowClr}`}
        >
          &lt;
        </button>

        <div className="text-center">
          <p className={`text-lg font-semibold ${textSub}`}>{longDate(day)}</p>
          <p className={`text-sm mb-1 ${textSub}`}>{currentTime}</p>
          <p className="text-3xl font-bold">{flareData[0].level}</p>
        </div>

        <button
          onClick={() => setDay((d) => shiftDay(d, 1))}
          className={`px-3 text-2xl font-bold select-none ${arrowClr}`}
        >
          &gt;
        </button>
      </section>

      {/* hourly strip (unchanged) */}
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
        <div
          className={`sm:hidden overflow-y-auto max-h-64 flex flex-col gap-2 ${panel} rounded-xl p-3`}
        >
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

      {/* chart – minute-by-minute long_flux on log scale */}
      <section className={`${panel} p-4 rounded-xl mb-6`}>
        <h3 className={`text-sm mb-2 ${textMain}`}>Minute Flux (Pred vs Actual)</h3>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={chartData}>
            <XAxis
              dataKey="t"
              tickFormatter={(d) => {
                // d can be "2025-08-16 05:12:00" or an ISO string/date/number.
                const s = typeof d === "string" ? d : new Date(d).toISOString();
                const dt = new Date(s.endsWith("Z") ? s : s + "Z");
                return dt.toLocaleTimeString([], {
                  hour: "2-digit",
                  hour12: true,
                  timeZone: "UTC",
                });
              }}
              minTickGap={30}
              stroke={dark ? "#94a3b8" : "#334155"}
            />
            <YAxis
              scale="log"
              domain={[yMin, yMax]}
              tick={{ fill: dark ? "#94a3b8" : "#334155", fontSize: 11 }}
              tickFormatter={(v) => sci.format(v)}
              width={60}
            />
            <Tooltip
              contentStyle={{
                background: dark ? "#1e293b" : "#f1f5f9",
                border: "none",
              }}
              labelFormatter={(d) =>
                new Date(d).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })
              }
              formatter={(val, name) => [sci.format(val), name]}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual Flux"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="pred"
              name="Predicted Flux"
              stroke="#38bdf8"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <footer className={`text-center text-xs ${textSub}`}>
        Updated {now.toLocaleTimeString()} · UTC Timezone
      </footer>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);