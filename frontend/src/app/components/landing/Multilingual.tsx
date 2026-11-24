import { motion } from "motion/react";
import { Globe2 } from "lucide-react";

const samples = [
  {
    flag: "🇬🇧",
    lang: "English",
    code: "EN",
    raw: "Tractor Model: Mahindra 575 DI",
    extracted: { model: "Mahindra 575 DI", hp: "50 HP" },
    note: "Standard Latin script, typed",
  },
  {
    flag: "🇮🇳",
    lang: "Hindi",
    code: "HI",
    raw: "ट्रैक्टर मॉडल: महिंद्रा 575 डीआई",
    extracted: { model: "Mahindra 575 DI", hp: "50 एचपी" },
    note: "Devanagari script, transliterated",
  },
  {
    flag: "🇮🇳",
    lang: "Gujarati",
    code: "GU",
    raw: "ટ્રેક્ટર મોડેલ: મહિન્દ્રા 575 ડીઆઈ",
    extracted: { model: "Mahindra 575 DI", hp: "50 બળ" },
    note: "Gujarati script, vernacular HP",
  },
];

export function Multilingual() {
  return (
    <section
      id="languages"
      className="section relative bg-gradient-to-b from-transparent via-pink-50/20 to-transparent"
    >
      <div className="container-wide relative">
        <div className="max-w-2xl mx-auto text-center mb-14">
          <span className="stage-badge mb-5">
            <Globe2 className="w-3.5 h-3.5" />
            Multilingual
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 tracking-tight mb-5">
            Three scripts. <span className="gradient-text">One unified output.</span>
          </h2>
          <p className="text-lg text-gray-600 leading-relaxed">
            The same field maps to the same canonical value, no matter what script the dealer used.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {samples.map((s, i) => (
            <motion.div
              key={s.code}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="frost-card p-6 tilt-card"
            >
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-2.5">
                  <div className="text-2xl">{s.flag}</div>
                  <div>
                    <div className="text-base font-bold text-gray-900">{s.lang}</div>
                    <div className="text-[10px] uppercase tracking-widest text-gray-500 font-semibold">
                      {s.code} · {s.note}
                    </div>
                  </div>
                </div>
              </div>

              {/* Raw OCR */}
              <div className="mb-4">
                <div className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold mb-2">
                  Raw OCR
                </div>
                <div className="px-3 py-3 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-800 font-medium min-h-[56px] flex items-center">
                  {s.raw}
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center my-2">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-100 to-pink-100 flex items-center justify-center">
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path
                      d="M7 1V13M7 13L1 7M7 13L13 7"
                      stroke="#6C5CE7"
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                  </svg>
                </div>
              </div>

              {/* Extracted */}
              <div>
                <div className="text-[10px] uppercase tracking-wider text-[#6C5CE7] font-semibold mb-2">
                  Extracted
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between px-3 py-2 bg-purple-50 border border-purple-200 rounded-lg">
                    <span className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold">
                      Model
                    </span>
                    <span className="text-sm font-bold text-[#6C5CE7]">{s.extracted.model}</span>
                  </div>
                  <div className="flex items-center justify-between px-3 py-2 bg-purple-50 border border-purple-200 rounded-lg">
                    <span className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold">
                      HP
                    </span>
                    <span className="text-sm font-bold text-[#6C5CE7]">{s.extracted.hp}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
