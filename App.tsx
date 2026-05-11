/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { motion, AnimatePresence } from "motion/react";
import { 
  FileText, 
  BarChart3, 
  Wand2, 
  ArrowRight, 
  Github, 
  Linkedin, 
  Twitter, 
  Mail,
  Database,
  LineChart,
  Eye,
  ExternalLink
} from "lucide-react";
import { useState } from "react";

// --- Components ---

const Navbar = () => (
  <nav className="fixed top-0 w-full z-50 bg-background/60 backdrop-blur-xl border-b border-outline-variant/10">
    <div className="flex justify-between items-center px-8 h-20 max-w-7xl mx-auto">
      <div className="text-2xl font-bold tracking-tighter text-primary-container drop-shadow-[0_0_8px_rgba(0,255,200,0.4)] font-nav">
        NEON_TERMINAL
      </div>
      <div className="hidden md:flex gap-8 items-center">
        {["About", "Skills", "Experience", "Projects", "Education"].map((item) => (
          <a 
            key={item}
            href={`#${item.toLowerCase()}`}
            className={`font-medium font-nav tracking-tight transition-all duration-300 hover:text-primary-container hover:drop-shadow-[0_0_5px_rgba(0,255,200,0.8)] ${item === "Projects" ? "text-primary-container border-b-2 border-primary-container pb-1" : "text-on-surface"}`}
          >
            {item}
          </a>
        ))}
      </div>
      <button className="bg-primary-container text-on-primary-container px-6 py-2 rounded font-bold font-nav scale-95 active:scale-90 transition-transform hover:opacity-90">
        Hire Me
      </button>
    </div>
  </nav>
);

const ArchitectureDiagram = ({ type }: { type: string }) => {
  if (type === "nlp") {
    return (
      <svg className="w-full max-w-2xl text-primary-container stroke-current fill-none" viewBox="0 0 400 200">
        <motion.rect 
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1 }}
          className="stroke-2" height="40" width="80" x="20" y="80" 
        />
        <text className="fill-current stroke-none text-[10px] font-label" textAnchor="middle" x="60" y="105">PDF_INPUT</text>
        <motion.path 
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1, delay: 0.5 }}
          className="stroke-2" d="M100 100 L140 100" 
        />
        <motion.rect 
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1, delay: 1 }}
          className="stroke-2" height="80" width="100" x="140" y="60" 
        />
        <text className="fill-current stroke-none text-[10px] font-label" textAnchor="middle" x="190" y="105">TRANSFORMER_MOD</text>
        <motion.path 
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1, delay: 1.5 }}
          className="stroke-2" d="M240 100 L280 100" 
        />
        <motion.rect 
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1, delay: 2 }}
          className="stroke-2" height="40" width="80" x="280" y="80" 
        />
        <text className="fill-current stroke-none text-[10px] font-label" textAnchor="middle" x="320" y="105">RANK_SCORING</text>
        <circle className="fill-primary-container" cx="190" cy="140" r="4" />
        <path className="stroke-1 opacity-50" d="M190 140 V170 H320 V120" strokeDasharray="4" />
      </svg>
    );
  }
  if (type === "crypto") {
    return (
      <div className="grid grid-cols-2 gap-4">
        <div className="w-12 h-12 border border-primary-container/30 flex items-center justify-center rounded">
          <Database className="text-primary-container w-6 h-6" />
        </div>
        <div className="w-12 h-12 border border-primary-container/30 flex items-center justify-center rounded">
          <LineChart className="text-primary-container w-6 h-6" />
        </div>
      </div>
    );
  }
  if (type === "vision") {
    return (
      <div className="flex flex-col items-center gap-2">
        <Eye className="text-primary-container w-12 h-12 animate-pulse" />
        <span className="font-label text-primary-container text-[10px] tracking-widest">SYSTEM_SCANNING...</span>
      </div>
    );
  }
  return null;
};

const PipelineStepper = () => {
  const steps = [
    { name: "Extraction", icon: FileText, color: "text-primary-container" },
    { name: "Analysis", icon: BarChart3, color: "text-on-surface" },
    { name: "Optimization", icon: Wand2, color: "text-on-surface-variant" },
  ];

  return (
    <div className="mt-auto bg-surface/50 border border-outline-variant/20 rounded-lg p-6">
      <div className="flex items-center justify-between mb-8">
        <span className="font-label text-xs uppercase tracking-widest flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-primary-container animate-pulse"></span> 
          Pipeline Live Status
        </span>
        <div className="flex gap-1">
          {[1, 2, 3].map(i => <span key={i} className="w-1 h-1 rounded-full bg-outline-variant"></span>)}
        </div>
      </div>
      <div className="flex items-center justify-between relative">
        <div className="absolute top-1/2 left-0 w-full h-[1px] bg-outline-variant/30 -z-10"></div>
        {steps.map((step, idx) => (
          <motion.div 
            key={step.name}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.2 }}
            className="flex flex-col items-center gap-3"
          >
            <div className={`w-10 h-10 rounded-full flex items-center justify-center border ${idx === 0 ? "bg-primary-container text-on-primary-container shadow-[0_0_15px_rgba(0,255,200,0.4)] border-transparent" : "bg-surface border-outline-variant text-on-surface-variant"}`}>
              <step.icon className="w-4 h-4" />
            </div>
            <span className={`font-label text-[10px] ${idx === 0 ? "text-primary-container" : "text-on-surface-variant"}`}>
              {step.name}
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

const ProjectCard = ({ 
  title, 
  description, 
  tags, 
  status, 
  id, 
  archType, 
  isMain = false,
  footerText = ""
}: { 
  title: string; 
  description: string; 
  tags?: string[]; 
  status?: string; 
  id: string; 
  archType: string;
  isMain?: boolean;
  footerText?: string;
}) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <motion.section 
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`${isMain ? "lg:col-span-8" : "flex-1"} group relative overflow-hidden project-card glass-card rounded-xl p-8 transition-all duration-500 hover:scale-[1.01] neon-glow cursor-pointer`}
    >
      <div className="absolute top-0 right-0 p-6 opacity-20 font-label text-xs">{id}</div>
      
      {/* Architecture Diagram (Hover State) */}
      <AnimatePresence>
        {isHovered && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 flex items-center justify-center p-12 pointer-events-none bg-background/95 z-20"
          >
            <ArchitectureDiagram type={archType} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Static Content */}
      <div className={`transition-opacity duration-300 relative z-10 flex flex-col h-full ${isHovered ? "opacity-0" : "opacity-100"}`}>
        {tags && (
          <div className="flex flex-wrap gap-2 mb-6">
            {tags.map(tag => (
              <span key={tag} className="px-3 py-1 bg-surface rounded-full text-[10px] font-label text-primary-container border border-primary-container/20">
                {tag}
              </span>
            ))}
          </div>
        )}
        
        {status && (
          <span className={`font-label text-[10px] mb-4 tracking-[0.2em] uppercase ${status.includes("Deployed") ? "text-secondary" : "text-primary-container"}`}>
            Status: {status}
          </span>
        )}

        <h2 className={`${isMain ? "text-4xl" : "text-2xl"} font-headline font-bold mb-4 tracking-tight`}>
          {title}
        </h2>
        
        <p className={`text-on-surface-variant font-light leading-relaxed mb-8 ${isMain ? "max-w-xl text-base" : "text-sm"}`}>
          {description}
        </p>

        {isMain && <PipelineStepper />}

        {!isMain && (
          <div className="mt-auto flex justify-between items-center">
            <span className="text-[10px] font-label opacity-40">{footerText}</span>
            <ArrowRight className="text-primary-container w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </div>
        )}
      </div>
    </motion.section>
  );
};

const Footer = () => (
  <footer className="w-full py-12 border-t border-outline-variant/20 bg-background">
    <div className="flex flex-col md:flex-row justify-between items-center px-8 max-w-7xl mx-auto gap-6">
      <div className="text-sm font-bold text-on-surface font-label text-xs uppercase tracking-[0.2em]">
        TERMINAL_PORTFOLIO
      </div>
      <div className="flex gap-8">
        {[
          { name: "GitHub", icon: Github },
          { name: "LinkedIn", icon: Linkedin },
          { name: "Twitter", icon: Twitter },
          { name: "Email", icon: Mail },
        ].map((social) => (
          <a 
            key={social.name}
            href="#" 
            className="text-on-surface-variant font-label text-xs uppercase tracking-[0.2em] hover:text-primary-container transition-colors duration-200 flex items-center gap-2"
          >
            {social.name}
          </a>
        ))}
      </div>
      <div className="text-on-surface-variant font-label text-xs uppercase tracking-[0.2em]">
        © 2024 TERMINAL_PORTFOLIO // ALL RIGHTS RESERVED
      </div>
    </div>
  </footer>
);

// --- Main App ---

export default function App() {
  return (
    <div className="min-h-screen">
      <Navbar />
      
      <main className="pt-32 pb-20 px-6 max-w-7xl mx-auto">
        {/* Section Header */}
        <header className="mb-20">
          <div className="flex items-center gap-3 mb-4">
            <span className="w-12 h-[2px] bg-primary-container"></span>
            <span className="font-label text-primary-container text-sm tracking-[0.3em] uppercase">Deployment Log 042</span>
          </div>
          <h1 className="font-headline text-6xl md:text-8xl font-extrabold tracking-tighter text-on-surface">
            FEATURED<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-container to-secondary">PROJECTS</span>
          </h1>
        </header>

        {/* Projects Bento Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Main Project */}
          <ProjectCard 
            isMain
            id="// SYS_ROOT/PROJECT_01"
            title="AI-Powered ATS Resume Optimizer"
            description="An advanced NLP engine designed to reverse-engineer Applicant Tracking System algorithms. It provides semantic matching and keyword density optimization using fine-tuned BERT embeddings."
            tags={["NLP", "PYTHON", "BERT"]}
            archType="nlp"
          />

          {/* Secondary Projects Column */}
          <div className="lg:col-span-4 flex flex-col gap-8">
            <ProjectCard 
              id="// SYS_ROOT/PROJECT_02"
              title="Crypto Yield Forecaster"
              description="LSTM-based time series forecasting for DeFi liquidity pool returns."
              status="Deployed"
              archType="crypto"
              footerText="0x2A...4F9"
            />
            <ProjectCard 
              id="// SYS_ROOT/PROJECT_03"
              title="Neural Object Sentinel"
              description="Real-time object detection and tracking for edge-computing IoT devices."
              status="Research"
              archType="vision"
              footerText="SENTINEL_V3"
            />
          </div>

          {/* Wide Bento Project */}
          <motion.section className="lg:col-span-12 group relative glass-card rounded-xl p-10 transition-all duration-500 hover:scale-[1.005] neon-glow overflow-hidden">
            <div className="absolute -right-20 -top-20 w-96 h-96 bg-primary-container/5 rounded-full blur-[120px]"></div>
            
            <div className="grid lg:grid-cols-2 gap-12 relative z-10 items-center">
              <div>
                <div className="flex items-center gap-2 mb-6">
                  <span className="w-3 h-3 bg-secondary rounded-sm"></span>
                  <span className="font-label text-xs tracking-widest uppercase">Big Data Architecture</span>
                </div>
                <h2 className="text-4xl lg:text-5xl font-headline font-bold mb-6 leading-tight">Hyper-Personalization Engine for E-commerce</h2>
                <p className="text-on-surface-variant mb-8 text-lg font-light leading-relaxed">
                  Processing 10M+ daily events through a distributed Spark cluster to deliver real-time product recommendations with &lt; 50ms latency.
                </p>
                <div className="flex gap-6">
                  <button className="bg-primary-container text-on-primary-container px-8 py-3 rounded font-bold font-nav flex items-center gap-2 group/btn hover:opacity-90 transition-opacity">
                    View Case Study
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </button>
                  <button className="text-primary-container font-label text-sm uppercase tracking-widest border-b border-primary-container/30 hover:border-primary-container transition-colors flex items-center gap-2">
                    GitHub Repo
                    <ExternalLink className="w-3 h-3" />
                  </button>
                </div>
              </div>
              <div className="relative rounded-lg overflow-hidden h-64 lg:h-80 border border-outline-variant/30">
                <img 
                  className="w-full h-full object-cover opacity-60 grayscale hover:grayscale-0 transition-all duration-700" 
                  src="https://picsum.photos/seed/data-viz/1200/800"
                  alt="Big Data Visualization"
                  referrerPolicy="no-referrer"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent"></div>
              </div>
            </div>
          </motion.section>
        </div>
      </main>

      <Footer />
    </div>
  );
}
