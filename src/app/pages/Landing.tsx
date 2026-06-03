import { LandingNav } from "../components/landing/LandingNav";
import { Hero } from "../components/landing/Hero";
import { TrustBar } from "../components/landing/TrustBar";
import { Features } from "../components/landing/Features";
import { Pipeline } from "../components/landing/Pipeline";
import { LiveDemo } from "../components/landing/LiveDemo";
import { Multilingual } from "../components/landing/Multilingual";
import { FinalCTA } from "../components/landing/FinalCTA";
import { LandingFooter } from "../components/landing/LandingFooter";

export function Landing() {
  return (
    <div className="min-h-screen bg-[#F8FAFC] relative overflow-x-hidden">
      <LandingNav />
      <main>
        <Hero />
        <TrustBar />
        <Features />
        <Pipeline />
        <LiveDemo />
        <Multilingual />
        <FinalCTA />
      </main>
      <LandingFooter />
    </div>
  );
}
