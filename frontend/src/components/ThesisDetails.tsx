import type { ThesisResponse } from "../types";
import { BulletList } from "./BulletList";
import { Section } from "./Section";

// Key Themes / Risks / Investment Signals as flat editorial sections. Rendered
// as a fragment so each section is a direct sibling in the document flow (the
// 1px separators come from the sections themselves).
export function ThesisDetails({ thesis }: { thesis: ThesisResponse }) {
  return (
    <>
      <Section label="Key Themes">
        <BulletList items={thesis.key_themes} empty="No themes found." dotClass="bg-primary" />
      </Section>

      <Section label="Risks">
        <BulletList items={thesis.risks} empty="No risks found." dotClass="bg-error" />
        {thesis.key_risk_factors.length > 0 && (
          <p className="text-xs text-base-content/60 font-mono mt-3">
            {`Key risk factors: ${thesis.key_risk_factors.join(", ")}`}
          </p>
        )}
      </Section>

      <Section label="Investment Signals">
        <BulletList items={thesis.investment_signals} empty="No signals found." dotClass="bg-accent" />
      </Section>
    </>
  );
}
