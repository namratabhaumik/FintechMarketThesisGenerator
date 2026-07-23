import type { ThesisResponse } from "../types";
import { BulletList } from "./BulletList";
import { Collapsible } from "./Collapsible";

// Key Themes / Risks / Investment Signals. Rendered after Sources/Raw Summary.
export function ThesisDetails({ thesis }: { thesis: ThesisResponse }) {
  return (
    <section className="space-y-4">
      <Collapsible summary="Key Themes" defaultOpen>
        <BulletList items={thesis.key_themes} empty="No themes found." dotClass="bg-primary" />
      </Collapsible>

      <Collapsible summary="Risks" defaultOpen>
        <div>
          <BulletList items={thesis.risks} empty="No risks found." dotClass="bg-error" />
          {thesis.key_risk_factors.length > 0 && (
            <p className="text-xs text-base-content/60 font-mono mt-3">
              {`Key risk factors: ${thesis.key_risk_factors.join(", ")}`}
            </p>
          )}
        </div>
      </Collapsible>

      <Collapsible summary="Investment Signals" defaultOpen>
        <BulletList items={thesis.investment_signals} empty="No signals found." dotClass="bg-accent" />
      </Collapsible>
    </section>
  );
}
