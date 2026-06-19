"""Run business impact scenarios for the Home Credit risk model.

The goal is to translate model-policy assumptions into expected-loss estimates.
This is not a claim of realized savings; it is a planning tool for comparing
approval policy, approved default rate, and loss assumptions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    annual_applications: int
    approval_rate: float
    approved_default_rate: float
    average_funded_amount: float
    loss_given_default: float
    notes: str

    @property
    def approved_loans(self) -> float:
        return self.annual_applications * self.approval_rate

    @property
    def expected_defaults(self) -> float:
        return self.approved_loans * self.approved_default_rate

    @property
    def expected_loss(self) -> float:
        return (
            self.expected_defaults
            * self.average_funded_amount
            * self.loss_given_default
        )


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.2%}"


def build_scenarios(
    annual_applications: int,
    baseline_approval_rate: float,
    baseline_default_rate: float,
    model_approval_rate: float,
    model_default_rate: float,
    average_funded_amount: float,
    loss_given_default: float,
) -> list[Scenario]:
    return [
        Scenario(
            "baseline_policy",
            annual_applications,
            baseline_approval_rate,
            baseline_default_rate,
            average_funded_amount,
            loss_given_default,
            "Planning baseline from the Course 1 business context.",
        ),
        Scenario(
            "same_approval_volume_model_risk",
            annual_applications,
            baseline_approval_rate,
            model_default_rate,
            average_funded_amount,
            loss_given_default,
            "Uses the model observed approved default rate while holding approval volume constant.",
        ),
        Scenario(
            "current_validation_policy",
            annual_applications,
            model_approval_rate,
            model_default_rate,
            average_funded_amount,
            loss_given_default,
            "Uses the current validation approval rate and approved default rate "
            "at threshold 0.549.",
        ),
    ]


def write_csv(path: Path, scenarios: list[Scenario], baseline_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario",
                "annual_applications",
                "approval_rate",
                "approved_loans",
                "approved_default_rate",
                "expected_defaults",
                "average_funded_amount",
                "loss_given_default",
                "expected_loss",
                "savings_vs_baseline",
                "notes",
            ],
        )
        writer.writeheader()
        for scenario in scenarios:
            writer.writerow(
                {
                    "scenario": scenario.name,
                    "annual_applications": scenario.annual_applications,
                    "approval_rate": f"{scenario.approval_rate:.6f}",
                    "approved_loans": f"{scenario.approved_loans:.0f}",
                    "approved_default_rate": f"{scenario.approved_default_rate:.6f}",
                    "expected_defaults": f"{scenario.expected_defaults:.2f}",
                    "average_funded_amount": f"{scenario.average_funded_amount:.2f}",
                    "loss_given_default": f"{scenario.loss_given_default:.6f}",
                    "expected_loss": f"{scenario.expected_loss:.2f}",
                    "savings_vs_baseline": f"{baseline_loss - scenario.expected_loss:.2f}",
                    "notes": scenario.notes,
                }
            )


def build_markdown(scenarios: list[Scenario], baseline_loss: float) -> str:
    lines = [
        "# Business Impact Simulation",
        "",
        "## Purpose",
        "",
        "This report estimates expected credit loss under simple planning "
        "scenarios. It should be treated as directional business analysis, "
        "not as proven financial savings.",
        "",
        "## Assumptions",
        "",
        "| Assumption | Value |",
        "| --- | ---: |",
        f"| Annual applications | `{scenarios[0].annual_applications:,}` |",
        f"| Average funded amount | `{money(scenarios[0].average_funded_amount)}` |",
        f"| Loss given default | `{pct(scenarios[0].loss_given_default)}` |",
        f"| Baseline approval rate | `{pct(scenarios[0].approval_rate)}` |",
        f"| Baseline approved default rate | `{pct(scenarios[0].approved_default_rate)}` |",
        f"| Current model threshold | `0.549` |",
        f"| Current validation approval rate | `{pct(scenarios[2].approval_rate)}` |",
        f"| Current approved default rate | `{pct(scenarios[2].approved_default_rate)}` |",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Approval rate | Approved loans | Approved default rate | "
        "Expected defaults | Expected loss | Savings vs baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in scenarios:
        lines.append(
            "| "
            f"{scenario.name} | "
            f"`{pct(scenario.approval_rate)}` | "
            f"`{scenario.approved_loans:,.0f}` | "
            f"`{pct(scenario.approved_default_rate)}` | "
            f"`{scenario.expected_defaults:,.0f}` | "
            f"`{money(scenario.expected_loss)}` | "
            f"`{money(baseline_loss - scenario.expected_loss)}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Holding approval volume constant, reducing the approved default rate "
            "from `6.00%` to `5.37%` lowers expected loss in this planning "
            "scenario.",
            "- The current validation threshold has a much higher approval rate "
            "(`86.62%`) than the Course 1 planning baseline (`60.00%`). "
            "Because more applicants are approved, total expected loss can "
            "increase even when the approved default rate is lower.",
            "- This is why the model score should be paired with a business "
            "approval policy. A lower default rate is valuable, but approval "
            "volume, manual review capacity, and risk appetite determine the "
            "final business impact.",
            "- These estimates should be updated if the team changes threshold, "
            "average loan amount, approval policy, or loss-given-default "
            "assumptions.",
            "",
            "## Recommendation",
            "",
            "Use this simulation to frame business tradeoffs in the final "
            "presentation. Avoid claiming realized savings. A careful wording "
            "is: under planning assumptions, the model-supported policy can "
            "reduce expected loss at the same approval volume, but final "
            "business value depends on the approval policy chosen by the lender.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-applications", type=int, default=100_000)
    parser.add_argument("--baseline-approval-rate", type=float, default=0.60)
    parser.add_argument("--baseline-default-rate", type=float, default=0.06)
    parser.add_argument("--model-approval-rate", type=float, default=0.8662)
    parser.add_argument("--model-default-rate", type=float, default=0.0537)
    parser.add_argument("--average-funded-amount", type=float, default=8000.0)
    parser.add_argument("--loss-given-default", type=float, default=0.45)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("reports/business_impact_model_scenarios.csv"),
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=Path("reports/business_impact_summary.md"),
    )
    args = parser.parse_args()

    scenarios = build_scenarios(
        annual_applications=args.annual_applications,
        baseline_approval_rate=args.baseline_approval_rate,
        baseline_default_rate=args.baseline_default_rate,
        model_approval_rate=args.model_approval_rate,
        model_default_rate=args.model_default_rate,
        average_funded_amount=args.average_funded_amount,
        loss_given_default=args.loss_given_default,
    )
    baseline_loss = scenarios[0].expected_loss
    write_csv(args.csv_output, scenarios, baseline_loss)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.write_text(build_markdown(scenarios, baseline_loss), encoding="utf-8")
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.md_output}")


if __name__ == "__main__":
    main()
