from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_JSON = PROJECT_ROOT / "data_engineering" / "data" / "reports" / "evaluation_report_20260406_011331.json"
OUTPUT_DIR = PROJECT_ROOT / "submission"
OUTPUT_DOCX = OUTPUT_DIR / "Project_Report_Critic_CoT_Data_Engineering.docx"


def _text_runs(text: str) -> str:
    parts = str(text).split("\n")
    runs: List[str] = []
    for index, part in enumerate(parts):
        if index:
            runs.append("<w:br/>")
        preserve = ' xml:space="preserve"' if part.startswith(" ") or part.endswith(" ") else ""
        runs.append(f"<w:t{preserve}>{escape(part)}</w:t>")
    return "".join(runs)


def paragraph(text: str = "", style: str | None = None, bold: bool = False, align: str | None = None) -> str:
    p_pr = []
    if style:
        p_pr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        p_pr.append(f'<w:jc w:val="{align}"/>')
    p_pr_xml = f"<w:pPr>{''.join(p_pr)}</w:pPr>" if p_pr else ""
    r_pr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f"<w:p>{p_pr_xml}<w:r>{r_pr}{_text_runs(text)}</w:r></w:p>"


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def table(rows: Sequence[Sequence[object]]) -> str:
    border = (
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
        '</w:tblBorders>'
    )
    xml = ['<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>', border, "</w:tblPr>"]
    for row_index, row in enumerate(rows):
        xml.append("<w:tr>")
        for cell in row:
            shade = '<w:shd w:fill="D9EAF7"/>' if row_index == 0 else ""
            cell_text = "" if cell is None else str(cell)
            xml.append("<w:tc>")
            xml.append(f'<w:tcPr><w:tcW w:w="2400" w:type="dxa"/>{shade}</w:tcPr>')
            xml.append(paragraph(cell_text, bold=row_index == 0))
            xml.append("</w:tc>")
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def bullets(items: Iterable[str]) -> str:
    return "".join(paragraph(f"- {item}") for item in items)


def load_results() -> dict:
    if REPORT_JSON.exists():
        return json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    return {}


def results_rows(results: dict) -> list[list[object]]:
    rows: list[list[object]] = [
        ["Strategy", "Samples", "Correct", "Accuracy", "Avg Latency", "Avg Tokens", "Cost"]
    ]
    for key in ["baseline", "iter_refine", "filter", "majority"]:
        item = results.get(key, {})
        rows.append(
            [
                item.get("strategy", key),
                item.get("total_samples", ""),
                item.get("correct_count", ""),
                f"{item.get('accuracy_pct', '')}%",
                f"{item.get('avg_latency_ms', '')} ms",
                item.get("avg_tokens", ""),
                f"${item.get('total_cost_usd', '')}",
            ]
        )
    return rows


def build_document_xml(results: dict) -> str:
    generated_on = datetime.now().strftime("%d %B %Y")
    body: list[str] = []

    body.append(paragraph("Critic-CoT Reasoning Pipeline with Data Engineering", "Title", align="center"))
    body.append(paragraph("A Structured and Traceable LLM Reasoning Evaluation System", "Subtitle", align="center"))
    body.append(paragraph(""))
    body.append(paragraph("Submitted by: ______________________________", align="center"))
    body.append(paragraph("Register Number: ___________________________", align="center"))
    body.append(paragraph("Department / Semester: ______________________", align="center"))
    body.append(paragraph("Submitted to: ______________________________", align="center"))
    body.append(paragraph(f"Generated on: {generated_on}", align="center"))
    body.append(page_break())

    body.append(paragraph("1. Project Explanation", "Heading1"))
    body.append(
        bullets(
            [
                "Large Language Models generate answers using chain-of-thought reasoning, but this reasoning is often inconsistent or logically flawed.",
                "The Critic-CoT approach introduces a critic model that evaluates and improves the reasoning process.",
                "This project implements the Critic-CoT methodology and extends it into a data-engineered system.",
                "The reasoning steps, critiques, and revisions are treated as structured data rather than plain text.",
                "Data engineering pipelines are used to automate ingestion, processing, iteration, and storage of reasoning traces.",
                "The system ensures repeatability, traceability, and observability of the reasoning process.",
                "Automation removes manual intervention and reduces hidden errors.",
                "Monitoring tools are used to analyze reasoning quality over multiple runs.",
                "The project bridges research-level reasoning methods with system-level engineering practices.",
                "It demonstrates how LLM reasoning can be made more reliable and production-ready.",
            ]
        )
    )

    body.append(paragraph("2. Abstract", "Heading1"))
    body.append(
        paragraph(
            "Large Language Models can solve reasoning problems by generating chain-of-thought explanations, "
            "but the generated reasoning may contain arithmetic mistakes, incomplete logic, or unverified intermediate steps. "
            "This project implements a Critic-CoT based reasoning pipeline and extends it with a data engineering layer. "
            "The system ingests GSM8K math reasoning samples, executes multiple reasoning strategies through OpenRouter, "
            "stores reasoning traces in SQLite, exports structured CSV files, and generates evaluation reports. "
            "The main contribution is that reasoning is converted from unstructured text into auditable data containing "
            "questions, answers, steps, critiques, revisions, latency, token usage, and accuracy."
        )
    )

    body.append(paragraph("3. Problem Statement", "Heading1"))
    body.append(
        paragraph(
            "Traditional LLM reasoning experiments are often done inside notebooks where prompts and outputs are manually checked. "
            "This makes the process difficult to repeat, debug, compare, or monitor. A model may produce a final answer, "
            "but there is no reliable storage of how the answer was produced, which reasoning steps were used, whether a critic "
            "detected mistakes, or how much time and token cost were required. The problem addressed in this project is to make "
            "LLM reasoning traceable, repeatable, and measurable using a data engineering pipeline."
        )
    )

    body.append(paragraph("4. Objectives", "Heading1"))
    body.append(
        bullets(
            [
                "Implement Critic-CoT reasoning strategies for GSM8K math problems.",
                "Use a critic model to inspect and improve generated reasoning.",
                "Convert reasoning outputs into structured traces, steps, and critiques.",
                "Store all experiment data in a SQLite database.",
                "Export results to CSV for analysis and visualization.",
                "Generate automated evaluation reports with accuracy, latency, token usage, and cost.",
                "Provide simple one-command execution for demonstration.",
                "Support scheduled runs for future monitoring and automation.",
            ]
        )
    )

    body.append(paragraph("5. System Architecture", "Heading1"))
    body.append(
        paragraph(
            "GSM8K Dataset -> Data Ingestion -> Standardized Sample Format -> Critic-CoT Wrapper -> "
            "Reasoning Strategies -> SQLite Storage -> CSV Exports -> Evaluation Reports -> Faculty Review"
        )
    )
    body.append(
        paragraph(
            "The architecture separates research logic from engineering logic. The notebook contains the original Critic-CoT idea, "
            "while the data engineering layer provides ingestion, execution, persistence, reporting, and scheduling."
        )
    )

    body.append(paragraph("6. Dataset", "Heading1"))
    body.append(
        paragraph(
            "The project uses the GSM8K dataset, which contains grade-school math word problems with final numerical answers. "
            "The dataset is suitable because every sample requires multi-step reasoning, arithmetic calculation, and final answer extraction. "
            "The ingestion layer loads the dataset from HuggingFace, caches it locally for efficiency, and converts each record into a "
            "consistent schema containing id, question, expected answer, raw answer, source, split, and metadata."
        )
    )

    body.append(paragraph("7. Methodology", "Heading1"))
    body.append(
        table(
            [
                ["Strategy", "Description"],
                ["baseline", "Generates one direct chain-of-thought answer and checks it against the expected answer."],
                ["iter_refine", "Generates an answer, asks the critic to evaluate it, and refines the solution for a fixed number of iterations."],
                ["filter", "Generates multiple candidate answers and uses critic feedback to select the best candidate."],
                ["majority", "Generates multiple answers and selects the answer that appears most frequently after normalization."],
            ]
        )
    )

    body.append(paragraph("8. Data Engineering Implementation", "Heading1"))
    body.append(
        table(
            [
                ["Layer", "File", "Purpose"],
                ["Storage", "data_engineering/storage/reasoning_db.py", "Creates and manages SQLite tables for traces, steps, critiques, and daily metrics."],
                ["Data Ingestion", "data_engineering/pipeline/data_ingestion.py", "Loads GSM8K, caches data, normalizes samples, and prepares input records."],
                ["Critic-CoT Wrapper", "data_engineering/pipeline/critic_cot_wrapper.py", "Connects LLM API calls with reasoning, critique, refinement, verification, and all four strategies."],
                ["Main Pipeline", "data_engineering/pipeline/simple_pipeline.py", "Runs strategies on samples, stores traces, exports CSV files, and computes metrics."],
                ["Scaled Evaluation", "data_engineering/pipeline/run_evaluation.py", "Provides command-line execution with sample size, strategy selection, and automated reports."],
                ["Scheduler", "data_engineering/pipeline/scheduler.py", "Supports one-time or continuous scheduled runs and logs results to daily_results.csv."],
                ["Configuration", "data_engineering/config/settings.py", "Reads API keys, model settings, paths, and runtime configuration safely."],
            ]
        )
    )

    body.append(paragraph("9. Database Design", "Heading1"))
    body.append(
        table(
            [
                ["Table", "Purpose"],
                ["traces", "Stores one complete reasoning trace per question and strategy, including final answer, correctness, model, latency, tokens, cost, and trace JSON."],
                ["steps", "Stores individual reasoning steps extracted from each trace so the reasoning process can be inspected step by step."],
                ["critiques", "Stores critic feedback, detected error step, error flag, and verification information."],
                ["daily_metrics", "Stores aggregated run-level metrics for monitoring accuracy, latency, and strategy performance over time."],
            ]
        )
    )

    body.append(paragraph("10. Results", "Heading1"))
    body.append(
        paragraph(
            "A one-sample demonstration run was executed through the one-command script. The goal of this run was to validate that "
            "all strategies, storage, exports, and reports work end-to-end. Since the sample size is small, the result should be treated "
            "as functional validation rather than a final statistical benchmark."
        )
    )
    body.append(table(results_rows(results)))
    body.append(
        paragraph(
            "Generated artifacts include reasoning_traces.db, traces.csv, steps.csv, critiques.csv, daily_metrics.csv, "
            "per-strategy result CSV files, and Markdown/CSV/JSON evaluation reports."
        )
    )

    body.append(paragraph("11. How To Run The Project", "Heading1"))
    body.append(paragraph("Open PowerShell and run the following commands:"))
    body.append(paragraph("cd C:\\Users\\Dell\\Downloads\\sem6\\softcomputing\\project", "Code"))
    body.append(paragraph("powershell -ExecutionPolicy Bypass -File .\\run_once.ps1", "Code"))
    body.append(
        paragraph(
            "The cursor may blink while the system is running because it is waiting for live LLM API responses. "
            "A one-sample demo usually takes a few minutes because baseline, iterative refinement, filter, and majority strategies "
            "all make model calls."
        )
    )

    body.append(paragraph("12. How To Visualize The Work", "Heading1"))
    body.append(
        bullets(
            [
                "Open the latest evaluation report from data_engineering/data/reports.",
                "Open traces.csv to show complete reasoning traces.",
                "Open steps.csv to show the extracted reasoning steps.",
                "Open critiques.csv to show critic feedback and error detection.",
                "Open reasoning_traces.db using any SQLite viewer to show the four-table storage design.",
                "Show the PowerShell output to prove the pipeline runs end-to-end.",
            ]
        )
    )

    body.append(paragraph("13. Explanation For Faculty Review", "Heading1"))
    body.append(
        paragraph(
            "The main point to explain is that this is not only a prompt engineering project. The Critic-CoT method was converted "
            "into a data pipeline. Each reasoning attempt becomes a record. Each step becomes a row. Each critique becomes a row. "
            "Each run generates measurable metrics. Because of this, the system can be audited, compared, repeated, and scheduled. "
            "This is the data engineering contribution of the project."
        )
    )

    body.append(paragraph("14. Limitations", "Heading1"))
    body.append(
        bullets(
            [
                "The demonstration run uses a small sample size to reduce API time and avoid free model rate limits.",
                "LLM responses can vary between runs because generation is probabilistic.",
                "Free OpenRouter models may be slower or rate-limited compared with paid models.",
                "Accuracy on one sample should not be interpreted as full benchmark accuracy.",
                "The current system is focused on GSM8K style mathematical reasoning.",
            ]
        )
    )

    body.append(paragraph("15. Future Enhancements", "Heading1"))
    body.append(
        bullets(
            [
                "Run larger evaluations with 20, 50, or 100 GSM8K samples.",
                "Add visualization dashboards using Streamlit, Power BI, or Tableau.",
                "Compare multiple LLM models using the same data pipeline.",
                "Add richer error categories such as arithmetic error, reasoning gap, and answer extraction error.",
                "Deploy the scheduler as a daily monitoring job.",
                "Store results in PostgreSQL or cloud storage for multi-user access.",
            ]
        )
    )

    body.append(paragraph("16. Conclusion", "Heading1"))
    body.append(
        paragraph(
            "This project successfully extends Critic-CoT from a notebook-level reasoning method into a structured data engineering system. "
            "It automates dataset loading, reasoning execution, critic-based evaluation, trace storage, CSV export, reporting, and scheduling. "
            "The final system demonstrates how LLM reasoning can be treated as data, making the reasoning process more transparent, repeatable, "
            "and suitable for production-style evaluation."
        )
    )

    body.append(paragraph("17. Submission Checklist", "Heading1"))
    body.append(
        bullets(
            [
                "Upload this Word report or export it as PDF.",
                "Upload source code excluding secrets and virtual environments.",
                "Upload generated result CSV files and evaluation reports.",
                "Do not upload config.py because it contains the API key.",
                "Do not upload .env, .venv, venv, cache folders, or temporary folders.",
                "Rotate or delete the OpenRouter API key after the demo/submission.",
            ]
        )
    )

    sect_pr = (
        '<w:sectPr>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
        '</w:sectPr>'
    )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}{sect_pr}</w:body>"
        "</w:document>"
    )


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="240"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="36"/><w:color w:val="1F4E79"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle">
    <w:name w:val="Subtitle"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:i/><w:sz w:val="24"/><w:color w:val="5B6770"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="28"/><w:color w:val="1F4E79"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Code">
    <w:name w:val="Code"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="80"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="20"/><w:color w:val="404040"/></w:rPr>
  </w:style>
</w:styles>"""


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def core_xml() -> str:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Critic-CoT Reasoning Pipeline with Data Engineering</dc:title>
  <dc:subject>LLM reasoning, Critic-CoT, data engineering</dc:subject>
  <dc:creator>Project Team</dc:creator>
  <cp:keywords>Critic-CoT, GSM8K, SQLite, data pipeline, LLM evaluation</cp:keywords>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
  xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex Report Generator</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
</Properties>"""


def write_docx() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    with zipfile.ZipFile(OUTPUT_DOCX, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", content_types_xml())
        package.writestr("_rels/.rels", rels_xml())
        package.writestr("word/_rels/document.xml.rels", document_rels_xml())
        package.writestr("word/document.xml", build_document_xml(results))
        package.writestr("word/styles.xml", styles_xml())
        package.writestr("docProps/core.xml", core_xml())
        package.writestr("docProps/app.xml", app_xml())
    return OUTPUT_DOCX


if __name__ == "__main__":
    path = write_docx()
    print(path)
