import pdfplumber
import pandas as pd
import re
import os


def extract_metrics_from_pdf(pdf_path):
    """Extract data from PDF that can be used to calculate NIRF metrics"""

    metrics = {}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

            # Extract Institute ID
            id_match = re.search(r"Institute ID[:\s]+([A-Z0-9\-]+)", all_text)
            if id_match:
                metrics["Institute_ID"] = id_match.group(1)

            # Extract Institute Name
            name_match = re.search(r"Institute Name[:\s]+([^\n]+)", all_text)
            if name_match:
                metrics["Institute"] = name_match.group(1).strip()

            # Faculty data - find numbers before "Faculty"
            faculty_pattern = r"(\d+)\s+(\d+)\s+(\d+).*?Faculty"
            faculty_match = re.search(faculty_pattern, all_text[:500])
            if faculty_match:
                try:
                    metrics["Faculty_PhD"] = int(faculty_match.group(1))
                    metrics["Faculty_Total"] = int(faculty_match.group(2))
                    metrics["Faculty_Women"] = int(faculty_match.group(3))
                except:
                    pass

            # Student data
            total_students = 0

            # Find all student numbers (UG, PG, PHD rows)
            student_pattern = r"(?:UG|PG|PHD)\s+[^\d]+(\d+)\s+(\d+)\s+(\d+)"
            for match in re.finditer(student_pattern, all_text[:3000]):
                try:
                    intake = int(match.group(1))
                    enrolled = int(match.group(2))
                    total_students += enrolled
                except:
                    pass

            metrics["Total_Students"] = total_students

            # Foreign students
            if "Outside Country" in all_text:
                parts = all_text.split("Outside Country")
                if len(parts) > 1:
                    foreign_text = parts[1][:100]
                    foreign_nums = re.findall(r"\d+", foreign_text)
                    if foreign_nums:
                        try:
                            metrics["Foreign_Students"] = int(foreign_nums[0])
                        except:
                            pass

            # Financial data - Total Annual Expenditure
            exp_pattern = r"([\d,\.]+)\s+([\d,\.]+)\s+([\d,\.]+)"
            for match in re.finditer(exp_pattern, all_text):
                try:
                    cap = float(match.group(1).replace(",", ""))
                    op = float(match.group(2).replace(",", ""))
                    tot = float(match.group(3).replace(",", ""))
                    if tot > 1000000:  # Valid expenditure
                        metrics["Capital_Expenditure"] = cap
                        metrics["Operational_Expenditure"] = op
                        metrics["Total_Expenditure"] = tot
                except:
                    pass

            # Publications
            wos_match = re.search(r"Web of Science\s+(\d+)", all_text)
            if wos_match:
                try:
                    metrics["WOS_Publications"] = int(wos_match.group(1))
                except:
                    pass

            scopus_match = re.search(r"Scopus\s+(\d+)", all_text)
            if scopus_match:
                try:
                    metrics["Scopus_Publications"] = int(scopus_match.group(1))
                except:
                    pass

            # Patents
            if "Patents Granted" in all_text:
                parts = all_text.split("Patents Granted")
                if len(parts) > 1:
                    patent_text = parts[1][:100]
                    nums = re.findall(r"\d+", patent_text)
                    if len(nums) >= 2:
                        try:
                            metrics["Patents_Granted"] = int(nums[0])
                            metrics["Patents_Published"] = int(nums[1])
                        except:
                            pass

            # Research funding
            if "Sponsored Research" in all_text:
                parts = all_text.split("Sponsored Research")
                if len(parts) > 1:
                    amounts = re.findall(r"([\d,\.]+)", parts[1][:300])
                    if amounts:
                        try:
                            metrics["Research_Funding"] = float(
                                amounts[0].replace(",", "")
                            )
                        except:
                            pass

    except Exception as e:
        pass

    return metrics


def calculate_nirf_metrics(raw):
    """Calculate NIRF sub-metrics from raw data"""

    metrics = {}

    total_students = raw.get("Total_Students", 0) or 0
    faculty = raw.get("Faculty_Total", 0) or 0
    phd = raw.get("Faculty_PhD", 0) or 0
    expenditure = raw.get("Total_Expenditure", 0) or 0
    pubs = (raw.get("WOS_Publications", 0) or 0) + (
        raw.get("Scopus_Publications", 0) or 0
    )
    patents = (raw.get("Patents_Granted", 0) or 0) + (
        raw.get("Patents_Published", 0) or 0
    )
    foreign = raw.get("Foreign_Students", 0) or 0

    # SS (Student Strength)
    if total_students > 0:
        metrics["SS"] = min(100, (total_students / 5000) * 100)
    else:
        metrics["SS"] = None

    # FSR (Faculty-Student Ratio)
    if faculty > 0 and total_students > 0:
        ratio = faculty / total_students
        metrics["FSR"] = min(100, ratio * 100)
    else:
        metrics["FSR"] = None

    # FQE (Faculty with PhD)
    if phd > 0 and faculty > 0:
        metrics["FQE"] = (phd / faculty) * 100
    else:
        metrics["FQE"] = None

    # FRU (Financial Resources)
    if expenditure > 0:
        metrics["FRU"] = min(100, (expenditure / 1000000000) * 100)
    else:
        metrics["FRU"] = None

    # PU (Publications)
    if pubs > 0:
        metrics["PU"] = min(100, (pubs / 3000) * 100)
    else:
        metrics["PU"] = None

    # QP (Patents)
    if patents > 0:
        metrics["QP"] = min(100, (patents / 300) * 100)
    else:
        metrics["QP"] = None

    # IPR
    ip_value = (raw.get("Patents_Published", 0) or 0) + (
        raw.get("Research_Funding", 0) or 0
    ) / 10000000
    if ip_value > 0:
        metrics["IPR"] = min(100, ip_value)
    else:
        metrics["IPR"] = None

    # FPPP (Foreign Students)
    if foreign > 0:
        metrics["FPPP"] = min(100, foreign * 10)
    else:
        metrics["FPPP"] = None

    return metrics


# Main execution
print("Extracting metrics from all PDFs...")

pdf_dir = "data/pdfs"
all_raw_metrics = []

# Process all PDFs for each year
for year_folder in sorted(os.listdir(pdf_dir)):
    year_path = os.path.join(pdf_dir, year_folder)
    if not os.path.isdir(year_path):
        continue

    year = year_folder
    pdf_files = [f for f in os.listdir(year_path) if f.endswith(".pdf")]

    for i, pdf_file in enumerate(pdf_files):
        if i % 50 == 0:
            print(f"Year {year}: Processing {i}/{len(pdf_files)}...")

        pdf_path = os.path.join(year_path, pdf_file)
        raw = extract_metrics_from_pdf(pdf_path)

        if raw.get("Institute"):
            raw["Year"] = int(year)
            raw["Source_File"] = pdf_file
            all_raw_metrics.append(raw)

print(f"\nExtracted data from {len(all_raw_metrics)} PDFs")

# Calculate metrics
print("Calculating NIRF sub-metrics...")
for raw in all_raw_metrics:
    calc = calculate_nirf_metrics(raw)
    raw.update(calc)

# Create DataFrame
df = pd.DataFrame(all_raw_metrics)

# Select relevant columns
cols = ["Year", "Institute", "SS", "FSR", "FQE", "FRU", "PU", "QP", "IPR", "FPPP"]
df_metrics = df[[c for c in cols if c in df.columns]]

print(f"\nExtracted metrics sample:")
print(df_metrics.head(10))

# Save
df.to_csv("data/csv/extracted_pdf_metrics.csv", index=False)
print(f"\nSaved to: data/csv/extracted_pdf_metrics.csv")
