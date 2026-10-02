"""Generate a synthetic corpus of 12 realistic PDFs for testing/evaluation.

Writes to data/raw/ by default.  Each PDF has realistic legal or policy text,
section headings, and page numbers in the footer.

Document mix:
  3 MSAs (Master Service Agreements)
  2 NDAs (Non-Disclosure Agreements)
  3 Policies (HR, Security, Remote Work)
  2 Handbooks (Employee, Contractor)
  2 DPAs (Data Processing Agreements)

Usage:
    python scripts/00_synthesize_corpus.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog

# ── Logging must be configured before any log.* call ──────────────────────────
from docintel.utils.logging import configure_logging

configure_logging()
log = structlog.get_logger()

# ── reportlab imports ──────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )
except ImportError as exc:
    log.error("reportlab_not_installed", error=str(exc))
    sys.exit(1)

# ── Page-number footer callback ────────────────────────────────────────────────

def _footer(canvas, doc) -> None:  # type: ignore[override]
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(
        letter[0] / 2.0,
        0.4 * inch,
        f"Page {doc.page}",
    )
    canvas.restoreState()


# ── Style factory ──────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocTitle", parent=base["Heading1"], fontSize=15, spaceAfter=14
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontSize=12, spaceAfter=8, spaceBefore=14
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"], fontSize=10, spaceAfter=10, leading=14
        ),
        "meta": ParagraphStyle(
            "Meta", parent=base["Normal"], fontSize=9, textColor="grey", spaceAfter=6
        ),
    }


# ── Generic PDF builder ────────────────────────────────────────────────────────

def _make_pdf(
    output_path: Path,
    title: str,
    subtitle: str,
    sections: list[tuple[str, str]],
) -> None:
    """Write a multi-section PDF to output_path."""
    styles = _build_styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=0.75 * inch,
    )

    story = [
        Paragraph(title, styles["title"]),
        Paragraph(subtitle, styles["meta"]),
        HRFlowable(width="100%", thickness=0.5),
        Spacer(1, 0.15 * inch),
    ]

    for i, (heading, body) in enumerate(sections):
        story.append(Paragraph(heading, styles["h2"]))
        for para in body.split("\n\n"):
            para = para.strip()
            if para:
                story.append(Paragraph(para, styles["body"]))
        story.append(Spacer(1, 0.05 * inch))
        # Force a page break every 4 sections to ensure multi-page output
        if (i + 1) % 4 == 0 and i + 1 < len(sections):
            story.append(PageBreak())

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


# ── Document content definitions ───────────────────────────────────────────────

# MSA boilerplate bodies
_MSA_SECTIONS = [
    (
        "1. PARTIES AND RECITALS",
        "This Master Services Agreement (\"Agreement\") is entered into as of January 1, 2024, "
        "by and between TechCorp Solutions, Inc., a Delaware corporation (\"Service Provider\"), "
        "and the customer identified in the applicable Statement of Work (\"Customer\").\n\n"
        "WHEREAS, Service Provider desires to provide certain professional and technology services "
        "to Customer, and Customer desires to obtain such services from Service Provider, subject "
        "to the terms and conditions set forth herein.",
    ),
    (
        "2. DEFINITIONS",
        '"Confidential Information" means any non-public information disclosed by one party '
        '("Disclosing Party") to the other ("Receiving Party") that is marked confidential or '
        "that reasonably should be understood to be confidential given the nature of the information "
        "and the circumstances of disclosure.\n\n"
        '"Services" means the professional, consulting, software development, and related '
        "services described in each Statement of Work executed by the parties.\n\n"
        '"Deliverable" means any work product, software, documentation, report, or other '
        "tangible or intangible output produced by Service Provider in performing the Services.",
    ),
    (
        "3. SERVICES",
        "Service Provider shall perform the Services set forth in each Statement of Work (\"SOW\") "
        "executed by the parties. Each SOW shall specify the scope of work, timeline, milestones, "
        "fees, and any special terms applicable to that engagement.\n\n"
        "Service Provider shall assign qualified personnel to perform the Services and shall "
        "maintain such personnel throughout the engagement unless Customer consents to substitution "
        "in writing. Service Provider may use subcontractors with Customer's prior written consent.",
    ),
    (
        "4. FEES AND PAYMENT",
        "Customer shall pay Service Provider the fees set forth in each SOW within thirty (30) days "
        "of the invoice date. Invoices not paid within thirty (30) days shall accrue interest at "
        "one and one-half percent (1.5%) per month, or the maximum rate permitted by law.\n\n"
        "All fees are exclusive of applicable taxes. Customer is responsible for all sales, use, "
        "value-added, and similar taxes levied on the Services, excluding taxes on Service "
        "Provider's income.",
    ),
    (
        "5. INTELLECTUAL PROPERTY",
        "Subject to full payment, Service Provider assigns to Customer all right, title, and "
        "interest in any Deliverables specifically developed for Customer under this Agreement. "
        "Service Provider retains ownership of all pre-existing intellectual property, tools, "
        "frameworks, and methodologies (\"Background IP\").\n\n"
        "Service Provider grants Customer a non-exclusive, perpetual, royalty-free license to "
        "use any Background IP embedded in the Deliverables solely as necessary to use the "
        "Deliverables for their intended purpose.",
    ),
    (
        "6. CONFIDENTIALITY",
        "Each party agrees to hold the other party's Confidential Information in strict confidence "
        "and not to disclose such information to any third party without prior written consent. "
        "Each party shall use the Confidential Information solely for purposes of performing its "
        "obligations under this Agreement.\n\n"
        "Confidentiality obligations survive termination of this Agreement for a period of five "
        "(5) years. The foregoing obligations do not apply to information that: (a) is or becomes "
        "publicly available through no breach of this Agreement; (b) was known prior to disclosure; "
        "or (c) is independently developed without use of Confidential Information.",
    ),
    (
        "7. TERM AND TERMINATION",
        "This Agreement commences on the Effective Date and continues for one (1) year, renewing "
        "automatically for successive one-year terms unless either party provides sixty (60) days' "
        "written notice of non-renewal.\n\n"
        "Either party may terminate this Agreement for cause upon thirty (30) days' written notice "
        "if the other party materially breaches this Agreement and fails to cure such breach within "
        "the notice period. Either party may terminate for convenience upon ninety (90) days' "
        "written notice.",
    ),
    (
        "8. LIMITATION OF LIABILITY",
        "IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER FOR ANY INDIRECT, INCIDENTAL, "
        "SPECIAL, CONSEQUENTIAL, EXEMPLARY, OR PUNITIVE DAMAGES, INCLUDING LOST PROFITS OR "
        "REVENUE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.\n\n"
        "Each party's total cumulative liability arising out of or related to this Agreement "
        "shall not exceed the total fees paid or payable to Service Provider in the twelve (12) "
        "months preceding the event giving rise to the claim.",
    ),
    (
        "9. INDEMNIFICATION",
        "Each party (\"Indemnitor\") shall defend, indemnify, and hold harmless the other party "
        "and its officers, directors, and employees from and against any third-party claims arising "
        "from: (a) Indemnitor's material breach of this Agreement; (b) Indemnitor's gross "
        "negligence or willful misconduct; or (c) infringement of any third-party intellectual "
        "property right by Indemnitor's materials.\n\n"
        "The indemnified party shall promptly notify the Indemnitor of any claim, cooperate "
        "reasonably in the defense, and grant the Indemnitor control of the defense and settlement.",
    ),
    (
        "10. DISPUTE RESOLUTION",
        "Any dispute arising out of or relating to this Agreement shall first be subject to "
        "good-faith negotiation between senior representatives of each party for thirty (30) days. "
        "If negotiation fails, the dispute shall be submitted to binding arbitration under the "
        "rules of the American Arbitration Association.\n\n"
        "The arbitration shall take place in New York, New York. The arbitrator's decision shall "
        "be final and binding, and judgment may be entered in any court of competent jurisdiction.",
    ),
    (
        "11. GENERAL PROVISIONS",
        "This Agreement constitutes the entire agreement between the parties with respect to its "
        "subject matter and supersedes all prior agreements and understandings. This Agreement may "
        "not be amended except in a writing signed by authorized representatives of both parties.\n\n"
        "If any provision of this Agreement is found to be unenforceable, the remaining provisions "
        "continue in full force. The failure to enforce any provision shall not constitute a "
        "waiver. This Agreement is governed by the laws of the State of New York.",
    ),
]

_NDA_SECTIONS = [
    (
        "1. PARTIES",
        "This Non-Disclosure Agreement (\"Agreement\") is entered into as of the date last signed "
        "below, between TechCorp Solutions, Inc. (\"Disclosing Party\") and the receiving party "
        "identified in the signature block (\"Receiving Party\"). The parties wish to explore a "
        "potential business relationship (the \"Purpose\") and, in connection therewith, the "
        "Disclosing Party may disclose certain Confidential Information to the Receiving Party.",
    ),
    (
        "2. DEFINITION OF CONFIDENTIAL INFORMATION",
        "\"Confidential Information\" means any information disclosed by the Disclosing Party "
        "to the Receiving Party, either directly or indirectly, in writing, orally, or by "
        "inspection of tangible objects, that is designated as confidential or that reasonably "
        "should be understood to be confidential given the nature of the information and "
        "circumstances of disclosure.\n\n"
        "Confidential Information includes, without limitation: business plans, financial data, "
        "technical data, software source code, customer lists, marketing plans, product "
        "roadmaps, pricing, trade secrets, and any other information of a proprietary nature.",
    ),
    (
        "3. OBLIGATIONS OF THE RECEIVING PARTY",
        "The Receiving Party shall: (a) hold the Confidential Information in strict confidence "
        "using no less than reasonable care; (b) not disclose the Confidential Information to "
        "any third party without prior written consent from the Disclosing Party; and (c) use "
        "the Confidential Information solely for the Purpose.\n\n"
        "The Receiving Party may disclose Confidential Information to its employees, officers, "
        "and advisors who have a need to know for the Purpose, provided that such persons are "
        "bound by confidentiality obligations at least as protective as those herein.",
    ),
    (
        "4. EXCLUSIONS FROM CONFIDENTIAL INFORMATION",
        "The obligations of this Agreement do not apply to information that: (a) is or becomes "
        "publicly known through no breach of this Agreement by the Receiving Party; (b) is "
        "rightfully received by the Receiving Party from a third party without restriction; "
        "(c) is independently developed by the Receiving Party without use of or reference to "
        "the Confidential Information; or (d) is required to be disclosed by law or court order, "
        "provided that the Receiving Party gives prompt written notice to the Disclosing Party "
        "and cooperates to seek a protective order.",
    ),
    (
        "5. RETURN OR DESTRUCTION OF CONFIDENTIAL INFORMATION",
        "Upon written request by the Disclosing Party, the Receiving Party shall promptly return "
        "or destroy all Confidential Information and any copies thereof, and certify in writing "
        "that it has done so. Notwithstanding the foregoing, the Receiving Party may retain "
        "archival copies subject to the ongoing confidentiality obligations herein.",
    ),
    (
        "6. TERM AND TERMINATION",
        "This Agreement is effective from the date of execution and continues for a period of "
        "two (2) years, unless earlier terminated by either party upon thirty (30) days' written "
        "notice. The confidentiality obligations survive termination for an additional three "
        "(3) years after the expiration or termination of this Agreement.",
    ),
    (
        "7. REMEDIES",
        "The Receiving Party acknowledges that any breach of this Agreement may cause "
        "irreparable harm to the Disclosing Party for which monetary damages may be inadequate. "
        "The Disclosing Party shall be entitled to seek equitable relief, including injunction "
        "and specific performance, in addition to all other remedies available at law or in equity.",
    ),
    (
        "8. MISCELLANEOUS",
        "This Agreement constitutes the entire agreement between the parties concerning its "
        "subject matter and supersedes all prior discussions and agreements. No modification "
        "shall be effective unless in writing signed by both parties. If any provision is held "
        "unenforceable, the remaining provisions remain in effect. This Agreement is governed "
        "by the laws of the State of Delaware, without regard to conflict-of-law principles.",
    ),
]

_POLICY_HR_SECTIONS = [
    (
        "1. PURPOSE AND SCOPE",
        "This Human Resources Policy establishes the standards, expectations, and procedures "
        "governing employment at TechCorp Solutions, Inc. (\"Company\"). It applies to all "
        "full-time and part-time employees and covers the full employment lifecycle from "
        "recruitment through separation.\n\n"
        "This policy is reviewed annually by the People Operations team and approved by the "
        "Chief People Officer. Managers are responsible for communicating this policy to their "
        "direct reports and ensuring compliance within their teams.",
    ),
    (
        "2. RECRUITMENT AND HIRING",
        "All open positions must be posted for a minimum of five (5) business days before an "
        "offer is extended. Job descriptions must be approved by the hiring manager's department "
        "head and reviewed by HR to ensure compliance with applicable employment laws.\n\n"
        "Interview panels must include at least one member outside the direct reporting chain "
        "to ensure diverse evaluation. References must be checked for all final candidates "
        "prior to offer. Background checks are required for roles with access to sensitive "
        "financial data or critical systems.",
    ),
    (
        "3. COMPENSATION AND BENEFITS",
        "Compensation is reviewed annually during the performance review cycle. Adjustments "
        "are based on individual performance, internal equity, and market benchmarks. Employees "
        "are paid semi-monthly on the 15th and last business day of each month.\n\n"
        "Benefits include: medical, dental, and vision insurance (effective first day of "
        "employment); 401(k) with 4% employer match (vesting over 3 years); 20 days paid "
        "vacation per year; 10 days paid sick leave; and 12 weeks paid parental leave.",
    ),
    (
        "4. PERFORMANCE MANAGEMENT",
        "All employees receive a formal performance review in Q4 of each year. Mid-year "
        "check-ins are conducted in Q2. Ratings are calibrated within departments to ensure "
        "consistency. Employees rated \"Below Expectations\" are placed on a 60-day "
        "Performance Improvement Plan (PIP) with specific, measurable goals.\n\n"
        "Managers are trained on unconscious bias and are required to document performance "
        "observations throughout the year rather than relying solely on recency.",
    ),
    (
        "5. CODE OF CONDUCT",
        "All employees are expected to conduct themselves professionally and ethically in all "
        "interactions with colleagues, customers, and partners. Harassment, discrimination, "
        "retaliation, and workplace violence are strictly prohibited.\n\n"
        "Conflicts of interest must be disclosed to HR. Outside employment that competes with "
        "the Company or interferes with job performance requires prior written approval. "
        "Employees must not accept gifts valued above $50 from vendors or customers.",
    ),
    (
        "6. LEAVE POLICIES",
        "Vacation leave accrues at a rate of 1.67 days per month and may carry over up to "
        "10 days at year end. Requests must be submitted in the HR system at least five "
        "business days in advance for absences of one week or more.\n\n"
        "Sick leave may be used for personal illness, medical appointments, or care of an "
        "immediate family member. Sick leave does not carry over and does not have a cash "
        "value upon separation. FMLA leave is available per applicable federal law.",
    ),
    (
        "7. DISCIPLINARY PROCEDURES",
        "Progressive discipline is used for most policy violations: verbal warning, written "
        "warning, final written warning, and termination. The severity and repetition of the "
        "violation determines the appropriate step. HR and legal review is required before "
        "any termination.\n\n"
        "Certain violations — including fraud, theft, violence, or severe harassment — may "
        "result in immediate termination without prior warning steps. All disciplinary actions "
        "must be documented and placed in the employee's personnel file.",
    ),
    (
        "8. SEPARATION",
        "Voluntary resignations require two (2) weeks' written notice for individual contributors "
        "and four (4) weeks for managers and above. Involuntary terminations are administered by "
        "HR in coordination with the employee's manager and, where applicable, legal counsel.\n\n"
        "All Company property, including equipment, access badges, and confidential documents, "
        "must be returned on or before the last day of employment. IT access is revoked on the "
        "last day. Final paychecks are issued within the timeframe required by applicable law.",
    ),
]

_POLICY_SECURITY_SECTIONS = [
    (
        "1. PURPOSE AND SCOPE",
        "This Information Security Policy defines the standards and controls required to protect "
        "TechCorp Solutions' information assets, systems, and data from unauthorized access, "
        "disclosure, modification, and destruction. It applies to all employees, contractors, "
        "and third-party vendors who access Company systems or data.",
    ),
    (
        "2. DATA CLASSIFICATION",
        "All Company data is classified into one of four categories:\n\n"
        "CONFIDENTIAL: Personal data, financial records, source code, trade secrets, attorney "
        "work product. Access restricted to authorized personnel on a need-to-know basis.\n\n"
        "INTERNAL: Business plans, internal communications, non-public product roadmaps. "
        "Available to all employees but not for external disclosure.\n\n"
        "PUBLIC: Marketing materials, public documentation. No restriction on distribution.",
    ),
    (
        "3. ACCESS CONTROL",
        "Access to systems and data is granted on the principle of least privilege. All access "
        "requests must be approved by the employee's manager and the system owner. Access is "
        "reviewed quarterly; unused accounts are disabled after 90 days of inactivity.\n\n"
        "Multi-factor authentication (MFA) is required for all remote access, administrative "
        "accounts, and access to Confidential data. Shared credentials are prohibited. "
        "Passwords must be at least 12 characters with complexity requirements.",
    ),
    (
        "4. NETWORK SECURITY",
        "All data in transit must be encrypted using TLS 1.2 or higher. Data at rest classified "
        "as Confidential must be encrypted using AES-256. Employees must not connect to "
        "Company systems over unsecured public Wi-Fi without using the approved VPN.\n\n"
        "The network is segmented to isolate production environments from development and "
        "corporate networks. Firewall rules are reviewed quarterly. Penetration testing is "
        "conducted annually by an approved third-party firm.",
    ),
    (
        "5. INCIDENT RESPONSE",
        "Any suspected security incident must be reported to security@techcorp.com within "
        "one (1) hour of discovery. The Security Operations team will triage and classify "
        "the incident within four (4) hours. A post-incident review is conducted for all "
        "incidents rated Severity 1 or Severity 2.\n\n"
        "For incidents involving personal data, the Privacy team must be notified immediately "
        "to assess notification obligations under applicable laws including GDPR and CCPA.",
    ),
    (
        "6. DATA RETENTION AND DISPOSAL",
        "Data must be retained for the periods specified in the Data Retention Schedule "
        "(available on the intranet). Data no longer needed must be securely destroyed: "
        "electronic data via NIST-compliant wiping or cryptographic erasure; physical media "
        "via cross-cut shredding or certified destruction vendor.\n\n"
        "Backup data follows the same retention schedule as primary data. Backups are tested "
        "for recoverability quarterly.",
    ),
    (
        "7. THIRD-PARTY AND VENDOR SECURITY",
        "All vendors with access to Company data or systems must complete a security assessment "
        "prior to engagement. Vendors processing Confidential data must sign a Data Processing "
        "Agreement and demonstrate compliance with ISO 27001 or SOC 2 Type II.\n\n"
        "Vendor access must be scoped to the minimum necessary and terminated immediately "
        "upon contract expiration. Ongoing vendor security posture is reviewed annually.",
    ),
    (
        "8. COMPLIANCE AND ENFORCEMENT",
        "Compliance with this policy is mandatory. Violations may result in disciplinary "
        "action up to and including termination, as well as civil or criminal liability. "
        "The Security team conducts annual training for all employees.\n\n"
        "Exceptions to this policy must be submitted to the Chief Information Security "
        "Officer (CISO) for approval, documented, and reviewed annually.",
    ),
]

_POLICY_REMOTE_SECTIONS = [
    (
        "1. POLICY STATEMENT",
        "TechCorp Solutions supports flexible work arrangements that enable employees to "
        "maintain high productivity while balancing personal and professional commitments. "
        "This Remote Work Policy sets the conditions under which employees may work outside "
        "of Company offices on a regular or occasional basis.",
    ),
    (
        "2. ELIGIBILITY",
        "Employees in roles that do not require physical presence (laboratory work, on-site "
        "customer support, or roles requiring specialized on-site equipment) may apply for "
        "a remote or hybrid work arrangement with manager approval.\n\n"
        "New employees are generally expected to work on-site for the first ninety (90) days "
        "to facilitate onboarding and team integration. Remote arrangements for new employees "
        "require VP-level approval.",
    ),
    (
        "3. EQUIPMENT AND TECHNOLOGY",
        "The Company provides eligible remote employees with a laptop, headset, and software "
        "licenses necessary to perform their job functions. Employees are responsible for "
        "providing a stable internet connection with minimum 25 Mbps download speed.\n\n"
        "Company-provided equipment must be used only for Company business and is subject "
        "to monitoring in accordance with the Acceptable Use Policy. Equipment must be "
        "returned immediately upon separation or transition to an on-site role.",
    ),
    (
        "4. SECURITY REQUIREMENTS",
        "Remote employees must comply with all requirements of the Information Security Policy. "
        "Key requirements include: use of the approved VPN for all Company system access; "
        "a dedicated workspace that prevents unauthorized persons from viewing Confidential "
        "data; full-disk encryption enabled on all devices; and screen lock after 5 minutes.\n\n"
        "Remote employees must not use personal devices to access Confidential data without "
        "prior approval from the CISO.",
    ),
    (
        "5. WORK HOURS AND AVAILABILITY",
        "Remote employees are expected to maintain their agreed-upon core hours (typically "
        "10:00 AM–3:00 PM in their time zone) and be available via Slack and video conference "
        "during those hours. Response to messages is expected within two (2) business hours.\n\n"
        "Attendance at required team meetings takes precedence over other commitments. "
        "Remote arrangements do not alter compensation, benefits, or performance expectations.",
    ),
    (
        "6. EXPENSE REIMBURSEMENT",
        "The Company reimburses remote employees for reasonable and documented business "
        "expenses, including: up to $500 one-time home office setup allowance; up to $50 "
        "per month toward internet service; and approved ergonomic equipment.\n\n"
        "Expenses must be submitted through the expense management system within 30 days "
        "of incurrence with receipts. Personal expenses and home modifications are not "
        "reimbursable.",
    ),
    (
        "7. PERFORMANCE EXPECTATIONS",
        "Remote work is a privilege, not an entitlement, and may be modified or revoked "
        "if performance declines or policy is violated. Managers conduct monthly check-ins "
        "with remote direct reports to assess workload, blockers, and team connection.\n\n"
        "Remote employees are subject to the same performance standards and review processes "
        "as on-site employees. Chronic availability or performance issues may result in "
        "a requirement to return to on-site work.",
    ),
]

_HANDBOOK_EMPLOYEE_SECTIONS = [
    (
        "WELCOME TO TECHCORP SOLUTIONS",
        "We are thrilled to have you join TechCorp Solutions. This handbook is your guide to "
        "our culture, policies, benefits, and expectations. We encourage you to read it "
        "carefully and refer to it throughout your employment.\n\n"
        "Our mission is to build technology that empowers enterprises to make better decisions "
        "faster. We achieve this by attracting exceptional talent, fostering a culture of "
        "intellectual curiosity, and relentlessly focusing on customer outcomes.",
    ),
    (
        "COMPANY VALUES",
        "We operate by five core values:\n\n"
        "Customer First: Every product and process decision starts with the impact on the "
        "customer. We seek to understand customer problems deeply before building solutions.\n\n"
        "Own It: We take full responsibility for outcomes, not just activities. We follow "
        "through, close loops, and escalate early when we are blocked.\n\n"
        "Be Direct: We communicate clearly and honestly, even when it is uncomfortable. "
        "We debate ideas vigorously and commit to decisions once made.",
    ),
    (
        "EMPLOYMENT POLICIES",
        "TechCorp Solutions is an at-will employer in all states where at-will employment "
        "is permitted. This means either the employee or the Company may terminate the "
        "employment relationship at any time, with or without cause or notice, subject to "
        "applicable laws.\n\n"
        "The Company is an equal opportunity employer. We prohibit discrimination or "
        "harassment based on race, color, religion, sex, national origin, age, disability, "
        "veteran status, sexual orientation, gender identity, or any other protected class.",
    ),
    (
        "BENEFITS SUMMARY",
        "Health Insurance: Medical, dental, and vision coverage effective on your first day. "
        "The Company covers 85% of premiums for employees and 75% for eligible dependents.\n\n"
        "Retirement: 401(k) plan with immediate enrollment eligibility. The Company matches "
        "100% of contributions up to 4% of eligible compensation, vesting over 3 years.\n\n"
        "Paid Time Off: 20 vacation days, 10 sick days, and 11 company holidays per year. "
        "New hires receive the full PTO allotment immediately.",
    ),
    (
        "WORKPLACE CONDUCT",
        "We expect all employees to treat colleagues, customers, and partners with "
        "professionalism and respect. The Company does not tolerate workplace harassment "
        "of any kind. If you experience or witness harassment, report it immediately to "
        "your HR Business Partner or via the anonymous Ethics Hotline.\n\n"
        "Use of Company resources for personal benefit, political activity, or outside "
        "employment without prior approval is prohibited. Social media conduct that "
        "disparages the Company, customers, or colleagues may result in disciplinary action.",
    ),
    (
        "HEALTH AND SAFETY",
        "TechCorp Solutions is committed to providing a safe working environment. Employees "
        "must immediately report workplace injuries, hazardous conditions, or near-miss "
        "incidents to their manager and facilities@techcorp.com.\n\n"
        "The Company maintains workers' compensation insurance as required by law. "
        "Employees who suffer work-related injuries or illnesses are encouraged to report "
        "promptly to preserve their rights under applicable workers' compensation laws.",
    ),
    (
        "TECHNOLOGY USE",
        "Company-issued devices and systems are provided for business purposes. Incidental "
        "personal use is permitted provided it does not interfere with work, violate policy, "
        "or consume significant resources. Employees have no expectation of privacy on "
        "Company systems.\n\n"
        "Accessing, downloading, or sharing illegal content, including pirated software or "
        "inappropriate material, is strictly prohibited and may result in immediate "
        "termination and referral to law enforcement.",
    ),
]

_HANDBOOK_CONTRACTOR_SECTIONS = [
    (
        "INTRODUCTION",
        "Welcome to TechCorp Solutions. This handbook applies to independent contractors "
        "engaged by TechCorp Solutions under an Independent Contractor Agreement. It "
        "describes our expectations for engagement conduct, intellectual property, and "
        "compliance with our policies.\n\n"
        "As an independent contractor, you are not an employee of TechCorp Solutions. "
        "You are responsible for managing your own taxes, benefits, and insurance. "
        "You retain the right to perform services for other clients, subject to the "
        "conflict-of-interest restrictions in your Contractor Agreement.",
    ),
    (
        "CONTRACTOR CONDUCT",
        "Contractors are expected to conduct themselves professionally in all interactions "
        "with TechCorp employees, customers, and partners. Contractors must comply with "
        "TechCorp's Code of Conduct with respect to non-discrimination and non-harassment.\n\n"
        "Contractors must not make commitments on behalf of TechCorp or represent themselves "
        "as employees. Any public statements about TechCorp must be pre-approved by the "
        "Communications team.",
    ),
    (
        "INTELLECTUAL PROPERTY",
        "All work product, inventions, and deliverables created by contractors in the "
        "performance of services for TechCorp are works-for-hire owned by TechCorp. "
        "To the extent any work product does not qualify as a work-for-hire, contractors "
        "assign all right, title, and interest to TechCorp.\n\n"
        "Contractors must not use TechCorp's Confidential Information in work for other "
        "clients and must segregate TechCorp deliverables from their own intellectual "
        "property and from work performed for other clients.",
    ),
    (
        "CONFIDENTIALITY",
        "Contractors have access to TechCorp's Confidential Information as defined in their "
        "Contractor Agreement. All Confidential Information must be protected with reasonable "
        "care and not disclosed to any third party.\n\n"
        "Upon completion of the engagement, contractors must return or certify destruction "
        "of all Confidential Information and must not retain copies in any form. "
        "Confidentiality obligations survive the end of the engagement indefinitely.",
    ),
    (
        "COMPLIANCE REQUIREMENTS",
        "Contractors must comply with TechCorp's Information Security Policy with respect "
        "to data handling, device security, and network access. Contractors who are issued "
        "TechCorp credentials must not share those credentials with others.\n\n"
        "Contractors working on-site must comply with TechCorp's workplace safety policies "
        "and sign in with reception. On-site access is limited to hours agreed in the "
        "engagement scope and must not extend to areas outside the authorized workspace.",
    ),
]

_DPA_SECTIONS = [
    (
        "1. DEFINITIONS",
        "\"Controller\" means the party that determines the purposes and means of processing "
        "Personal Data. \"Processor\" means TechCorp Solutions, acting on behalf of the "
        "Controller. \"Personal Data\" has the meaning given in the GDPR and applicable "
        "data protection laws.\n\n"
        "\"Processing\" means any operation performed on Personal Data including collection, "
        "recording, storage, adaptation, retrieval, disclosure, or deletion. "
        "\"Data Subject\" means an identified or identifiable natural person whose Personal "
        "Data is processed.",
    ),
    (
        "2. SUBJECT MATTER AND DURATION",
        "This Data Processing Agreement (\"DPA\") forms part of the Master Services Agreement "
        "between the Controller and the Processor and sets out the terms governing the "
        "processing of Personal Data by the Processor on behalf of the Controller.\n\n"
        "The DPA remains in effect for the duration of the MSA and until all Personal Data "
        "is returned or deleted in accordance with the deletion obligations below.",
    ),
    (
        "3. NATURE AND PURPOSE OF PROCESSING",
        "Processor processes Personal Data solely to provide the Services described in the MSA "
        "and any applicable Statements of Work. Categories of Personal Data may include: "
        "employee data (name, contact information, HR records); customer data (name, email, "
        "usage data); and end-user behavioral data.\n\n"
        "Processing is carried out in the United States and, where required, in the European "
        "Economic Area under appropriate transfer mechanisms.",
    ),
    (
        "4. OBLIGATIONS OF THE PROCESSOR",
        "Processor shall: (a) process Personal Data only on documented instructions from the "
        "Controller; (b) ensure persons authorized to process Personal Data are bound by "
        "confidentiality; (c) implement appropriate technical and organizational security "
        "measures; (d) assist the Controller with data subject rights requests within "
        "thirty (30) days; (e) notify the Controller of any security breach without undue "
        "delay and in any event within seventy-two (72) hours.",
    ),
    (
        "5. TECHNICAL AND ORGANIZATIONAL SECURITY MEASURES",
        "Processor maintains the following security measures:\n\n"
        "Encryption: Personal Data is encrypted in transit (TLS 1.2+) and at rest (AES-256). "
        "Access Control: Role-based access with MFA for all administrative access. "
        "Audit Logging: All access to Personal Data systems is logged and retained for 12 months. "
        "Penetration Testing: Annual third-party penetration testing of production systems. "
        "Incident Response: 24/7 security monitoring with 4-hour response SLA for critical incidents.",
    ),
    (
        "6. SUBPROCESSORS",
        "Controller authorizes Processor to engage the subprocessors listed in Annex I to "
        "this DPA. Processor shall notify Controller at least thirty (30) days in advance "
        "of any intended changes concerning the addition or replacement of subprocessors.\n\n"
        "Processor shall impose data protection obligations on subprocessors equivalent to "
        "those in this DPA. Processor remains liable for acts and omissions of subprocessors "
        "to the same extent as for its own acts and omissions.",
    ),
    (
        "7. DATA SUBJECT RIGHTS",
        "Upon receiving a request from a data subject to access, rectify, erase, restrict, "
        "or port their Personal Data, Processor shall: (a) forward the request to Controller "
        "within five (5) business days; (b) not respond directly to the data subject without "
        "Controller authorization; and (c) provide reasonable assistance to enable Controller "
        "to fulfill the request within the statutory timeframe.",
    ),
    (
        "8. PERSONAL DATA BREACH NOTIFICATION",
        "Upon becoming aware of a Personal Data breach, Processor shall notify Controller "
        "without undue delay and within seventy-two (72) hours. The notification shall "
        "include: the nature of the breach; categories and approximate number of data "
        "subjects and records affected; contact details of the Data Protection Officer; "
        "likely consequences; and measures taken or proposed to address the breach.",
    ),
    (
        "9. DELETION AND RETURN OF DATA",
        "Upon termination of the MSA or written request by the Controller, Processor shall "
        "delete or return all Personal Data within thirty (30) days, at the Controller's "
        "election. Processor shall certify deletion in writing. Processor may retain "
        "Personal Data for the period required by applicable law, subject to ongoing "
        "confidentiality and security obligations.",
    ),
]


# ── Document specifications ────────────────────────────────────────────────────

_DOCUMENTS: list[dict] = [
    # MSAs
    {
        "filename": "msa_acme_2024.pdf",
        "title": "MASTER SERVICES AGREEMENT",
        "subtitle": "TechCorp Solutions, Inc. and Acme Corporation — Effective January 1, 2024",
        "sections": _MSA_SECTIONS,
    },
    {
        "filename": "msa_globex_2024.pdf",
        "title": "MASTER SERVICES AGREEMENT",
        "subtitle": "TechCorp Solutions, Inc. and Globex Industries, Inc. — Effective March 15, 2024",
        "sections": _MSA_SECTIONS,
    },
    {
        "filename": "msa_initech_2024.pdf",
        "title": "MASTER SERVICES AGREEMENT",
        "subtitle": "TechCorp Solutions, Inc. and Initech LLC — Effective June 1, 2024",
        "sections": _MSA_SECTIONS,
    },
    # NDAs
    {
        "filename": "nda_acme_2024.pdf",
        "title": "NON-DISCLOSURE AGREEMENT",
        "subtitle": "TechCorp Solutions, Inc. and Acme Corporation — Effective February 1, 2024",
        "sections": _NDA_SECTIONS,
    },
    {
        "filename": "nda_globex_2024.pdf",
        "title": "NON-DISCLOSURE AGREEMENT",
        "subtitle": "TechCorp Solutions, Inc. and Globex Industries, Inc. — Effective April 10, 2024",
        "sections": _NDA_SECTIONS,
    },
    # Policies
    {
        "filename": "policy_hr_2024.pdf",
        "title": "HUMAN RESOURCES POLICY",
        "subtitle": "TechCorp Solutions, Inc. — Version 3.2, Effective January 1, 2024",
        "sections": _POLICY_HR_SECTIONS,
    },
    {
        "filename": "policy_security_2024.pdf",
        "title": "INFORMATION SECURITY POLICY",
        "subtitle": "TechCorp Solutions, Inc. — Version 2.5, Effective January 1, 2024",
        "sections": _POLICY_SECURITY_SECTIONS,
    },
    {
        "filename": "policy_remote_work_2024.pdf",
        "title": "REMOTE WORK POLICY",
        "subtitle": "TechCorp Solutions, Inc. — Version 1.4, Effective September 1, 2024",
        "sections": _POLICY_REMOTE_SECTIONS,
    },
    # Handbooks
    {
        "filename": "handbook_employee_2024.pdf",
        "title": "EMPLOYEE HANDBOOK",
        "subtitle": "TechCorp Solutions, Inc. — 2024 Edition",
        "sections": _HANDBOOK_EMPLOYEE_SECTIONS,
    },
    {
        "filename": "handbook_contractor_2024.pdf",
        "title": "CONTRACTOR HANDBOOK",
        "subtitle": "TechCorp Solutions, Inc. — 2024 Edition",
        "sections": _HANDBOOK_CONTRACTOR_SECTIONS,
    },
    # DPAs
    {
        "filename": "dpa_acme_2024.pdf",
        "title": "DATA PROCESSING AGREEMENT",
        "subtitle": "TechCorp Solutions, Inc. (Processor) and Acme Corporation (Controller) — 2024",
        "sections": _DPA_SECTIONS,
    },
    {
        "filename": "dpa_globex_2024.pdf",
        "title": "DATA PROCESSING AGREEMENT",
        "subtitle": "TechCorp Solutions, Inc. (Processor) and Globex Industries, Inc. (Controller) — 2024",
        "sections": _DPA_SECTIONS,
    },
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    for spec in _DOCUMENTS:
        path = output_dir / spec["filename"]
        _make_pdf(
            path,
            title=spec["title"],
            subtitle=spec["subtitle"],
            sections=spec["sections"],
        )
        log.info(
            "corpus_pdf_generated",
            filename=spec["filename"],
            path=str(path),
        )
        generated += 1

    log.info("corpus_synthesis_complete", generated=generated, output_dir=str(output_dir))


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
    main(out)
