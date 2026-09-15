import pytest
import csv
import os
import inspect
from response_cache import cached_query_rag
from agent import query_rag
from langchain_google_genai import ChatGoogleGenerativeAI

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
Determine if the Actual Response is semantically equivalent to the Expected Response — 
that is, does it convey the same core meaning, facts, and intent, even if the wording differs?

For out-of-scope or refusal-type expected responses, the actual response should count as a 
match if it also declines to answer or indicates the information is not available, even if 
phrased differently.

If the actual response contains additional information not in the expected response, but that 
information doesn't contradict the expected response, still answer 'true'.
If the actual response contains information that contradicts the expected response, answer 'false'.
So for example, if the expected response is 'no' but the actual response contains a lengthy explanation
for why the answer is no, that should still be marked as true since the overall semantic meaning is the same.

(Answer with only 'true' or 'false') 
"""

def query_and_validate(question: str, expected_response: str, category: str):
    response_text = cached_query_rag(question, query_rag, force_refresh=False)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.content.strip().lower()

    print(prompt)

    grader_verdict = False
    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        grader_verdict = True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        grader_verdict = False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
        
    # --- CSV LOGGING LOGIC ---
    csv_file = "test_results_log.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Category', 'Question', 'Expected Response', 'Actual Response', 'Grader Verdict', 'Manual Judgment'])
        
        writer.writerow([category, question, expected_response, response_text, str(grader_verdict).lower(), ""])

    return grader_verdict
    
    

#Category breakdown (50 total):
#  1. Prerequisite Relationships           6
#  2. Major Admission/Progression          5
#  3. Negative/Exclusion Facts             4
#  4. Course Metadata                      6
#  5. University-Wide / Gen Ed             5
#  6. Double Dawgs / Graduate Programs     10
#  7. Out-of-Scope / Refusal               3
#  8. Major Identity (AB vs BS, ABET)      2
#  9. Multi-Hop Reasoning                  4
#  10. Cross-Program Comparison            2
#  11. Course Content/Description          3
 

# ======================================================================
# 1. PREREQUISITE RELATIONSHIPS (6)
# ======================================================================
 
@pytest.mark.prerequisites
def test_csci2720_prerequisite():
    assert query_and_validate(
        question="Is CSCI 2670 a prerequisite of CSCI 2720?",
        expected_response="No",
        category="prerequisites"
    )
 
@pytest.mark.prerequisites
def test_csci2670_prerequisite():
    assert query_and_validate(
        question="Is CSCI 2610 a prerequisite of CSCI 2670?",
        expected_response="Yes",
        category="prerequisites"
    )
 
@pytest.mark.prerequisites
def test_csci4370_prerequisite():
    assert query_and_validate(
        question="What is the prerequisite for CSCI 4370 (Database Management)?",
        expected_response="CSCI 2720 or CSCI 2725",
        category="prerequisites"
    )
 
@pytest.mark.prerequisites
def test_csci4470_prerequisite():
    assert query_and_validate(
        question="What are the prerequisites for CSCI 4470 (Algorithms)?",
        expected_response="(CSCI 2720 or CSCI 2725) and CSCI 2670",
        category="prerequisites"
    )
 
@pytest.mark.prerequisites
def test_csci4720_prerequisite():
    assert query_and_validate(
        question="What are the prerequisites for CSCI 4720 (Computer Architecture and Organization)?",
        expected_response="(CSCI 2670 or CSEE 2220) and CSCI 1730",
        category="prerequisites"
    )
 
@pytest.mark.prerequisites
def test_csci1730_is_prereq_for_csci4720():
    assert query_and_validate(
        question="Is CSCI 1730 a prerequisite for CSCI 4720?",
        expected_response="Yes",
        category="prerequisites"
    )
 
 
# ======================================================================
# 2. MAJOR ADMISSION / PROGRESSION (5)
# ======================================================================
 
@pytest.mark.admission
def test_cs_intended_major_requirements():
    assert query_and_validate(
        question="What computer science courses must an intended CS major at UGA complete before they can be fully admitted into the major and thus no longer be 'intended'?",
        expected_response="CSCI 1301, CSCI 1302, CSCI 2610, and CSCI 2720",
        category="admission"
    )
 
@pytest.mark.admission
def test_cs_sequence():
    assert query_and_validate(
        question="If I've taken CSCI 1301 and 1302, what are the most logical CS courses for me to take next?",
        expected_response="CSCI 1730 or CSCI 2610",
        category="admission"
    )
 
@pytest.mark.admission
def test_intended_major_grade_requirement():
    assert query_and_validate(
        question="What grade must intended CS majors earn in CSCI 1301, CSCI 1302, CSCI 2610, and CSCI 2720 to be admitted to the major?",
        expected_response="A grade of C (2.0) or higher",
        category="admission"
    )
 
@pytest.mark.admission
def test_area_vi_math_requirement():
    assert query_and_validate(
        question="Which math course is part of the Area VI entrance requirements for the CS major?",
        expected_response="MATH 2250 (Calculus I)",
        category="admission"
    )
 
@pytest.mark.admission
def test_major_electives_hours():
    assert query_and_validate(
        question="How many hours of CSCI 4000-level Major Elective courses must BS-CS students complete?",
        expected_response="12 hours",
        category="admission"
    )
 
 
# ======================================================================
# 3. NEGATIVE / EXCLUSION FACTS (4)
# ======================================================================
 
@pytest.mark.exclusion
def test_diffeq_is_required():
    assert query_and_validate(
        question="Are cs majors at uga required to take differential equations?",
        expected_response="No",
        category="exclusion"
    )
 
@pytest.mark.exclusion
def test_csci1300_not_major_requirement():
    assert query_and_validate(
        question="Does CSCI 1300 (Introduction to Programming with Python) satisfy any CSCI major requirements?",
        expected_response="No, CSCI 1300 is an elective course and does not satisfy any CSCI major requirements",
        category="exclusion"
    )
 
@pytest.mark.exclusion
def test_no_ba_in_cs():
    assert query_and_validate(
        question="Does UGA offer a Bachelor of Arts (B.A. or A.B.) in Computer Science?",
        expected_response="No, UGA only offers a Bachelor of Science (BS) in Computer Science, not a B.A. or A.B.",
        category="exclusion"
    )
 
@pytest.mark.exclusion
def test_grade_d_not_acceptable():
    assert query_and_validate(
        question="Is a grade of D acceptable for the prerequisite courses required for the CS major?",
        expected_response="No, a grade of C (2.0) or higher is required",
        category="exclusion"
    )
 
 
# ======================================================================
# 4. COURSE METADATA (6)
# ======================================================================
 
@pytest.mark.metadata
def test_csci1730_credit_hours():
    assert query_and_validate(
        question="How many credit hours is CSCI 1730 Systems Programming?",
        expected_response="4",
        category="metadata"
    )
 
@pytest.mark.metadata
def test_csci4300_credit_hours():
    assert query_and_validate(
        question="How many credit hours is CSCI 4300 (Web Programming)?",
        expected_response="4",
        category="metadata"
    )
 
@pytest.mark.metadata
def test_csci4300_prerequisite():
    assert query_and_validate(
        question="What is the prerequisite for CSCI 4300 (Web Programming)?",
        expected_response="CSCI 1302",
        category="metadata"
    )
 
@pytest.mark.metadata
def test_csci3030_semester_offered():
    assert query_and_validate(
        question="In which semester is CSCI 3030 (Computing, Ethics, and Society) typically offered?",
        expected_response="Spring",
        category="metadata"
    )
 
@pytest.mark.metadata
def test_csci4370_semesters_offered():
    assert query_and_validate(
        question="In which semesters is CSCI 4370 (Database Management) offered?",
        expected_response="Fall and Spring",
        category="metadata"
    )
 
@pytest.mark.metadata
def test_csci4720_semesters_offered():
    assert query_and_validate(
        question="Is CSCI 4720 (Computer Architecture) offered every semester or only occasionally?",
        expected_response="It is offered every year, in fall, spring, and summer semesters",
        category="metadata"
    )
 
 
# ======================================================================
# 5. UNIVERSITY-WIDE / GEN ED (5)
# ======================================================================
 
@pytest.mark.gen_ed
def test_pe_required():
    assert query_and_validate(
        question="Are UGA CS students required to take a Physical Education course?",
        expected_response="Yes",
        category="gen_ed"
    )
 
@pytest.mark.gen_ed
def test_total_major_hours():
    assert query_and_validate(
        question="How many total hours must a UGA CS student complete to graduate with a BS?",
        expected_response="120 hours, excluding the 1 hour PE requirement",
        category="gen_ed"
    )
 
@pytest.mark.gen_ed
def test_upper_division_hours():
    assert query_and_validate(
        question="How many hours of Franklin College upper-division coursework must BS-CS students complete?",
        expected_response="39 hours",
        category="gen_ed"
    )
 
@pytest.mark.gen_ed
def test_foreign_language_requirement():
    assert query_and_validate(
        question="Do BS-CS students need to complete a foreign language requirement as part of Franklin College's requirements?",
        expected_response="Yes",
        category="gen_ed"
    )
 
@pytest.mark.gen_ed
def test_first_english_composition_course():
    assert query_and_validate(
        question="What is the first English Composition course required for the BS-CS Foundation requirements?",
        expected_response="ENGL 1101 (English Composition I)",
        category="gen_ed"
    )
 
 
# ======================================================================
# 6. DOUBLE DAWGS / GRADUATE PROGRAMS (10)
# Covers all 4 Double Dawgs program URLs, not just the AI one.
# ======================================================================
 
@pytest.mark.double_dawgs
def test_ai_masters_thesis_course():
    assert query_and_validate(
        question="What course must I register for as part of my thesis in the Double Dawgs Artificial Intelligence MS program?",
        expected_response="ARTI 7300 Master's Thesis",
        category="double_dawgs"
    )
 
@pytest.mark.double_dawgs
def test_csms_doubledawgs_gpa_requirement():
    assert query_and_validate(
        question="What GPA is required for admission to the CS BS/Computer Science MS (non-thesis) Double Dawgs pathway?",
        expected_response="A cumulative GPA of 3.0 or higher",
        category="double_dawgs"
    )
 
@pytest.mark.double_dawgs
def test_ai_doubledawgs_admission_requirements():
    assert query_and_validate(
        question="What GPA and credit hour requirements apply to the CS BS/Artificial Intelligence MS Double Dawgs pathway?",
        expected_response="At least 60 hours completed, with at least 40 completed at UGA, and an overall undergraduate GPA of 3.5",
        category="double_dawgs"
    )
 
@pytest.mark.double_dawgs
def test_cybersecurity_doubledawgs_required_courses():
    assert query_and_validate(
        question="Which courses must be completed with a C or better to be admitted to the CS BS/Cybersecurity and Privacy MS Double Dawgs pathway?",
        expected_response="CSCI 1302 Software Development, CSCI 2670 Introduction to Theory of Computing, CSCI 2720 Data Structures, and MATH 2250 Calculus I",
        category="double_dawgs"
    )
 
@pytest.mark.double_dawgs
def test_emerging_media_doubledawgs_deadline():
    assert query_and_validate(
        question="How does a student apply to the CS BS/Journalism and Mass Communication MA (Emerging Media) Double Dawgs program, and when is the deadline?",
        expected_response="They apply through the New Media Institute; applications open on the first day of fall classes and are due by October 31 annually",
        category="double_dawgs"
    )
 
@pytest.mark.double_dawgs
def test_csms_doubledawgs_fifth_year_hours():
    assert query_and_validate(
        question="In the fifth year of the CS BS/Computer Science MS (non-thesis) Double Dawgs program, how many credit hours of 6000/8000-level CSCI coursework are required, and how many must be at the 8000 level?",
        expected_response="16 credit hours total, with at least 8 credit hours at the 8000 level",
        category="double_dawgs"
    )
 
@pytest.mark.double_dawgs
def test_nmix_courses_program():
    assert query_and_validate(
        question="Which Double Dawgs program allows students to take NMIX-prefixed courses?",
        expected_response="The CS BS/Journalism and Mass Communication MA (Emerging Media) Double Dawgs program",
        category="double_dawgs"
    )
 
@pytest.mark.double_dawgs
def test_cybersecurity_doubledawgs_project_course():
    assert query_and_validate(
        question="Which course must be registered for as a two-semester sequence in the CS BS/Cybersecurity and Privacy MS Double Dawgs program?",
        expected_response="CSCI 7200 Master's Project, for 3-4 credit hours spread over two semesters",
        category="double_dawgs"
    )
 
@pytest.mark.double_dawgs
def test_csms_doubledawgs_grad_only_hours():
    assert query_and_validate(
        question="How many hours of graduate-student-only coursework must be indicated on the Program of Study for the CS BS/Computer Science MS (non-thesis) Double Dawgs program?",
        expected_response="12 credit hours",
        category="double_dawgs"
    )
 
@pytest.mark.double_dawgs
def test_systems_design_options():
    assert query_and_validate(
        question="Under the BS-CS Systems Design requirement, which three courses can a student choose from?",
        expected_response="CSCI 4570 (Compilers), CSCI 4730 (Operating Systems), and CSCI 4760 (Computer Networks)",
        category="double_dawgs"
    )
 
 
# ======================================================================
# 7. OUT-OF-SCOPE / REFUSAL (3)
# ======================================================================
 
@pytest.mark.out_of_scope
def test_out_of_scope():
    assert query_and_validate(
        question="Tell me about the CS program at Georgia Tech.",
        expected_response="I don't have information about Georgia Tech's CS program. My knowledge is limited to UGA's Computer Science program.",
        category="out_of_scope"
    )
 
@pytest.mark.out_of_scope
def test_out_of_scope_financial_aid():
    assert query_and_validate(
        question="What is the deadline to apply for financial aid at UGA?",
        expected_response="I don't have information about financial aid deadlines. My knowledge is limited to UGA's Computer Science program curriculum and requirements.",
        category="out_of_scope"
    )
 
@pytest.mark.out_of_scope
def test_out_of_scope_weather():
    assert query_and_validate(
        question="What's the weather like in Athens, GA today?",
        expected_response="I don't have access to current weather information. My knowledge is limited to UGA's Computer Science program.",
        category="out_of_scope"
    )
 
 
# ======================================================================
# 8. MAJOR IDENTITY (2)
# ======================================================================
 
@pytest.mark.major_identity
def test_ai_area_of_emphasis_availability():
    assert query_and_validate(
        question="Does UGA offer an Artificial Intelligence Area of Emphasis for CS majors, and when does it become available?",
        expected_response="Yes, starting Fall 2026",
        category="major_identity"
    )
 
@pytest.mark.major_identity
def test_abet_accreditation():
    assert query_and_validate(
        question="Is the BS-CS program at UGA ABET accredited?",
        expected_response="Yes, it is accredited by the CAC Commission of ABET",
        category="major_identity"
    )
 
 
# ======================================================================
# 9. MULTI-HOP REASONING (4)
# ======================================================================
 
@pytest.mark.multihop
def test_multihop_csms_doubledawgs_math_course():
    assert query_and_validate(
        question="Besides CSCI 1302, CSCI 2670, and CSCI 2720, what math course must a student complete with a C or better to apply to the CS BS/Computer Science MS (non-thesis) Double Dawgs pathway?",
        expected_response="MATH 2250 (Calculus I)",
        category="multihop"
    )
 
@pytest.mark.multihop
def test_multihop_partial_completion_eligibility():
    assert query_and_validate(
        question="A student has completed CSCI 1302, CSCI 2670, and MATH 2250 with a C or better, but has not completed or enrolled in CSCI 2720. Can they still apply to the CS BS/Computer Science MS (non-thesis) Double Dawgs pathway?",
        expected_response="Not automatically. Admission would only be considered on a case-by-case basis by a faculty admissions committee, since they have not completed or are not currently enrolled in CSCI 2720",
        category="multihop"
    )
 
@pytest.mark.multihop
def test_multihop_shared_prerequisite():
    assert query_and_validate(
        question="Which single course is a prerequisite for both CSCI 4470 (Algorithms) and CSCI 4720 (Computer Architecture)?",
        expected_response="CSCI 2670",
        category="multihop"
    )
 
@pytest.mark.multihop
def test_multihop_dual_requirement_course():
    assert query_and_validate(
        question="Which single course can satisfy both the Application Design requirement and the Teamwork requirement in the BS-CS major?",
        expected_response="CSCI 4050 (Software Engineering)",
        category="multihop"
    )
 
 
# ======================================================================
# 10. CROSS-PROGRAM COMPARISON (2)
# ======================================================================
 
@pytest.mark.comparison
def test_compare_csms_vs_ai_doubledawgs_admission():
    assert query_and_validate(
        question="What is the difference in GPA and credit-hour admission requirements between the CS BS/Computer Science MS (non-thesis) pathway and the CS BS/Artificial Intelligence MS pathway?",
        expected_response="The CS/CS MS pathway requires 60-90 hours and a 3.0 GPA. The CS/AI MS pathway requires at least 60 hours (40 at UGA) and a higher 3.5 GPA.",
        category="comparison"
    )
 
@pytest.mark.comparison
def test_which_doubledawgs_leads_to_ma():
    assert query_and_validate(
        question="Which Double Dawgs CS pathway leads to a Master of Arts rather than a Master of Science?",
        expected_response="The CS BS/Journalism and Mass Communication MA (Emerging Media) pathway",
        category="comparison"
    )
 
 
# ======================================================================
# 11. COURSE CONTENT / DESCRIPTION (3)
# ======================================================================
 
@pytest.mark.course_content
def test_csci4470_topics():
    assert query_and_validate(
        question="What topics are covered in CSCI 4470 (Algorithms)?",
        expected_response="Basic analysis techniques, basic design techniques (divide-and-conquer, dynamic programming, greedy), basic and advanced graph algorithms, and NP-completeness theory",
        category="course_content"
    )
 
@pytest.mark.course_content
def test_csci4370_topics():
    assert query_and_validate(
        question="What topics are covered in CSCI 4370 (Database Management)?",
        expected_response="Efficient file access techniques, the relational data model and other data models, query languages, database design using entity-relationship diagrams and normalization theory, query optimization, and transaction processing",
        category="course_content"
    )
 
@pytest.mark.course_content
def test_csci1300_core_curriculum_credit():
    assert query_and_validate(
        question="Does CSCI 1300 (Introduction to Programming with Python) satisfy any UGA core curriculum requirement, even though it doesn't count for the CS major?",
        expected_response="Yes, it satisfies Area III Quantitative Reasoning in the General Education Core Curriculum",
        category="course_content"
    )