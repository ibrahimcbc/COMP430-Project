import pandas as pd
import os

# Create directories
os.makedirs("data/hierarchies/adult", exist_ok=True)
os.makedirs("data/hierarchies/bank", exist_ok=True)
os.makedirs("data/hierarchies/german", exist_ok=True)

# =============================================================================
# ADULT HIERARCHIES
# =============================================================================

# age (numeric - need to create row for each value)
adult = pd.read_csv("data/clean/adult_clean.csv")
age_values = sorted(adult["age"].unique())

age_rows = []
for age in age_values:
    # Level 1: 5-year bands
    if age <= 19:
        l1 = "17-19"
    elif age <= 24:
        l1 = "20-24"
    elif age <= 29:
        l1 = "25-29"
    elif age <= 34:
        l1 = "30-34"
    elif age <= 39:
        l1 = "35-39"
    elif age <= 44:
        l1 = "40-44"
    elif age <= 49:
        l1 = "45-49"
    elif age <= 54:
        l1 = "50-54"
    elif age <= 59:
        l1 = "55-59"
    elif age <= 64:
        l1 = "60-64"
    elif age <= 69:
        l1 = "65-69"
    elif age <= 74:
        l1 = "70-74"
    elif age <= 79:
        l1 = "75-79"
    else:
        l1 = "80-90"

    # Level 2: 10-year bands
    if age <= 19:
        l2 = "17-19"
    elif age <= 29:
        l2 = "20-29"
    elif age <= 39:
        l2 = "30-39"
    elif age <= 49:
        l2 = "40-49"
    elif age <= 59:
        l2 = "50-59"
    elif age <= 69:
        l2 = "60-69"
    elif age <= 79:
        l2 = "70-79"
    else:
        l2 = "80-90"

    # Level 3: Life stage
    if age <= 29:
        l3 = "Young"
    elif age <= 59:
        l3 = "Adult"
    else:
        l3 = "Senior"

    age_rows.append([str(age), l1, l2, l3, "*"])

pd.DataFrame(age_rows).to_csv("data/hierarchies/adult/age.csv", index=False, header=False)

# workclass
workclass_hierarchy = [
    ["Private", "Private", "Working", "*"],
    ["Self-emp-not-inc", "Self-Employed", "Working", "*"],
    ["Self-emp-inc", "Self-Employed", "Working", "*"],
    ["Federal-gov", "Government", "Working", "*"],
    ["Local-gov", "Government", "Working", "*"],
    ["State-gov", "Government", "Working", "*"],
    ["Without-pay", "Not-Working", "Not-Working", "*"],
    ["Never-worked", "Not-Working", "Not-Working", "*"],
    ["Unknown", "Unknown", "Unknown", "*"],
]
pd.DataFrame(workclass_hierarchy).to_csv("data/hierarchies/adult/workclass.csv", index=False, header=False)

# education
education_hierarchy = [
    ["Preschool", "No-HS-Diploma", "No-Degree", "*"],
    ["1st-4th", "No-HS-Diploma", "No-Degree", "*"],
    ["5th-6th", "No-HS-Diploma", "No-Degree", "*"],
    ["7th-8th", "No-HS-Diploma", "No-Degree", "*"],
    ["9th", "No-HS-Diploma", "No-Degree", "*"],
    ["10th", "No-HS-Diploma", "No-Degree", "*"],
    ["11th", "No-HS-Diploma", "No-Degree", "*"],
    ["12th", "No-HS-Diploma", "No-Degree", "*"],
    ["HS-grad", "HS-Diploma", "No-Degree", "*"],
    ["Some-college", "HS-Diploma", "No-Degree", "*"],
    ["Assoc-voc", "HS-Diploma", "No-Degree", "*"],
    ["Assoc-acdm", "HS-Diploma", "No-Degree", "*"],
    ["Bachelors", "College-Degree", "Degree", "*"],
    ["Masters", "Graduate-Degree", "Degree", "*"],
    ["Prof-school", "Graduate-Degree", "Degree", "*"],
    ["Doctorate", "Graduate-Degree", "Degree", "*"],
]
pd.DataFrame(education_hierarchy).to_csv("data/hierarchies/adult/education.csv", index=False, header=False)

# marital-status
marital_hierarchy = [
    ["Married-civ-spouse", "Married", "Married", "*"],
    ["Married-spouse-absent", "Married", "Married", "*"],
    ["Married-AF-spouse", "Married", "Married", "*"],
    ["Divorced", "Previously-Married", "Not-Married", "*"],
    ["Separated", "Previously-Married", "Not-Married", "*"],
    ["Widowed", "Previously-Married", "Not-Married", "*"],
    ["Never-married", "Never-Married", "Not-Married", "*"],
]
pd.DataFrame(marital_hierarchy).to_csv("data/hierarchies/adult/marital-status.csv", index=False, header=False)

# occupation
occupation_hierarchy = [
    ["Prof-specialty", "White-Collar", "*"],
    ["Exec-managerial", "White-Collar", "*"],
    ["Tech-support", "White-Collar", "*"],
    ["Adm-clerical", "White-Collar", "*"],
    ["Sales", "White-Collar", "*"],
    ["Craft-repair", "Blue-Collar", "*"],
    ["Machine-op-inspct", "Blue-Collar", "*"],
    ["Transport-moving", "Blue-Collar", "*"],
    ["Handlers-cleaners", "Blue-Collar", "*"],
    ["Farming-fishing", "Blue-Collar", "*"],
    ["Protective-serv", "Service", "*"],
    ["Priv-house-serv", "Service", "*"],
    ["Other-service", "Service", "*"],
    ["Armed-Forces", "Other", "*"],
    ["Unknown", "Unknown", "*"],
]
pd.DataFrame(occupation_hierarchy).to_csv("data/hierarchies/adult/occupation.csv", index=False, header=False)

# race
race_hierarchy = [
    ["White", "White", "*"],
    ["Black", "Non-White", "*"],
    ["Asian-Pac-Islander", "Non-White", "*"],
    ["Amer-Indian-Eskimo", "Non-White", "*"],
    ["Other", "Non-White", "*"],
]
pd.DataFrame(race_hierarchy).to_csv("data/hierarchies/adult/race.csv", index=False, header=False)

# sex
sex_hierarchy = [
    ["Male", "*"],
    ["Female", "*"],
]
pd.DataFrame(sex_hierarchy).to_csv("data/hierarchies/adult/sex.csv", index=False, header=False)

# native-country
country_hierarchy = [
    ["United-States", "United-States", "United-States", "*"],
    ["Outlying-US(Guam-USVI-etc)", "United-States", "United-States", "*"],
    ["Canada", "Americas", "Foreign", "*"],
    ["Mexico", "Americas", "Foreign", "*"],
    ["Cuba", "Americas", "Foreign", "*"],
    ["Jamaica", "Americas", "Foreign", "*"],
    ["Dominican-Republic", "Americas", "Foreign", "*"],
    ["Haiti", "Americas", "Foreign", "*"],
    ["Trinadad&Tobago", "Americas", "Foreign", "*"],
    ["Puerto-Rico", "Americas", "Foreign", "*"],
    ["El-Salvador", "Americas", "Foreign", "*"],
    ["Guatemala", "Americas", "Foreign", "*"],
    ["Honduras", "Americas", "Foreign", "*"],
    ["Nicaragua", "Americas", "Foreign", "*"],
    ["Columbia", "Americas", "Foreign", "*"],
    ["Ecuador", "Americas", "Foreign", "*"],
    ["Peru", "Americas", "Foreign", "*"],
    ["England", "Europe", "Foreign", "*"],
    ["Germany", "Europe", "Foreign", "*"],
    ["Italy", "Europe", "Foreign", "*"],
    ["Poland", "Europe", "Foreign", "*"],
    ["Portugal", "Europe", "Foreign", "*"],
    ["Greece", "Europe", "Foreign", "*"],
    ["France", "Europe", "Foreign", "*"],
    ["Ireland", "Europe", "Foreign", "*"],
    ["Scotland", "Europe", "Foreign", "*"],
    ["Hungary", "Europe", "Foreign", "*"],
    ["Holand-Netherlands", "Europe", "Foreign", "*"],
    ["Yugoslavia", "Europe", "Foreign", "*"],
    ["China", "Asia", "Foreign", "*"],
    ["Japan", "Asia", "Foreign", "*"],
    ["India", "Asia", "Foreign", "*"],
    ["Taiwan", "Asia", "Foreign", "*"],
    ["Hong", "Asia", "Foreign", "*"],
    ["Vietnam", "Asia", "Foreign", "*"],
    ["Philippines", "Asia", "Foreign", "*"],
    ["Thailand", "Asia", "Foreign", "*"],
    ["Cambodia", "Asia", "Foreign", "*"],
    ["Laos", "Asia", "Foreign", "*"],
    ["South", "Asia", "Foreign", "*"],
    ["Iran", "Asia", "Foreign", "*"],
    ["Unknown", "Unknown", "Unknown", "*"],
]
pd.DataFrame(country_hierarchy).to_csv("data/hierarchies/adult/native-country.csv", index=False, header=False)

# relationship (private)
relationship_hierarchy = [
    ["Husband", "Spouse", "*"],
    ["Wife", "Spouse", "*"],
    ["Own-child", "Family", "*"],
    ["Other-relative", "Family", "*"],
    ["Not-in-family", "Non-Family", "*"],
    ["Unmarried", "Non-Family", "*"],
]
pd.DataFrame(relationship_hierarchy).to_csv("data/hierarchies/adult/relationship.csv", index=False, header=False)

# capital-gain (private, numeric)
cg_values = sorted(adult["capital-gain"].unique())
cg_rows = []
for val in cg_values:
    if val == 0:
        l1 = "None"
        l2 = "None"
    elif val <= 5000:
        l1 = "Low"
        l2 = "Some"
    elif val <= 20000:
        l1 = "Medium"
        l2 = "Some"
    else:
        l1 = "High"
        l2 = "Some"
    cg_rows.append([str(val), l1, l2, "*"])
pd.DataFrame(cg_rows).to_csv("data/hierarchies/adult/capital-gain.csv", index=False, header=False)

# capital-loss (private, numeric)
cl_values = sorted(adult["capital-loss"].unique())
cl_rows = []
for val in cl_values:
    if val == 0:
        l1 = "None"
        l2 = "None"
    elif val <= 1500:
        l1 = "Low"
        l2 = "Some"
    elif val <= 2500:
        l1 = "Medium"
        l2 = "Some"
    else:
        l1 = "High"
        l2 = "Some"
    cl_rows.append([str(val), l1, l2, "*"])
pd.DataFrame(cl_rows).to_csv("data/hierarchies/adult/capital-loss.csv", index=False, header=False)

# hours-per-week (private, numeric)
hpw_values = sorted(adult["hours-per-week"].unique())
hpw_rows = []
for val in hpw_values:
    if val <= 19:
        l1 = "Few"
        l2 = "Part-Time"
    elif val <= 34:
        l1 = "Part-Time"
        l2 = "Part-Time"
    elif val <= 45:
        l1 = "Full-Time"
        l2 = "Full-Time-Plus"
    elif val <= 60:
        l1 = "Overtime"
        l2 = "Full-Time-Plus"
    else:
        l1 = "Extreme"
        l2 = "Full-Time-Plus"
    hpw_rows.append([str(val), l1, l2, "*"])
pd.DataFrame(hpw_rows).to_csv("data/hierarchies/adult/hours-per-week.csv", index=False, header=False)

print("Adult hierarchies created: 12 files")

# =============================================================================
# BANK HIERARCHIES
# =============================================================================

bank = pd.read_csv("data/clean/bank_clean.csv")

# age
bank_age_values = sorted(bank["age"].unique())
bank_age_rows = []
for age in bank_age_values:
    # Level 1: 5-year bands
    if age <= 19:
        l1 = "18-19"
    elif age <= 24:
        l1 = "20-24"
    elif age <= 29:
        l1 = "25-29"
    elif age <= 34:
        l1 = "30-34"
    elif age <= 39:
        l1 = "35-39"
    elif age <= 44:
        l1 = "40-44"
    elif age <= 49:
        l1 = "45-49"
    elif age <= 54:
        l1 = "50-54"
    elif age <= 59:
        l1 = "55-59"
    elif age <= 64:
        l1 = "60-64"
    elif age <= 69:
        l1 = "65-69"
    elif age <= 74:
        l1 = "70-74"
    elif age <= 79:
        l1 = "75-79"
    else:
        l1 = "80-95"

    # Level 2: 10-year bands
    if age <= 19:
        l2 = "18-19"
    elif age <= 29:
        l2 = "20-29"
    elif age <= 39:
        l2 = "30-39"
    elif age <= 49:
        l2 = "40-49"
    elif age <= 59:
        l2 = "50-59"
    elif age <= 69:
        l2 = "60-69"
    elif age <= 79:
        l2 = "70-79"
    else:
        l2 = "80-95"

    # Level 3: Life stage
    if age <= 29:
        l3 = "Young"
    elif age <= 59:
        l3 = "Adult"
    else:
        l3 = "Senior"

    bank_age_rows.append([str(age), l1, l2, l3, "*"])
pd.DataFrame(bank_age_rows).to_csv("data/hierarchies/bank/age.csv", index=False, header=False)

# job
job_hierarchy = [
    ["management", "White-Collar", "Working", "*"],
    ["technician", "White-Collar", "Working", "*"],
    ["admin.", "White-Collar", "Working", "*"],
    ["blue-collar", "Blue-Collar", "Working", "*"],
    ["services", "Blue-Collar", "Working", "*"],
    ["housemaid", "Blue-Collar", "Working", "*"],
    ["self-employed", "Self-Employed", "Working", "*"],
    ["entrepreneur", "Self-Employed", "Working", "*"],
    ["retired", "Not-Working", "Not-Working", "*"],
    ["unemployed", "Not-Working", "Not-Working", "*"],
    ["student", "Not-Working", "Not-Working", "*"],
    ["Unknown", "Unknown", "Unknown", "*"],
]
pd.DataFrame(job_hierarchy).to_csv("data/hierarchies/bank/job.csv", index=False, header=False)

# marital
bank_marital_hierarchy = [
    ["married", "Married", "*"],
    ["single", "Not-Married", "*"],
    ["divorced", "Not-Married", "*"],
]
pd.DataFrame(bank_marital_hierarchy).to_csv("data/hierarchies/bank/marital.csv", index=False, header=False)

# education
bank_education_hierarchy = [
    ["primary", "No-Higher-Ed", "*"],
    ["secondary", "No-Higher-Ed", "*"],
    ["tertiary", "Higher-Ed", "*"],
    ["Unknown", "Unknown", "*"],
]
pd.DataFrame(bank_education_hierarchy).to_csv("data/hierarchies/bank/education.csv", index=False, header=False)

# default (private)
default_hierarchy = [
    ["yes", "*"],
    ["no", "*"],
]
pd.DataFrame(default_hierarchy).to_csv("data/hierarchies/bank/default.csv", index=False, header=False)

# balance (private, numeric)
balance_values = sorted(bank["balance"].unique())
balance_rows = []
for val in balance_values:
    if val < 0:
        l1 = "Negative"
        l2 = "Negative"
        l3 = "Non-Positive"
    elif val <= 1000:
        l1 = "Low"
        l2 = "Low-Medium"
        l3 = "Non-Negative"
    elif val <= 5000:
        l1 = "Medium"
        l2 = "Low-Medium"
        l3 = "Non-Negative"
    elif val <= 20000:
        l1 = "High"
        l2 = "High"
        l3 = "Non-Negative"
    else:
        l1 = "Very-High"
        l2 = "High"
        l3 = "Non-Negative"
    balance_rows.append([str(val), l1, l2, l3, "*"])
pd.DataFrame(balance_rows).to_csv("data/hierarchies/bank/balance.csv", index=False, header=False)

# housing (private)
housing_hierarchy = [
    ["yes", "*"],
    ["no", "*"],
]
pd.DataFrame(housing_hierarchy).to_csv("data/hierarchies/bank/housing.csv", index=False, header=False)

# loan (private)
loan_hierarchy = [
    ["yes", "*"],
    ["no", "*"],
]
pd.DataFrame(loan_hierarchy).to_csv("data/hierarchies/bank/loan.csv", index=False, header=False)

print("Bank hierarchies created: 8 files")

# =============================================================================
# GERMAN HIERARCHIES
# =============================================================================

german = pd.read_csv("data/clean/german_clean.csv")

# checking_status (private)
checking_hierarchy = [
    ["A11", "Negative", "Has-Account", "*"],
    ["A12", "Positive", "Has-Account", "*"],
    ["A13", "Positive", "Has-Account", "*"],
    ["A14", "No-Account", "No-Account", "*"],
]
pd.DataFrame(checking_hierarchy).to_csv("data/hierarchies/german/checking_status.csv", index=False, header=False)

# duration (private, numeric)
duration_values = sorted(german["duration"].unique())
duration_rows = []
for val in duration_values:
    if val <= 12:
        l1 = "Short"
        l2 = "Short-Medium"
    elif val <= 24:
        l1 = "Medium"
        l2 = "Short-Medium"
    elif val <= 48:
        l1 = "Long"
        l2 = "Long"
    else:
        l1 = "Very-Long"
        l2 = "Long"
    duration_rows.append([str(val), l1, l2, "*"])
pd.DataFrame(duration_rows).to_csv("data/hierarchies/german/duration.csv", index=False, header=False)

# credit_history (private)
credit_history_hierarchy = [
    ["A30", "Good", "*"],
    ["A31", "Good", "*"],
    ["A32", "Good", "*"],
    ["A33", "Poor", "*"],
    ["A34", "Poor", "*"],
]
pd.DataFrame(credit_history_hierarchy).to_csv("data/hierarchies/german/credit_history.csv", index=False, header=False)

# purpose (private)
purpose_hierarchy = [
    ["A40", "Vehicle", "Purchase", "*"],
    ["A41", "Vehicle", "Purchase", "*"],
    ["A42", "Goods", "Purchase", "*"],
    ["A43", "Goods", "Purchase", "*"],
    ["A44", "Goods", "Purchase", "*"],
    ["A45", "Home", "Purchase", "*"],
    ["A46", "Personal", "Non-Purchase", "*"],
    ["A48", "Personal", "Non-Purchase", "*"],
    ["A49", "Business", "Non-Purchase", "*"],
    ["A410", "Other", "Other", "*"],
]
pd.DataFrame(purpose_hierarchy).to_csv("data/hierarchies/german/purpose.csv", index=False, header=False)

# credit_amount (private, numeric)
credit_amount_values = sorted(german["credit_amount"].unique())
credit_amount_rows = []
for val in credit_amount_values:
    if val <= 1500:
        l1 = "Very-Low"
        l2 = "Low"
    elif val <= 3000:
        l1 = "Low"
        l2 = "Low"
    elif val <= 6000:
        l1 = "Medium"
        l2 = "High"
    elif val <= 12000:
        l1 = "High"
        l2 = "High"
    else:
        l1 = "Very-High"
        l2 = "High"
    credit_amount_rows.append([str(val), l1, l2, "*"])
pd.DataFrame(credit_amount_rows).to_csv("data/hierarchies/german/credit_amount.csv", index=False, header=False)

# savings_status (private)
savings_hierarchy = [
    ["A61", "Low", "Has-Savings", "*"],
    ["A62", "Low", "Has-Savings", "*"],
    ["A63", "Low", "Has-Savings", "*"],
    ["A64", "High", "Has-Savings", "*"],
    ["A65", "Unknown", "Unknown", "*"],
]
pd.DataFrame(savings_hierarchy).to_csv("data/hierarchies/german/savings_status.csv", index=False, header=False)

# employment (public)
employment_hierarchy = [
    ["A71", "Unemployed", "Not-Employed", "*"],
    ["A72", "Short-Term", "Employed", "*"],
    ["A73", "Long-Term", "Employed", "*"],
    ["A74", "Long-Term", "Employed", "*"],
    ["A75", "Long-Term", "Employed", "*"],
]
pd.DataFrame(employment_hierarchy).to_csv("data/hierarchies/german/employment.csv", index=False, header=False)

# installment_rate (private, numeric)
installment_hierarchy = [
    ["1", "Low", "*"],
    ["2", "Low", "*"],
    ["3", "High", "*"],
    ["4", "High", "*"],
]
pd.DataFrame(installment_hierarchy).to_csv("data/hierarchies/german/installment_rate.csv", index=False, header=False)

# personal_status (public)
personal_status_hierarchy = [
    ["A91", "Male", "*"],
    ["A92", "Female", "*"],
    ["A93", "Male", "*"],
    ["A94", "Male", "*"],
]
pd.DataFrame(personal_status_hierarchy).to_csv("data/hierarchies/german/personal_status.csv", index=False, header=False)

# other_parties (private)
other_parties_hierarchy = [
    ["A101", "None", "*"],
    ["A102", "Has-Support", "*"],
    ["A103", "Has-Support", "*"],
]
pd.DataFrame(other_parties_hierarchy).to_csv("data/hierarchies/german/other_parties.csv", index=False, header=False)

# residence_since (private, numeric)
residence_hierarchy = [
    ["1", "Short", "*"],
    ["2", "Short", "*"],
    ["3", "Long", "*"],
    ["4", "Long", "*"],
]
pd.DataFrame(residence_hierarchy).to_csv("data/hierarchies/german/residence_since.csv", index=False, header=False)

# property (public)
property_hierarchy = [
    ["A121", "Has-Property", "*"],
    ["A122", "Has-Property", "*"],
    ["A123", "Has-Property", "*"],
    ["A124", "No-Property", "*"],
]
pd.DataFrame(property_hierarchy).to_csv("data/hierarchies/german/property.csv", index=False, header=False)

# age (public, numeric)
german_age_values = sorted(german["age"].unique())
german_age_rows = []
for age in german_age_values:
    # Level 1: 5-year bands
    if age <= 24:
        l1 = "19-24"
    elif age <= 29:
        l1 = "25-29"
    elif age <= 34:
        l1 = "30-34"
    elif age <= 39:
        l1 = "35-39"
    elif age <= 44:
        l1 = "40-44"
    elif age <= 49:
        l1 = "45-49"
    elif age <= 54:
        l1 = "50-54"
    elif age <= 59:
        l1 = "55-59"
    elif age <= 64:
        l1 = "60-64"
    elif age <= 69:
        l1 = "65-69"
    else:
        l1 = "70-75"

    # Level 2: 10-year bands
    if age <= 29:
        l2 = "19-29"
    elif age <= 39:
        l2 = "30-39"
    elif age <= 49:
        l2 = "40-49"
    elif age <= 59:
        l2 = "50-59"
    elif age <= 69:
        l2 = "60-69"
    else:
        l2 = "70-75"

    # Level 3: Life stage
    if age <= 29:
        l3 = "Young"
    elif age <= 59:
        l3 = "Adult"
    else:
        l3 = "Senior"

    german_age_rows.append([str(age), l1, l2, l3, "*"])
pd.DataFrame(german_age_rows).to_csv("data/hierarchies/german/age.csv", index=False, header=False)

# other_payment_plans (private)
other_payment_hierarchy = [
    ["A141", "Has-Other", "*"],
    ["A142", "Has-Other", "*"],
    ["A143", "None", "*"],
]
pd.DataFrame(other_payment_hierarchy).to_csv("data/hierarchies/german/other_payment_plans.csv", index=False,
                                             header=False)

# housing (public)
german_housing_hierarchy = [
    ["A151", "Non-Owner", "*"],
    ["A152", "Owner", "*"],
    ["A153", "Non-Owner", "*"],
]
pd.DataFrame(german_housing_hierarchy).to_csv("data/hierarchies/german/housing.csv", index=False, header=False)

# existing_credits (private, numeric)
existing_credits_hierarchy = [
    ["1", "Single", "*"],
    ["2", "Multiple", "*"],
    ["3", "Multiple", "*"],
    ["4", "Multiple", "*"],
]
pd.DataFrame(existing_credits_hierarchy).to_csv("data/hierarchies/german/existing_credits.csv", index=False,
                                                header=False)

# job (public)
german_job_hierarchy = [
    ["A171", "Unskilled", "*"],
    ["A172", "Unskilled", "*"],
    ["A173", "Skilled", "*"],
    ["A174", "Skilled", "*"],
]
pd.DataFrame(german_job_hierarchy).to_csv("data/hierarchies/german/job.csv", index=False, header=False)

# num_dependents (private, numeric)
num_dependents_hierarchy = [
    ["1", "*"],
    ["2", "*"],
]
pd.DataFrame(num_dependents_hierarchy).to_csv("data/hierarchies/german/num_dependents.csv", index=False, header=False)

# telephone (public)
telephone_hierarchy = [
    ["A191", "*"],
    ["A192", "*"],
]
pd.DataFrame(telephone_hierarchy).to_csv("data/hierarchies/german/telephone.csv", index=False, header=False)

# foreign_worker (public)
foreign_worker_hierarchy = [
    ["A201", "*"],
    ["A202", "*"],
]
pd.DataFrame(foreign_worker_hierarchy).to_csv("data/hierarchies/german/foreign_worker.csv", index=False, header=False)

print("German hierarchies created: 20 files")
print("\nDone. All hierarchy files saved to data/hierarchies/")