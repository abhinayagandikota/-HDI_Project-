import numpy as np
import pandas as pd

np.random.seed(42)

n = 400

life_exp = np.random.uniform(50, 85, n)
mean_school = np.random.uniform(3, 14, n)
expected_school = np.random.uniform(5, 20, n)
gni = np.random.uniform(500, 70000, n)

def categorize_hdi(le, ms, es, gni):
    hdi_score = (le - 50) / 35 * 0.4
    hdi_score += ((ms * es) ** 0.5 - 2.5) / (18.5 - 2.5) * 0.3
    hdi_score += (np.log(gni) - np.log(500)) / (np.log(70000) - np.log(500)) * 0.3
    hdi_score = min(1.0, max(0.0, hdi_score))
    if hdi_score >= 0.80:
        return "Very High"
    elif hdi_score >= 0.65:
        return "High"
    elif hdi_score >= 0.50:
        return "Medium"
    else:
        return "Low"

categories = [categorize_hdi(le, ms, es, g) for le, ms, es, g in zip(life_exp, mean_school, expected_school, gni)]

countries = [
    "Norway", "Switzerland", "Ireland", "Germany", "Hong Kong", "Australia", "Iceland",
    "Sweden", "Singapore", "Netherlands", "Denmark", "Finland", "Canada", "New Zealand",
    "United Kingdom", "United States", "Belgium", "Liechtenstein", "Japan", "Austria",
    "Luxembourg", "Israel", "South Korea", "Slovenia", "Spain", "Czechia", "France",
    "Malta", "Italy", "Estonia", "Cyprus", "Greece", "Poland", "Lithuania",
    "United Arab Emirates", "Saudi Arabia", "Andorra", "Slovakia", "Latvia", "Portugal",
    "Qatar", "Chile", "Croatia", "Bahrain", "Kuwait", "Hungary", "Argentina", "Oman",
    "Russia", "Montenegro", "Bulgaria", "Romania", "Belarus", "Bahamas", "Uruguay",
    "Kazakhstan", "Malaysia", "Serbia", "Thailand", "Costa Rica", "Albania", "Iran",
    "Georgia", "Sri Lanka", "Cuba", "Mexico", "Brazil", "Colombia", "Armenia",
    "Maldives", "Peru", "China", "Ecuador", "Azerbaijan", "Ukraine", "Indonesia",
    "Philippines", "Egypt", "South Africa", "Vietnam", "Iraq", "Morocco",
    "Kyrgyzstan", "India", "Bangladesh", "Cambodia", "Kenya", "Nepal", "Pakistan",
    "Ghana", "Myanmar", "Angola", "Nigeria", "Ethiopia", "Mozambique", "Sierra Leone",
    "Niger", "Chad", "South Sudan", "Burundi", "Somalia", "Central African Republic",
    "Haiti", "Afghanistan", "Yemen", "Madagascar", "Togo", "Malawi", "Benin",
    "Zimbabwe", "Tanzania", "Senegal", "Uganda", "Zambia", "Mali", "Burkina Faso",
    "Rwanda", "Guinea", "Liberia", "Mauritania", "Eritrea", "Gambia", "Lesotho",
    "Comoros", "Sudan", "Djibouti", "Congo", "Cameroon", "Ivory Coast", "Nicaragua",
    "Honduras", "Guatemala", "Paraguay", "Bolivia", "El Salvador", "Panama",
    "Trinidad and Tobago", "Mauritius", "Fiji", "Botswana", "Gabon", "Namibia",
    "Algeria", "Tunisia", "Jordan", "Lebanon", "Turkey", "Uzbekistan", "Turkmenistan",
    "Mongolia", "Tajikistan", "Laos", "Papua New Guinea", "Solomon Islands",
    "Vanuatu", "Samoa", "Tonga", "Micronesia", "Kiribati", "Timor-Leste"
]

selected = [countries[i % len(countries)] for i in range(n)]

df = pd.DataFrame({
    "country": selected,
    "life_expectancy": np.round(life_exp, 1),
    "mean_schooling_years": np.round(mean_school, 1),
    "expected_schooling_years": np.round(expected_school, 1),
    "gni_per_capita": np.round(gni, 0),
    "hdi_category": categories
})

df.to_csv("C:/Users/Admin/Desktop/smartbridge2/HDI_Project/data/hdi_data.csv", index=False)
print(f"Dataset generated: {len(df)} rows")
print(df["hdi_category"].value_counts())
print(df.head())
