def confidence_badge(conf):
    if conf >= 0.75:
        return "🟢 High Confidence"
    elif conf >= 0.5:
        return "🟡 Medium Confidence"
    else:
        return "🔴 Low Confidence"


def clinical_recommendation(disease):
    recs = {
        "Acne Vulgaris": [
            "Maintain gentle skin hygiene",
            "Avoid squeezing lesions",
            "Topical treatments may help"
        ],
        "Atopic Dermatitis": [
            "Use moisturizers regularly",
            "Avoid known allergens",
            "Dermatology consultation recommended"
        ],
        "Plaque Psoriasis": [
            "Chronic condition – long-term care needed",
            "Avoid skin trauma",
            "Medical supervision advised"
        ],
        "Tinea Corporis (Body Ringworm)": [
            "Fungal infection – avoid sharing towels",
            "Keep area dry",
            "Antifungal treatment recommended"
        ],
        "Melanoma": [
            "URGENT medical attention required",
            "Avoid sun exposure",
            "Immediate dermatologist consultation"
        ]
    }

    return recs.get(disease, ["Consult a dermatologist for further evaluation"])
