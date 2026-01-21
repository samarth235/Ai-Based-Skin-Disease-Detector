def refine_diagnosis(category, confidence):
    disease_map = {

        "Acne": [
            "Acne Vulgaris",
            "Cystic Acne",
            "Hormonal Acne"
        ],

        "Eczema": [
            "Atopic Dermatitis",
            "Contact Dermatitis",
            "Nummular Eczema"
        ],

        "Psoriasis": [
            "Plaque Psoriasis",
            "Guttate Psoriasis",
            "Inverse Psoriasis"
        ],

        "Ringworm": [
            "Tinea Corporis (Body Ringworm)",
            "Tinea Cruris (Jock Itch)",
            "Tinea Faciei (Facial Ringworm)"
        ],

        "Melanoma": [
            "Superficial Spreading Melanoma",
            "Nodular Melanoma",
            "Lentigo Maligna Melanoma"
        ]
    }

    if category not in disease_map:
        return category, "Uncertain subtype"

    # High confidence → specific diagnosis
    if confidence > 0.75:
        return disease_map[category][0], "High confidence subtype match"

    # Medium confidence
    elif confidence > 0.55:
        return disease_map[category][1], "Moderate confidence subtype match"

    # Low confidence
    else:
        return disease_map[category][2], "Low confidence subtype match"

