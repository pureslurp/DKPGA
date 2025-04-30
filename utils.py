import courses
from courses import Course

TOURNAMENT_LIST_2023 = {'Fortinet_Championship': 401552854, 'Presidents_Cup': 401465497, 'Sanderson_Farms_Championship': 401552855, "Shriners_Children's_Open": 401552856, 'ZOZO_CHAMPIONSHIP': 401552857, 'THE_CJ_CUP_in_South_Carolina': 401465501, 'Butterfield_Bermuda_Championship': 401552859, 'World_Wide_Technology_Championship_at_Mayakoba': 401465504, 'Cadence_Bank_Houston_Open': 401465505, 'The_RSM_Classic': 401552860, 'Hero_World_Challenge': 401552861, 'The_Match': 401546052, 'Sentry_Tournament_of_Champions': 401465512, 'Sony_Open_in_Hawaii': 401465513, 'The_American_Express': 401465514, 'Farmers_Insurance_Open': 401465516, 'AT&T_Pebble_Beach_Pro-Am': 401465517, 'WM_Phoenix_Open': 401465518, 'The_Genesis_Invitational': 401465521, 'The_Honda_Classic': 401465522, 'Puerto_Rico_Open': 401465525, 'Arnold_Palmer_Invitational_pres._by_Mastercard': 401465524, 'THE_PLAYERS_Championship': 401465526, 'Valspar_Championship': 401465527, 'WGC-Dell_Technologies_Match_Play': 401465528, 'Corales_Puntacana_Championship': 401465529, 'Valero_Texas_Open': 401465502, 'Masters_Tournament': 401465508, 'RBC_Heritage': 401465509, 'Zurich_Classic_of_New_Orleans': 401465511, 'Mexico_Open': 401465515, 'Wells_Fargo_Championship': 401465519, 'AT&T_Byron_Nelson': 401465520, 'PGA_Championship': 401465523, 'Charles_Schwab_Challenge': 401465530, 'The_Memorial_Tournament_pres._by_Workday': 401465531, 'RBC_Canadian_Open': 401465532, 'U.S._Open': 401465533, 'Travelers_Championship': 401465534, 'Rocket_Mortgage_Classic': 401465535, 'John_Deere_Classic': 401465536, 'Genesis_Scottish_Open': 401465537, 'Barbasol_Championship': 401465538, 'The_Open': 401465539, 'Barracuda_Championship': 401465540, '3M_Open': 401465541, 'Wyndham_Championship': 401465542, 'FedEx_St._Jude_Championship': 401465543, 'BMW_Championship': 401465544, 'TOUR_Championship': 401465545, 'Ryder_Cup': 401552863, 'World_Wide_Technology_Championship': 401552858, 'PGA_TOUR_Q-School_presented_by_Korn_Ferry': 401558309}

TOURNAMENT_LIST_2024 = {
    "The_Sentry": {
        "ID": 401580329,
        "Course": "Kapalua_Resort_(Plantation_Course)"
    },
    "Sony_Open_in_Hawaii": {
        "ID": 401580330,
        "Course": "Waialae_Country_Club"
    },
    "The_American_Express": {
        "ID": 401580331,
        "Course": "La_Quinta_Country_Club"
    },
    "Farmers_Insurance_Open": {
        "ID": 401580332,
        "Course": "Torrey_Pines_(North_Course)"
    },
    "AT&T_Pebble_Beach_Pro-Am": {
        "ID": 401580333,
        "Course": "Spyglass_Hill_GC"
    },
    "WM_Phoenix_Open": {
        "ID": 401580334,
        "Course": "TPC_Scottsdale_(Stadium_Course)"
    },
    "The_Genesis_Invitational": {
        "ID": 401580335,
        "Course": "Riviera_Country_Club"
    },
    "Mexico_Open_at_Vidanta": {
        "ID": 401580336,
        "Course": "Vidanta_Vallarta"
    },
    "Cognizant_Classic": {
        "ID": 401580337,
        "Course": "PGA_National_Resort_&_Spa_(The_Champion)"
    },
    "Arnold_Palmer_Invitational_pres._by_Mastercard": {
        "ID": 401580338,
        "Course": "Arnold_Palmer's_Bay_Hill_Club_&_Lodge"
    },
    "Puerto_Rico_Open": {
        "ID": 401580339,
        "Course": "Grand_Reserve_Country_Club"
    },
    "THE_PLAYERS_Championship": {
        "ID": 401580340,
        "Course": "TPC_Sawgrass_(THE_PLAYERS_Stadium_Course)"
    },
    "Valspar_Championship": {
        "ID": 401580341,
        "Course": "Innisbrook_Resort_(Copperhead_Course)"
    },
    "Texas_Children's_Houston_Open": {
        "ID": 401580342,
        "Course": "Memorial_Park_Golf_Course"
    },
    "Valero_Texas_Open": {
        "ID": 401580343,
        "Course": "TPC_San_Antonio_(Oaks_Course)"
    },
    "Masters_Tournament": {
        "ID": 401580344,
        "Course": "Augusta_National_Golf_Club"
    },
    "RBC_Heritage": {
        "ID": 401580345,
        "Course": "Harbour_Town_Golf_Links"
    },
    "Corales_Puntacana_Championship": {
        "ID": 401580346,
        "Course": "Puntacana_Resort_&_Club_(Corales_Golf_Course)"
    },
    "THE_CJ_CUP_Byron_Nelson": {
        "ID": 401580348,
        "Course": "TPC_Craig_Ranch"
    },
    "Wells_Fargo_Championship": {
        "ID": 401580349,
        "Course": "Quail_Hollow_Club"
    },
    "Myrtle_Beach_Classic": {
        "ID": 401580350,
        "Course": "Dunes_Golf_&_Beach_Club"
    },
    "PGA_Championship": {
        "ID": 401580351,
        "Course": "Valhalla_Golf_Club"
    },
    "Charles_Schwab_Challenge": {
        "ID": 401580352,
        "Course": "Colonial_Country_Club"
    },
    "RBC_Canadian_Open": {
        "ID": 401580353,
        "Course": "Hamilton_Golf_&_Country_Club"
    },
    "the_Memorial_Tournament_pres._by_Workday": {
        "ID": 401580354,
        "Course": "Muirfield_Village_Golf_Club"
    },
    "U.S._Open": {
        "ID": 401580355,
        "Course": "Pinehurst_No._2"
    },
    "Travelers_Championship": {
        "ID": 401580356,
        "Course": "TPC_River_Highlands"
    },
    "Rocket_Mortgage_Classic": {
        "ID": 401580357,
        "Course": "Detroit_Golf_Club"
    },
    "John_Deere_Classic": {
        "ID": 401580358,
        "Course": "TPC_Deere_Run"
    },
    "Genesis_Scottish_Open": {
        "ID": 401580359,
        "Course": "The_Renaissance_Club"
    },
    "The_Open": {
        "ID": 401580360,
        "Course": "Royal_Troon_Golf_Course"
    },
    "Barracuda_Championship": {
        "ID": 401580361,
        "Course": "Tahoe_Mountain_Club_(Old_Greenwood)"
    },
    "3M_Open": {
        "ID": 401580362,
        "Course": "TPC_Twin_Cities"
    },
    "Wyndham_Championship": {
        "ID": 401580363,
        "Course": "Sedgefield_Country_Club"
    },
    "FedEx_St._Jude_Championship": {
        "ID": 401580364,
        "Course": "TPC_Southwind"
    },
    "BMW_Championship": {
        "ID": 401580365,
        "Course": "Castle_Pines_Golf_Club"
    },
    "TOUR_Championship": {
        "ID": 401580366,
        "Course": "East_Lake_Golf_Club"
    }
}

TOURNAMENT_LIST_2025 = {
    "The_Sentry": {
        "ID": 401703489,
        "Course": Course.get("Kapalua_Resort_(Plantation_Course)", 2025),
        "pga-url": "the-sentry/R2025016"
    },
    "Sony_Open_in_Hawaii": {
        "ID": 401703490,
        "Course": Course.get("Waialae_Country_Club", 2025),
        "pga-url": "sony-open-in-hawaii/R2025006"
    },
    "The_American_Express": {
        "ID": 401703491,
        "Course": "La_Quinta_Country_Club",
        "pga-url": "the-american-express/R2025002"
    },
    "Farmers_Insurance_Open": {
        "ID": 401703492,
        "Course": "Torrey_Pines_(South_Course)",
        "pga-url": "farmers-insurance-open/R2025004"
    },
    "AT&T_Pebble_Beach_Pro-Am": {
        "ID": 401703493,
        "Course": "Pebble_Beach_Golf_Links",
        "pga-url": "att-pebble-beach-pro-am/R2025005"
    },
    "WM_Phoenix_Open": {
        "ID": 401703494,
        "Course": "TPC_Scottsdale_(Stadium_Course)",
        'pga-url': 'wm-phoenix-open/R2025003'
    },
    "The_Genesis_Invitational": {
        "ID": 401703495,
        "Course": "Torrey Pines_(South_Course)",
        "pga-url": "the-genesis-invitational/R2025007"
    },
    "Mexico_Open_at_VidantaWorld": {
        "ID": 401703496,
        "Course": "Vidanta_Vallarta",
        "pga-url": "mexico-open-at-vidantaworld/R2025540"
    },
    "Cognizant_Classic_in_The_Palm_Beaches": {
        "ID": 401703497,
        "Course": "PGA_National_Resort_&_Spa_(The_Champion)",
        "pga-url": "cognizant-classic-in-the-palm-beaches/R2025010"
    },
    "Arnold_Palmer_Invitational_presented_by_Mastercard": {
        "ID": 401703498,
        "Course": "Arnold_Palmer's_Bay_Hill_Club_&_Lodge",
        "pga-url": "arnold-palmer-invitational-presented-by-mastercard/R2025009"
    },
    "Puerto_Rico_Open": {
        "ID": 401703499,
        "Course": "Grand_Reserve_Country_Club"
    },
    "THE_PLAYERS_Championship": {
        "ID": 401703500,
        "Course": "TPC_Sawgrass_(THE_PLAYERS_Stadium_Course)",
        "pga-url": "the-players-championship/R2025011"
    },
    "Valspar_Championship": {
        "ID": 401703501,
        "Course": "Innisbrook_Resort_(Copperhead_Course)",
        "pga-url": "valspar-championship/R2025475"
    },
    "Texas_Children's_Houston_Open": {
        "ID": 401703502,
        "Course": "Memorial_Park_Golf_Course",
        "pga-url": "texas-childrens-houston-open/R2025020"
    },
    "Valero_Texas_Open": {
        "ID": 401703503,
        "Course": None,
        "pga-url": "valero-texas-open/R2025041",
        "Course": "TPC_San_Antonio_(Oaks_Course)"
    },
    "Masters_Tournament": {
        "ID": 401703504,
        "Course": "Augusta_National_Golf_Club",
        "pga-url": "masters-tournament/R2025014"
    },
    "RBC_Heritage": {
        "ID": 401703505,
        "Course": "Harbour_Town_Golf_Links",
        "pga-url": "rbc-heritage/R2025012"
    },
    "Corales_Puntacana_Championship": {
        "ID": 401703506,
        "Course": "Puntacana_Resort_&_Club_(Corales_Golf_Course)",
        "pga-url": "corales-puntacana-championship/R2025016"
    },
    "THE_CJ_CUP_Byron_Nelson": {
        "ID": 401703508,
        "Course": "TPC_Craig_Ranch",
        "pga-url": "the-cj-cup-byron-nelson/R2025019"
    },
    "Truist_Championship": {
        "ID": 401703509,
        "Course": "Quail_Hollow_Club"
    },
    "Myrtle_Beach_Classic": {
        "ID": 401703510,
        "Course": "Dunes_Golf_&_Beach_Club"
    },
    "PGA_Championship": {
        "ID": 401703511,
        "Course": "Valhalla_Golf_Club"
    },
    "Charles_Schwab_Challenge": {
        "ID": 401703512,
        "Course": "Colonial_Country_Club"
    },
    "the_Memorial_Tournament_pres._by_Workday": {
        "ID": 401703513,
        "Course": "Muirfield_Village_Golf_Club"
    },
    "RBC_Canadian_Open": {
        "ID": 401703514,
        "Course": "Hamilton_Golf_&_Country_Club"
    },
    "U.S._Open": {
        "ID": 401703515,
        "Course": "Pinehurst_No._2"
    },
    "Travelers_Championship": {
        "ID": 401703516,
        "Course": "TPC_River_Highlands"
    },
    "Rocket_Mortgage_Classic": {
        "ID": 401703517,
        "Course": "Detroit_Golf_Club"
    },
    "John_Deere_Classic": {
        "ID": 401703518,
        "Course": "TPC_Deere_Run"
    },
    "Genesis_Scottish_Open": {
        "ID": 401703519,
        "Course": "The_Renaissance_Club"
    },
    "The_Open": {
        "ID": 401703521,
        "Course": "Royal_Troon_Golf_Course"
    },
    "Barracuda_Championship": {
        "ID": 401703522,
        "Course": "Tahoe_Mountain_Club_(Old_Greenwood)"
    },
    "3M_Open": {
        "ID": 401703523,
        "Course": "TPC_Twin_Cities"
    },
    "Wyndham_Championship": {
        "ID": 401703524,
        "Course": "Sedgefield_Country_Club"
    },
    "FedEx_St._Jude_Championship": {
        "ID": 401703525,
        "Course": "TPC_Southwind"
    },
    "BMW_Championship": {
        "ID": 401703530,
        "Course": "Castle_Pines_Golf_Club"
    },
    "TOUR_Championship": {
        "ID": 401703531,
        "Course": "East_Lake_Golf_Club"
    },
    "Procore_Championship": {
        "ID": 401738553,
        "Course": ""
    },
    "Ryder_Cup": {
        "ID": 401734110,
        "Course": ""
    },
    "Sanderson_Farms_Championship": {
        "ID": 401738554,
        "Course": ""
    },
    "Baycurrent_Classic": {
        "ID": 401738555,
        "Course": ""
    },
    "Black_Desert_Championship": {
        "ID": 401738556,
        "Course": ""
    },
    "World_Wide_Technology_Championship": {
        "ID": 401738557,
        "Course": ""
    },
    "Butterfield_Bermuda_Championship": {
        "ID": 401738558,
        "Course": ""
    },
    "The_RSM_Classic": {
        "ID": 401738559,
        "Course": ""
    },
    "Hero_World_Challenge": {
        "ID": 401738560,
        "Course": ""
    }
}

def fix_names(name):
    if name == "Si Woo":
        return "si woo kim"
    elif name == "Byeong Hun":
        return "byeong hun an"
    elif name == "Erik Van":
        return "erik van rooyen"
    elif name == "Adrien Dumont":
        return "adrien dumont de chassart"
    elif name == "Matthias Schmid":
        return "matti schmid"
    elif name == "Samuel Stevens":
        return "sam stevens"
    elif name == "Benjamin Silverman":
        return "ben silverman"
    elif name =="Min Woo":
        return "min woo lee"
    elif name == "Santiago De":
        return "santiago de la fuente"
    elif name == "Jose Maria":
        return "jose maria olazabal"
    elif name == "Niklas Norgaard Moller":
        return "niklas moller"
    elif name == "Jordan L. Smith":
        return "jordan l."
    elif name == "daniel bradbury":
        return "dan bradbury"
    elif name == "Ludvig Åberg":
        return "ludvig aberg"
    elif name == "Cam Davis":
        return "cameron davis"
    elif name == "Nicolai Højgaard":
        return "nicolai hojgaard"
    elif name == "Nico Echavarria":
        return "nicolas echavarria"
    elif name == "Rasmus Højgaard":
        return "rasmus hojgaard"
    elif name == "Thorbjørn Olesen":
        return "thorbjorn olesen"
    elif name == "k.h. lee":
        return "kyoung-hoon lee"
    else:
        return name.lower()
