from enum import Enum, auto

class GrassFamily(Enum):
    BERMUDA = "Bermuda"
    RYE = "Rye"
    BENT = "Bent"
    FESCUE = "Fescue"
    ZOYSIA = "Zoysia"
    POA = "Poa"
    KIKUYU = "Kikuyu"
    PASPALUM = "Paspalum"

class GrassType(Enum):
    # Bermuda varieties
    COMMON_BERMUDA = ("Bermudagrass", GrassFamily.BERMUDA)
    TIFEAGLE_BERMUDA = ("Tifeagle Bermudagrass", GrassFamily.BERMUDA)
    TIFSPORT_BERMUDA = ("TifSport Bermudagrass", GrassFamily.BERMUDA)
    
    # Rye varieties
    COMMON_RYE = ("Rye", GrassFamily.RYE)
    PERENNIAL_RYE = ("Perennial Rye", GrassFamily.RYE)
    OVERSEEDED_RYE = ("Overseeded Rye", GrassFamily.RYE)
    
    # Other grass types
    BENTGRASS = ("Bentgrass", GrassFamily.BENT)
    FESCUE = ("Fescue", GrassFamily.FESCUE)
    ZOYSIA = ("Zoysia", GrassFamily.ZOYSIA)
    POA_ANNUA = ("Poa annua", GrassFamily.POA)
    KIKUYUGRASS = ("Kikuyugrass", GrassFamily.KIKUYU)
    PASPALUM = ("Paspalum", GrassFamily.PASPALUM)

    def __init__(self, label, family):
        self.label = label
        self.family = family

    @classmethod
    def get_by_family(cls, family):
        """Return all grass types belonging to a specific family"""
        return [grass for grass in cls if grass.family == family]

class Course:
    _instances = {}  # Class variable to store all course instances
    
    def __init__(self, name=None, year=None, city=None, state=None, par=None, par5=None, par3=None, yardage=None, 
                 fairway_grass=None, rough_grass=None, green_grass=None, 
                 record_score=None, record_holder=None, record_year=None, 
                 year_established=None, designer=None):
        self.name = name
        self.year = year
        self.location = {
            "city": city,
            "state": state
        }
        # Convert single grass types to lists and ensure inputs are always lists
        self.details = {
            "par": par,
            "par5": par5,
            "par3": par3,
            "yardage": yardage,
            "grass_type": {
                "fairway": [fairway_grass] if isinstance(fairway_grass, GrassType) else fairway_grass,
                "rough": [rough_grass] if isinstance(rough_grass, GrassType) else rough_grass,
                "green": [green_grass] if isinstance(green_grass, GrassType) else green_grass
            },
            "record" : {
                "score" : record_score,
                "holder" : record_holder,
                "year" : record_year
            },
            "established" : year_established,
            "designer" : designer
        }
        if name:  # Only store named courses
            Course._instances[(name, year)] = self

    @classmethod
    def get(cls, name, year):
        """Get a course instance by name and year"""
        key = (name, year)
        if key not in cls._instances:
            raise KeyError(f"No course found for {name} in year {year}")
        return cls._instances[key]

    def __repr__(self):
        return f"GolfCourse(name='{self.name}')"

# Example course definition
Course(
    name="Kapalua_Resort_(Plantation_Course)",
    year = 2026,
    city="Kapalua",
    state="HI",
    par=73,
    par5=4,
    par3=3,
    yardage=7596,
    fairway_grass=[GrassType.COMMON_BERMUDA],
    rough_grass=[GrassType.COMMON_BERMUDA],
    green_grass=[GrassType.TIFEAGLE_BERMUDA],
    record_score=258,
    record_holder="cameron smith",
    record_year=2022,
    year_established=1991,
    designer="Bill Coore / Ben Crenshaw"
)

Course(
    name="Waialae_Country_Club",
    year=2025,
    city="Honolulu",
    state="HI",
    par=70,
    par5=2,
    par3=4,
    yardage=7044,
    fairway_grass=[GrassType.COMMON_BERMUDA],
    rough_grass=[GrassType.COMMON_BERMUDA],
    green_grass=[GrassType.TIFEAGLE_BERMUDA],
    record_score=253,
    record_holder="justin thomas",
    record_year=2017,
    year_established=1927,
    designer="Seth Raynor"
)

Course(
    name="La_Quinta_Country_Club",
    year=2025,
    city="La Quinta",
    state="CA",
    par=72,
    par5=4,
    par3=4,
    yardage=7210,
    fairway_grass=[GrassType.OVERSEEDED_RYE, GrassType.COMMON_RYE],
    rough_grass=[GrassType.OVERSEEDED_RYE, GrassType.COMMON_RYE],
    green_grass=[GrassType.TIFEAGLE_BERMUDA],
    record_score=259,
    record_holder="nick dunlap",
    record_year=2024,
    year_established=1959,
    designer="Lawrence Hughes"
)

Course(
    name="Torrey_Pines_(South_Course)",
    year=2025,
    city="San Diego",
    state="CA",
    par=72,
    par5=4,
    par3=4,
    yardage=7765,
    fairway_grass=[GrassType.KIKUYUGRASS, GrassType.COMMON_RYE],
    rough_grass=[GrassType.KIKUYUGRASS, GrassType.COMMON_RYE],
    green_grass=[GrassType.BENTGRASS, GrassType.POA_ANNUA],
    record_score=266,
    record_holder="tiger woods",
    record_year=1999,
    year_established=1957,
    designer="William Bell"
)

Course(
    name="Pebble_Beach_Golf_Links",
    year=2025,
    city="Pebble Beach",
    state="CA",
    par=72,
    par5=4,
    par3=4,
    yardage=6972,
    fairway_grass=[GrassType.BENTGRASS],
    rough_grass=[GrassType.BENTGRASS],
    green_grass=[GrassType.POA_ANNUA],
    record_score=265,
    record_holder="brandt snedeker",
    record_year=2015,
    year_established=1919,
    designer="Jack Neville & Douglas Grant"
)

Course(
    name="TPC_Scottsdale_(Stadium_Course)",
    year=2025,
    city="Scottsdale",
    state="AZ",
    par=72,
    par5=3,
    par3=4,
    yardage=7261,
    fairway_grass=[GrassType.COMMON_BERMUDA, GrassType.COMMON_RYE],
    rough_grass=[GrassType.COMMON_BERMUDA, GrassType.COMMON_RYE],
    green_grass=[GrassType.COMMON_BERMUDA, GrassType.COMMON_RYE],
    record_score=256,
    record_holder="phil mickelson",
    record_year=2013,
    year_established=1988,
    designer="Tom Weiskopf \ Jay Morrish"
)

Course(
    name="Vidanta_Vallarta",
    year=2025,
    city="Puerto Vallarta",
    state="MX",
    par=71,
    par5=4,
    par3=5,
    yardage=7436,
    fairway_grass=[GrassType.PASPALUM],
    rough_grass=[GrassType.PASPALUM],
    green_grass=[GrassType.PASPALUM],
    record_score=260,
    record_holder="tony finau",
    record_year=2023,
    year_established=1974,
    designer="Greg Norman"
)

Course(
    name="PGA_National_Resort_&_Spa_(The_Champion)",
    year=2025,
    city="Palm Beach Gardens",
    state="FL",
    par=71,
    par5=3,
    par3=4,
    yardage=7167,
    fairway_grass=[GrassType.COMMON_BERMUDA],
    rough_grass=[GrassType.COMMON_BERMUDA, GrassType.OVERSEEDED_RYE],
    green_grass=[GrassType.TIFEAGLE_BERMUDA],
    record_score=264,
    record_holder="justin leonard",
    record_year=2003,
    year_established=1981,
    designer="Tom Fazio"
)

Course(
    name="Arnold_Palmer's_Bay_Hill_Club_&_Lodge",
    year=2025,
    city="Orlando",
    state="FL",
    par=72,
    par5=4,
    par3=4,
    yardage=7466,
    fairway_grass=[GrassType.COMMON_BERMUDA, GrassType.COMMON_RYE],
    rough_grass=[GrassType.COMMON_BERMUDA, GrassType.COMMON_RYE],
    green_grass=[GrassType.TIFEAGLE_BERMUDA],
    record_score=265,
    record_holder="payne stewart",
    record_year=1987,
    year_established=1961,
    designer="Dick Wilson\Joe Lee"
)

Course(
    name="TPC_Sawgrass_(THE_PLAYERS_Stadium_Course)",
    year=2025,
    city="Ponte Vedra Beach",
    state="FL",
    par=72,
    par5=4,
    par3=4,
    yardage=7352,
    fairway_grass=[GrassType.TIFSPORT_BERMUDA],
    rough_grass=[GrassType.TIFSPORT_BERMUDA],
    green_grass=[GrassType.TIFSPORT_BERMUDA],
    record_score=264,
    record_holder="greg norman",
    record_year=1994,
    year_established=1980,
    designer="Pete Dye"
)

Course(
    name='Innisbrook_Resort_(Copperhead_Course)',
    year=2025,
    city="Palm Harbor",
    state="FL",
    par=71,
    par5=4,
    par3=5,
    yardage=7352,
    record_score=266,
    record_holder="vijay singh",
    record_year=2004,
    year_established=1972,
    designer="Larry Packard"
)

Course(
    name="Memorial_Park_Golf_Course",
    year=2025,
    city="Houston",
    state="TX",
    par=70,
    par5=3, 
    par3=5,
    yardage=7475,
    fairway_grass=[GrassType.PERENNIAL_RYE],
    rough_grass=[GrassType.PERENNIAL_RYE],
    green_grass=[GrassType.COMMON_BERMUDA],
    record_score=260,
    record_holder="min woo lee",
    record_year=2016,
    year_established=1936,
    designer="John Bredemus"
)







