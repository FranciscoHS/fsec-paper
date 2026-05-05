"""
50 semantically matched prompt sets across 6 language registers for
multi-contrastive steering experiments.

Registers: formal, casual, academic, simple, literary, slang
Each set expresses the same meaning in all 6 registers.
Short prompts (5-12 tokens) designed for activation extraction.

Used by scripts/directions/run_register_contrastive.py to compute
register-specific feature directions via all-pairwise contrasts (Section 8
of multi_contrastive_projection.pdf).

Note: After running test_register_completions.py, prune REGISTERS and
corresponding keys if some registers don't work with the target model.
"""

REGISTERS = ['formal', 'casual', 'academic', 'simple', 'literary', 'slang']

PROMPT_SETS = [
    # === Nature & Weather ===
    {
        'formal': "The temperature has risen considerably this",
        'casual': "It's gotten super hot out",
        'academic': "Ambient temperatures have elevated significantly during this",
        'simple': "It is very hot outside today",
        'literary': "The sun poured its molten gold upon the",
        'slang': "It's literally so hot out rn",
    },
    {
        'formal': "Heavy rainfall is anticipated throughout the",
        'casual': "Looks like it's gonna pour all",
        'academic': "Sustained precipitation events are projected for the",
        'simple': "It will rain a lot today",
        'literary': "The sky, heavy with promise, prepared to weep its",
        'slang': "It's about to dump rain all",
    },
    {
        'formal': "The winter season has been particularly severe this",
        'casual': "This winter has been so brutal",
        'academic': "The current winter season exhibits anomalous severity in",
        'simple': "This winter is very cold and there is",
        'literary': "Winter descended with an iron grip, cloaking the world in",
        'slang': "Winter this year is absolutely insane it's",
    },
    {
        'formal': "The coastal winds have intensified over recent",
        'casual': "The wind by the beach has been crazy",
        'academic': "Littoral wind velocities have increased markedly during recent",
        'simple': "The wind near the ocean is very strong",
        'literary': "Along the shore, the winds sang their wild and restless",
        'slang': "The wind at the coast is going absolutely",
    },
    {
        'formal': "Spring has brought an abundance of flowering",
        'casual': "Spring is here and everything is blooming like",
        'academic': "Vernal conditions have precipitated extensive floral emergence across",
        'simple': "In spring many flowers grow and the trees get",
        'literary': "Spring unfurled its tender petals, painting the meadows in",
        'slang': "Spring hit and now everything is blooming so",
    },
    # === Food & Cooking ===
    {
        'formal': "The restaurant has received exceptional reviews for its",
        'casual': "Everyone says this place has amazing",
        'academic': "Critical evaluations of the establishment consistently commend its",
        'simple': "People say the food here is very good",
        'literary': "The kitchen conjured dishes of extraordinary beauty and",
        'slang': "This restaurant is getting hyped up for its",
    },
    {
        'formal': "The preparation of this dish requires considerable",
        'casual': "Making this recipe takes a lot of",
        'academic': "The culinary preparation necessitates a substantial investment of",
        'simple': "This food takes a long time to make",
        'literary': "To craft such a dish demands patience and the hand of",
        'slang': "This recipe is lowkey such a pain to",
    },
    {
        'formal': "The market offers an impressive selection of seasonal",
        'casual': "The market has a bunch of really fresh",
        'academic': "The marketplace presents a diverse array of seasonally available",
        'simple': "The market has many fruits and vegetables that are",
        'literary': "The stalls overflowed with autumn's generous bounty of",
        'slang': "The market rn has the freshest stuff like",
    },
    {
        'formal': "Coffee consumption has increased significantly among young",
        'casual': "So many young people are drinking coffee now",
        'academic': "Caffeine consumption patterns among younger demographics show marked",
        'simple': "Many young people drink a lot of coffee",
        'literary': "The dark elixir found eager devotees among the",
        'slang': "Everyone and their mom is on coffee now like",
    },
    {
        'formal': "Traditional cooking methods are experiencing a notable",
        'casual': "Old school cooking is making a big",
        'academic': "Conventional culinary methodologies are undergoing a significant resurgence in",
        'simple': "People are cooking the old way again and it is",
        'literary': "Ancient recipes stirred once more in modern kitchens, their flavors",
        'slang': "Old school cooking is having a whole moment right",
    },
    # === Technology ===
    {
        'formal': "The latest software update addresses several critical",
        'casual': "They finally fixed a bunch of annoying",
        'academic': "The recent software iteration resolves multiple previously identified",
        'simple': "The new update fixes problems with the",
        'literary': "With quiet precision, the update mended the fractures within the",
        'slang': "They dropped a patch and fixed all the dumb",
    },
    {
        'formal': "Artificial intelligence continues to transform numerous",
        'casual': "AI is changing everything these days especially in",
        'academic': "Artificial intelligence systems are increasingly permeating diverse sectors of",
        'simple': "Smart computers are helping people do many things",
        'literary': "The machine intelligence unfurled its reach across the landscape of",
        'slang': "AI is literally taking over everything and it's",
    },
    {
        'formal': "Battery technology has advanced considerably in the past",
        'casual': "Batteries have gotten way better over the last",
        'academic': "Electrochemical energy storage has progressed substantially during the preceding",
        'simple': "Batteries last much longer now than they did",
        'literary': "Captured lightning, once fleeting, now endures within vessels of",
        'slang': "Battery tech is insane now compared to what it",
    },
    {
        'formal': "Cybersecurity concerns have prompted organizations to invest in",
        'casual': "Companies are spending way more on security because of",
        'academic': "Escalating cybersecurity threats have catalyzed institutional investment in",
        'simple': "Companies worry about computer hackers so they spend money on",
        'literary': "In the shadowed corridors of the digital realm, guardians now",
        'slang': "Everyone's freaking out about hackers so now companies are",
    },
    {
        'formal': "Remote communication tools have fundamentally altered workplace",
        'casual': "Video calls and chat apps totally changed how we",
        'academic': "Asynchronous and synchronous digital communication modalities have transformed organizational",
        'simple': "People use video calls to work together from",
        'literary': "Across invisible threads, voices now carry where once only letters",
        'slang': "Zoom and Slack literally changed work forever and now",
    },
    # === Education ===
    {
        'formal': "The institution has announced significant changes to its admissions",
        'casual': "The school just changed how they pick who gets",
        'academic': "The institution has promulgated substantial revisions to its admissions",
        'simple': "The school changed the rules for who can come",
        'literary': "The gates of learning swing upon new hinges as the",
        'slang': "They totally changed how admissions work and now it's",
    },
    {
        'formal': "Student performance has improved following the implementation of",
        'casual': "Kids are doing better in school since they started",
        'academic': "Measurable improvements in student outcomes have been observed subsequent to",
        'simple': "Students are learning better now because of the new",
        'literary': "Young minds, tended with care, blossomed under the light of",
        'slang': "Students are actually doing better now since they rolled out",
    },
    {
        'formal': "The library has expanded its digital resource",
        'casual': "The library added a ton of online stuff",
        'academic': "The institutional library has augmented its digital holdings with",
        'simple': "The library now has many more books you can read",
        'literary': "The library, that cathedral of words, opened new chambers of",
        'slang': "The library just dropped a bunch of new digital",
    },
    {
        'formal': "Graduate enrollment has declined for the third consecutive",
        'casual': "Fewer people are going to grad school for like the",
        'academic': "Post-baccalaureate enrollment figures have exhibited sustained decline across three",
        'simple': "Not as many people want to go to more school",
        'literary': "The halls of advanced study grow quieter with each",
        'slang': "Nobody wants to do grad school anymore it's been",
    },
    {
        'formal': "The curriculum now emphasizes practical skills alongside theoretical",
        'casual': "They're teaching more hands-on stuff now along with",
        'academic': "Curricular revisions have integrated applied competencies with existing theoretical",
        'simple': "Schools now teach useful skills along with book",
        'literary': "Knowledge, once cloistered in theory alone, now walks hand in hand with",
        'slang': "Schools are finally teaching actual useful stuff not just",
    },
    # === Health & Wellness ===
    {
        'formal': "Regular physical activity is essential for maintaining optimal",
        'casual': "You really gotta exercise if you want to stay",
        'academic': "Consistent engagement in physical activity is demonstrably associated with",
        'simple': "Moving your body every day keeps you healthy and",
        'literary': "The body, honored by movement, rewards its keeper with",
        'slang': "You gotta hit the gym if you wanna feel",
    },
    {
        'formal': "Mental health awareness has increased substantially among the general",
        'casual': "People are way more open about mental health now",
        'academic': "Public consciousness of psychological well-being has expanded significantly in",
        'simple': "More people know about taking care of their feelings",
        'literary': "The soul's quiet struggles at last find voice in a world that",
        'slang': "Mental health talk is so normalized now like everyone's",
    },
    {
        'formal': "Vaccination rates have improved following the public awareness",
        'casual': "More people are getting vaccinated since they started doing",
        'academic': "Immunization uptake has increased subsequent to coordinated public health",
        'simple': "More people are getting their shots because the doctors told",
        'literary': "The shield of science, once met with suspicion, now finds",
        'slang': "Vax rates are way up since they started pushing",
    },
    {
        'formal': "Sleep quality significantly impacts daily cognitive",
        'casual': "How you sleep really affects how well you think",
        'academic': "Sleep architecture demonstrably modulates subsequent cognitive performance and",
        'simple': "Getting good sleep helps your brain work better",
        'literary': "In the quiet kingdom of sleep, the mind restores its",
        'slang': "Sleep is no joke it literally makes or breaks your",
    },
    {
        'formal': "The healthcare system faces unprecedented demand for",
        'casual': "Hospitals are totally overwhelmed right now with all the",
        'academic': "The healthcare infrastructure is experiencing extraordinary strain on its",
        'simple': "Hospitals have too many sick people and not enough",
        'literary': "The healers, weary yet resolute, faced a tide of need that",
        'slang': "Hospitals are completely slammed rn there's not enough",
    },
    # === Economy & Work ===
    {
        'formal': "The labor market has demonstrated remarkable resilience despite",
        'casual': "Jobs have held up pretty well even though things",
        'academic': "Labor market indicators exhibit surprising robustness notwithstanding prevailing",
        'simple': "People can still find jobs even though the economy is",
        'literary': "The engine of commerce, battered yet unbroken, continues its",
        'slang': "The job market is somehow still going strong even with",
    },
    {
        'formal': "Interest rates are expected to remain elevated throughout",
        'casual': "Rates are gonna stay high for a while",
        'academic': "Monetary policy projections indicate sustained elevation of interest rates during",
        'simple': "Banks will keep charging more for borrowing money",
        'literary': "The cost of borrowed gold climbs ever higher as",
        'slang': "Interest rates staying high is killing everyone's plans for",
    },
    {
        'formal': "Remote work arrangements have become a permanent feature of",
        'casual': "Working from home is just the norm now for",
        'academic': "Distributed work modalities have been institutionalized as a structural component of",
        'simple': "Many people now work at home instead of going to",
        'literary': "The commuter's journey, once a daily pilgrimage, now fades into",
        'slang': "WFH is permanent now like nobody's going back to",
    },
    {
        'formal': "Small businesses are adapting to the evolving digital",
        'casual': "Small shops are going online more because they have",
        'academic': "Small and medium enterprises are undergoing digital transformation in response to",
        'simple': "Small stores are using the internet to sell things",
        'literary': "The humble shopkeeper, keeper of tradition, now reaches through the",
        'slang': "Small businesses are all going digital now because they literally",
    },
    {
        'formal': "Housing prices have escalated beyond the reach of many",
        'casual': "Houses are so expensive now that most people can't",
        'academic': "Residential real estate valuations have appreciated beyond the financial capacity of",
        'simple': "Houses cost too much money for most people to",
        'literary': "The dream of a home, once within reach, now recedes like",
        'slang': "Housing prices are absolutely unhinged right now nobody can",
    },
    # === Society & Culture ===
    {
        'formal': "Social media has profoundly influenced contemporary public",
        'casual': "Social media is seriously changing how people think about",
        'academic': "Digital media platforms have exerted a transformative influence on public",
        'simple': "What people post online changes how others think",
        'literary': "In the vast chorus of digital voices, public sentiment now",
        'slang': "Social media runs everything now like it literally shapes how",
    },
    {
        'formal': "Diversity and inclusion initiatives have gained considerable momentum in",
        'casual': "Companies are really pushing diversity stuff more than",
        'academic': "Organizational commitments to diversity equity and inclusion have intensified across",
        'simple': "Workplaces want different kinds of people to feel welcome",
        'literary': "The tapestry of the workplace grows richer as new threads of",
        'slang': "DEI is everywhere now like every company is suddenly",
    },
    {
        'formal': "Urbanization continues to reshape the demographic landscape of",
        'casual': "More and more people are moving to cities these",
        'academic': "Accelerating urbanization trends are fundamentally reconfiguring demographic distributions across",
        'simple': "Many people move from small towns to big cities",
        'literary': "The city, that great magnet of ambition, draws ever more souls to",
        'slang': "Everyone's moving to the city now like small towns are",
    },
    {
        'formal': "Volunteerism has experienced a notable resurgence in recent",
        'casual': "Way more people are volunteering now than they used",
        'academic': "Civic engagement through voluntarism has exhibited a marked increase in",
        'simple': "More people are helping others for free because they",
        'literary': "The spirit of generosity, dormant but not dead, awoke anew in",
        'slang': "Volunteering is actually trending right now which is pretty",
    },
    {
        'formal': "The arts sector has adapted to digital platforms for",
        'casual': "Artists are putting their stuff online now more than",
        'academic': "Cultural production has increasingly migrated to digital distribution channels for",
        'simple': "Artists share their work on the internet so more people",
        'literary': "The muse, ever adaptable, found new galleries in the luminous",
        'slang': "Artists are all going digital now because that's where the",
    },
    # === Science & Research ===
    {
        'formal': "Recent discoveries have advanced our understanding of planetary",
        'casual': "Scientists just found out some cool stuff about",
        'academic': "Novel empirical findings have substantively contributed to our understanding of",
        'simple': "Scientists learned new things about the planets in",
        'literary': "The cosmos yielded another of its ancient secrets as",
        'slang': "Scientists just dropped some crazy findings about space and",
    },
    {
        'formal': "Gene therapy research has entered a promising new",
        'casual': "Gene therapy stuff is looking really promising these",
        'academic': "Therapeutic genomic interventions have progressed into a substantively novel developmental",
        'simple': "Doctors are finding new ways to fix sickness using",
        'literary': "Within the spiraling code of life, healers now inscribe their",
        'slang': "Gene therapy research is going off right now like they",
    },
    {
        'formal': "Climate research has produced increasingly alarming projections for",
        'casual': "Climate scientists keep saying things are getting worse and",
        'academic': "Climatological modeling yields progressively more severe projections for",
        'simple': "Scientists say the weather will get worse because of",
        'literary': "The earth's fever rises, and those who chart its pulse warn of",
        'slang': "Climate research keeps dropping bad news and honestly it's",
    },
    {
        'formal': "Archaeological findings have revised our understanding of ancient",
        'casual': "They found some old stuff that changes what we thought about",
        'academic': "Recent archaeological excavations have necessitated revision of prevailing theories regarding",
        'simple': "People who dig up old things found something that changes",
        'literary': "Beneath layers of forgotten time, the earth surrendered relics that",
        'slang': "Archaeologists just found something that changes everything we thought about",
    },
    {
        'formal': "Quantum computing has achieved several significant milestones in",
        'casual': "Quantum computers are actually getting somewhere now with",
        'academic': "Quantum computational systems have attained multiple notable benchmarks in",
        'simple': "Very special computers are getting better at solving hard",
        'literary': "At the frontier of possibility, the quantum machine hums with",
        'slang': "Quantum computing is actually making moves now and it's",
    },
    # === Transportation ===
    {
        'formal': "Electric vehicle adoption has accelerated substantially across major",
        'casual': "So many people are buying electric cars now in",
        'academic': "Battery electric vehicle market penetration has increased markedly across principal",
        'simple': "More people are buying cars that run on electricity",
        'literary': "Silent chariots now glide where engines once roared across",
        'slang': "EVs are taking over the roads like everyone's switching to",
    },
    {
        'formal': "Public transportation infrastructure requires significant investment to",
        'casual': "Buses and trains need way more money to actually",
        'academic': "Mass transit systems necessitate substantial capital allocation to address",
        'simple': "Cities need to spend more money on buses and trains",
        'literary': "The arteries of the city cry out for renewal as",
        'slang': "Public transit is so underfunded it's embarrassing like they need",
    },
    {
        'formal': "Aviation safety standards have improved markedly over the past",
        'casual': "Flying has gotten so much safer compared to how it",
        'academic': "Aeronautical safety protocols have demonstrated measurable improvement across the previous",
        'simple': "Airplanes are much safer now than they were",
        'literary': "The silver birds that cross our skies now carry their precious cargo with",
        'slang': "Flying is way safer now than it used to be which is",
    },
    {
        'formal': "Traffic congestion presents a significant challenge for urban",
        'casual': "Traffic is such a nightmare in every big",
        'academic': "Vehicular congestion constitutes a persistent impediment to urban mobility and",
        'simple': "Too many cars on the road make it hard to",
        'literary': "Rivers of steel choke the city's veins as commuters",
        'slang': "Traffic is literally the worst thing about living in a",
    },
    {
        'formal': "Autonomous vehicle technology continues to undergo rigorous",
        'casual': "Self-driving cars are still being tested and they keep",
        'academic': "Autonomous vehicular systems remain subject to extensive validation and",
        'simple': "Cars that drive themselves are still being tested to make",
        'literary': "The driverless carriage, that marvel of our age, inches closer to",
        'slang': "Self-driving cars are still kinda sketchy but they keep",
    },
    # === Daily Life ===
    {
        'formal': "Morning routines significantly influence productivity throughout the",
        'casual': "What you do in the morning really sets up your",
        'academic': "Matutinal behavioral patterns demonstrably modulate subsequent daily productivity",
        'simple': "Having a good morning helps you do well all",
        'literary': "The morning hours, golden with possibility, shape the clay of",
        'slang': "Your morning routine literally determines if your day is gonna be",
    },
    {
        'formal': "Pet ownership has been associated with numerous health",
        'casual': "Having a pet is actually really good for your",
        'academic': "Companion animal ownership correlates positively with multiple indices of",
        'simple': "Having a pet can make you feel happier and",
        'literary': "In the loyal gaze of a beloved creature, the heart finds",
        'slang': "Having a pet is honestly the best thing for your",
    },
    {
        'formal': "Reading habits among young adults have shifted toward digital",
        'casual': "Young people read on their phones more than actual",
        'academic': "Literacy practices among young adult cohorts have transitioned predominantly to",
        'simple': "Young people read on screens more than paper",
        'literary': "The page, once turned by hand, now glows beneath the fingertips of",
        'slang': "Nobody reads actual books anymore it's all on their",
    },
    {
        'formal': "Household energy consumption can be reduced through simple",
        'casual': "You can save on energy bills just by doing some",
        'academic': "Residential energy expenditure is amenable to reduction via straightforward",
        'simple': "You can use less electricity at home if you",
        'literary': "With small acts of stewardship, the home becomes a vessel of",
        'slang': "Saving energy at home is actually so easy you just gotta",
    },
    {
        'formal': "Time management is essential for achieving a balanced",
        'casual': "Managing your time is so important if you want a",
        'academic': "Temporal resource allocation is fundamental to maintaining equilibrium between professional and",
        'simple': "Planning your time well helps you do everything you",
        'literary': "Time, that most precious currency, rewards those who spend it with",
        'slang': "Time management is literally the key to not losing your",
    },
]


# --- Helper functions for extracting prompts by register ---

def get_prompts(register):
    """Get all prompts for a single register."""
    return [ps[register] for ps in PROMPT_SETS]


def get_prompt_pairs(reg_a, reg_b):
    """Get matched prompt pairs for two registers."""
    return [(ps[reg_a], ps[reg_b]) for ps in PROMPT_SETS]
