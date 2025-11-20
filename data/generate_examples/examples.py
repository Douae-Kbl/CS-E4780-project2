import pandas as pd

data = [
    {
        "question": "Who won the Nobel Prize in Physics in 2020?",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize {awardYear: 2020, category: "physics"}) RETURN s.knownName"""
    },
    {
        "question": "List all Nobel laureates in Chemistry.",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize {category: "chemistry"}) RETURN DISTINCT s.knownName"""
    },
    {
        "question": "Which laureates won more than one Nobel Prize?",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize) WITH s, COUNT(p) as prizes WHERE prizes > 1 RETURN s.knownName, prizes"""
    },
    {
        "question": "Find Nobel laureates born before 1900.",
        "cypher": """MATCH (s:Scholar) WHERE s.birthDate < "1900-01-01" RETURN s.knownName, s.birthDate"""
    },
    {
        "question": "What did Marie Curie win?",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE s.knownName CONTAINS "Curie" AND s.knownName CONTAINS "Marie" RETURN p.category, p.awardYear"""
    },
    {
        "question": "Who won the Nobel Prize in peace in 1962?",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize {awardYear: 1962, category: "peace"}) RETURN s.knownName"""
    },
    {
        "question": "List all Nobel Peace Prize winners.",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize {category: "peace"}) RETURN DISTINCT s.knownName"""
    },
    {
        "question": "How many laureates are female?",
        "cypher": """MATCH (s:Scholar) WHERE s.gender = "female" RETURN COUNT(s) AS female_laureates"""
    },
    {
        "question": "Find laureates who won the Nobel Prize in Medicine in 2001.",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize {awardYear: 2001, category: "medicine"}) RETURN s.knownName"""
    },
    {
        "question": "Show Nobel laureates along with the amount of their prize.",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize) RETURN s.knownName, p.prizeAmount"""
    },
    {
        "question": "Show top 10 Nobel laureates along with the amount of their prize.",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize) RETURN s.knownName, p.prizeAmount ORDER BY p.prizeAmount DESC LIMIT 10"""
    },
    {
        "question": "Which laureates died before 1950?",
        "cypher": """MATCH (s:Scholar) WHERE s.deathDate < "1950-01-01" RETURN s.knownName, s.deathDate"""
    },
    {
        "question": "List Nobel laureates who won in Economics.",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize {category: "economics"}) RETURN DISTINCT s.knownName"""
    },
    {
        "question": "Which year had the most Nobel Prize awards?",
        "cypher": """MATCH (p:Prize) RETURN p.awardYear, COUNT(p) as award_count ORDER BY award_count DESC LIMIT 1"""
    },
    {
        "question": "Find Nobel laureates born after 1980.",
        "cypher": """MATCH (s:Scholar) WHERE s.birthDate > "1980-01-01" RETURN s.knownName, s.birthDate"""
    },
    {
        "question": "Which Nobel laureate received the highest prize amount?",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize) RETURN s.knownName, p.prizeAmount ORDER BY p.prizeAmount DESC LIMIT 1"""
    },
    {
        "question": "Show prizes awarded in the Physics category.",
        "cypher": """MATCH (p:Prize {category: "physics"}) RETURN p.prize_id, p.awardYear"""
    },
    {
        "question": "Which laureate won the Nobel Peace Prize in 2010?",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize {awardYear: 2010, category: "peace"}) RETURN s.knownName"""
    },
    {
        "question": "Did Einstein win any prize?",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE s.knownName CONTAINS "Einstein" RETURN count(p) > 0"""
    },
    {
        "question": "Count the number of Nobel laureates.",
        "cypher": """MATCH (s:Scholar) RETURN COUNT(s) AS total_laureates"""
    },
    {
        "question": "List Nobel Prizes awarded in 2022.",
        "cypher": """MATCH (p:Prize {awardYear: 2022}) RETURN p.category, p.prize_id"""
    },
    {
        "question": "Find laureates who won in Physics and Chemistry.",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE p.category in ["physics", "chemistry"] RETURN DISTINCT s.knownName"""
    },
    {
        "question": "Show Nobel Prize categories.",
        "cypher": """MATCH (p:Prize) RETURN DISTINCT p.category"""
    },
    {
        "question": "Which laureates won in the year 2003?",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize {awardYear: 2003}) RETURN s.knownName, p.category"""
    },
    {
        "question": "Find laureates awarded after 2015.",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize) WHERE p.awardYear > 2015 RETURN s.knownName, p.awardYear, p.category"""
    },
    {
        "question": "List laureates who won a prize for chemistry before 1950.",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize {category: "chemistry"}) WHERE p.awardYear < 1950 RETURN s.knownName, p.awardYear"""
    },
    {
        "question": "In which city was the last chemistry laureate born?",
        "cypher": """MATCH (c:City)<-[:BORN_IN]-(s:Scholar)-[:WON]->(p:Prize {category: "chemistry"}) RETURN s.knownName, c.name ORDER BY p.awardYear DESC LIMIT 1"""
    },
    {
        "question": "Which nobel laureates were born and died in the same city?",
        "cypher": """MATCH (c:City)<-[:BORN_IN]-(s:Scholar)-[:DIED_IN]->(c) RETURN s.knownName"""
    },
    {
        "question": "Which nobel laureates were born and died in the same country?",
        "cypher": """MATCH (cc:Country)<-[:IS_CITY_IN]-(c:City)<-[:BORN_IN]-(s:Scholar)-[:DIED_IN]->(c2:City)-[:IS_CITY_IN]->(cc) RETURN s.knownName, cc.name, c.name, c2.name"""
    },
    {
        "question": "Which nobel laureates were born and died on the same continent?",
        "cypher": """MATCH (con:Continent)<-[:IS_COUNTRY_IN]-(cc:Country)<-[:IS_CITY_IN]-(c:City)<-[:BORN_IN]-(s:Scholar)-[:DIED_IN]->(c2:City)-[:IS_CITY_IN]->(cc2:Country)-[:IS_COUNTRY_IN]->(con) RETURN s.knownName, con.name, cc.name, c.name, cc2.name, c2.name"""
    },
    {
        "question": "Which Nobel laureates were born in Germany?",
        "cypher": """MATCH (cc:Country)<-[:IS_CITY_IN]-(c:City)<-[:BORN_IN]-(s:Scholar) WHERE cc.name = "Germany" RETURN s.knownName"""
    },
    {
        "question": "Which Nobel laureates died in the USA?",
        "cypher": """MATCH (cc:Country)<-[:IS_CITY_IN]-(c:City)<-[:DIED_IN]-(s:Scholar) WHERE cc.name = "USA" RETURN s.knownName"""
    },
    {
        "question": "Which country has the most Nobel laureates?",
        "cypher": """MATCH (cc:Country)<-[:IS_CITY_IN]-(c:City)<-[:BORN_IN]-(s:Scholar) RETURN cc.name, COUNT(s) AS laureate_count ORDER BY laureate_count DESC LIMIT 1"""
    },
    {
        "question": "What are the laureates associated with more than one institute?",
        "cypher": """MATCH (s:Scholar)-[:AFFILIATED_WITH]->(i:Institution) WITH s, COUNT(i) AS institute_count WHERE institute_count > 1 RETURN s.knownName"""
    },
    {
        "question": "What is the institution with most Nobel laureates affiliated?",
        "cypher": """MATCH (i:Institution)<-[:AFFILIATED_WITH]-(s:Scholar) RETURN i.name, COUNT(s) AS laureate_count ORDER BY laureate_count DESC LIMIT 1"""
    },
    {
        "question": "What is the total prize won by scholars affiliated with the institution with most scholars affiliated?",
        "cypher": """MATCH (i:Institution)<-[:AFFILIATED_WITH]-(s:Scholar)-[:WON]->(p:Prize) WITH i, SUM(p.prizeAmount) AS total_prize RETURN i.name, total_prize ORDER BY total_prize DESC LIMIT 1"""
    },
    {
        "question": "What is the youngest age a scholar won a Nobel Prize?",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize) RETURN s.knownName, p.awardYear - CAST(substring(s.birthDate, 1, 4), "INT") AS age ORDER BY age ASC LIMIT 1"""
    },
    {
        "question": "What is the oldest age a scholar won a Nobel Prize?",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize) RETURN s.knownName, p.awardYear - CAST(substring(s.birthDate, 1, 4), "INT") AS age ORDER BY age DESC LIMIT 1"""
    },
    {
        "question": "What is the average age a scholar won a Nobel Prize?",
        "cypher": """MATCH (s:Scholar)-[:WON]->(p:Prize) RETURN AVG(p.awardYear - CAST(substring(s.birthDate, 1, 4), "INT")) AS average_age"""
    },
    {
        "question": "Give me top 10 countries by average age of scholars winning Nobel Prizes.",
        "cypher": """MATCH (cc:Country)<-[:IS_CITY_IN]-(c:City)<-[:BORN_IN]-(s:Scholar)-[:WON]->(p:Prize) RETURN cc.name, AVG(p.awardYear - CAST(substring(s.birthDate, 1, 4), "INT")) AS average_age ORDER BY average_age ASC LIMIT 10"""
    },
    {
        
    }
    
]

df = pd.DataFrame(data)
df.to_csv("./data/generate_examples/nobel_questions_queries.csv", index=False)

df.head(), "CSV saved as nobel_questions_queries.csv"
