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
]

df = pd.DataFrame(data)
df.to_csv("./data/generate_examples/nobel_questions_queries.csv", index=False)

df.head(), "CSV saved as nobel_questions_queries.csv"
