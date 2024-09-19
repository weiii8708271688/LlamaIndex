import arxiv


def search_paper(query: str) -> str:
    try:
        search = arxiv.Search(query=query, max_results = 5)
        results = []
        for result in search.results():
            results.append(f"Title: {result.title}, ID: {result.entry_id.split('/')[-1]}")
        return "\n".join(results) if results else "No papers found."
    except Exception as e:
        return f"Error searching for papers with query '{query}': {str(e)}"
    


print(search_paper('AI'))