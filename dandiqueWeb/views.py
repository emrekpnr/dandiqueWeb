import time

from django.shortcuts import render
from .demo import load_data, search_query, relevance_feedback

# Load data
embedder, doc_embeds, doc_texts = load_data()


def search(request):
    search_text = ''
    if request.method == 'POST':
        search_text = request.POST.get('query', '')
        print(f'Search method called with search text: {search_text}')

        if search_text == '':
            return render(request, 'search.html')

        # Simulate a delay
        time.sleep(0.5)

        # If 'lucky' is true, only return one result without any threshold
        if 'lucky' in request.POST:
            results = search_query(search_text, embedder, doc_embeds, doc_texts, threshold=0.0)[0]
            return render(request, 'search.html', {'results': [results], 'query': search_text})

        # Get search result
        results = search_query(search_text, embedder, doc_embeds, doc_texts)
        if results == []:
            return render(request, 'search.html', {'results': [{'title': '', 'answer': 'No results found'}], 'query': search_text})

        # search again with the most relevant document (Relevance Feedback)
        RF = 1  # Number of most relevant documents to use for relevance feedback
        most_relevant_docs = [result["answer"] for result in results[:RF]]
        results = relevance_feedback(search_text, most_relevant_docs, embedder, doc_embeds, doc_texts)
        if results == []:
            return render(request, 'search.html', {'results': [{'title': '', 'answer': 'No results found'}], 'query': search_text})

        return render(request, 'search.html', {'results': results, 'query': search_text})

    return render(request, 'search.html')
