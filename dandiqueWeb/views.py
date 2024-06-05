import random
import string
import time

from django.shortcuts import render
from .demo import search_query

def search(request):
    search_text = ''
    if request.method == 'POST':
        search_text = request.POST.get('query', '')
        print(f'Search method called with search text: {search_text}')

        if search_text == '':
            return render(request, 'search.html')

        # Simulate a delay
        time.sleep(1)

        # Get search result
        results = search_query(search_text)

        # Generate random search results
        # results = [{'title': ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)),
        #             'passage': ''.join(random.choices(string.ascii_uppercase + string.digits, k=616))} for _ in
        #            range(10)]

        # If 'lucky' is true, only return one result
        if 'lucky' in request.POST:
            results = [results[0]]

        return render(request, 'search.html', {'results': results, 'query': search_text})

    return render(request, 'search.html')
