import random
import string
import time

from django.shortcuts import render


def search(request):
    search_text = ''
    if request.method == 'POST':
        search_text = request.POST.get('query', '')
        print(f'Search method called with search text: {search_text}')

        if search_text == '':
            return render(request, 'search.html')

        # Simulate a delay
        time.sleep(1)

        # Generate random search results
        results = [{'title': ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)),
                    'passage': ''.join(random.choices(string.ascii_uppercase + string.digits, k=616))} for _ in
                   range(10)]

        return render(request, 'search.html', {'results': results, 'query': search_text})

    return render(request, 'search.html')
