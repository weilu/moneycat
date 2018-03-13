import csv
from bs4 import BeautifulSoup


def extract():
    with open('categories.csv', 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['category', 'subcategory'])

        soup = BeautifulSoup(open('mint.html'), 'html.parser')
        categories = soup.findAll('li', {'class': 'isL1'})
        for category in categories:
            category_name = category.find('a').text
            subcategories = category.findAll('li', id=lambda x: x and x.startswith('menu-category-'))
            for sub in subcategories:
                csv_writer.writerow([category_name, sub.text])


extract()
