from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1) 씹어먹는 기능과 말하는 기능 모두에 심한 장해를 남긴 때 | 100 |\n'
 '| 2) 씹어먹는 기능에 심한 장해를 남긴 때 | 80 |\n'
 '| 3) 말하는 기능에 심한 장해를 남긴 때 | 60 |\n'
 '| 4) 씹어먹는 기능과 말하는 기능 모두에 뚜렷한 장해를 남긴 때 | 40 |\n'
 '| 5) 씹어먹는 기능 또는 말하는 기능에 뚜렷한 장해를 남긴 때 | 20 |\n'
 '| 6) 씹어먹는 기능과 말하는 기능 모두에 약간의 장해를 남긴때 | 10 |\n'
 '| 7) 씹어먹는 기능 또는 말하는 기능에 약간의 장해를 남긴 때 | 5 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
