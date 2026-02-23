from langchain_core.documents import Document

chunk = Document(
    page_content=('| 7) 한 다리에 가관절이 남아 뚜렷한 장해를 남긴 때 | 20 |\n'
 '| 8) 한 다리에 가관절이 남아 약간의 장해를 남긴 때 | 10 |\n'
 '| 9) 한 다리의 뼈에 기형을 남긴 때 | 5 |\n'
 '| 10) 한 다리가 5cm 이상 짧아지거나 길어진 때 | 30 |\n'
 '| 11) 한 다리가 3cm 이상 짧아지거나 길어진 때 | 15 |\n'
 '| 12) 한 다리가 1cm 이상 짧아지거나 길어진 때 | 5 |\n'
 '| 나. | 장해판정기준 |\n'
 '| --- | --- |\n'
 '1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것이 기능장해의 원인이 되'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000798',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
