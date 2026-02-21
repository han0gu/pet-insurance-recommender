from langchain_core.documents import Document

chunk = Document(
    page_content=('| 7) 한다리에 가관절이 남아 뚜렷한 장해를 남긴 때 | 20 |\n'
 '| 8) 한다리에 가관절이 남아 약간의 장해를 남긴 때 | 10 |\n'
 '| 9) 한다리의 뼈에 기형을 남긴 때 | 5 |\n'
 '| 10) 한 다리가 5cm 이상 짧아지거나 길어진 때 | 3 0 |\n'
 '| 11) 한 다리가 3cm 이상 짧아지거나 길어진 때 | 15 |\n'
 '| 12) 한 다리가 1cm 이상 짧아지거나 길어진 때 | 5 |\n'
 '# 나. 장해판정기준- 1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것\n'
 '- 이 기능장해의 원인이 되는 때에는 그 내고정물 등이'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000578',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
