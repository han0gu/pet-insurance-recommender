from langchain_core.documents import Document

chunk = Document(
    page_content=('한 다리에 가관절이 남아 뚜렷한 장해를 남긴 때</td><td>20</td></tr><tr><td>8) 한 다리에 가관절이 남아 약간의 '
 '장해를 남긴 때</td><td>10</td></tr><tr><td>9) 한 다리의 뼈에 기형을 남긴 '
 '때</td><td>5</td></tr><tr><td>10) 한 다리가 5cm 이상 짧아지거나 길어진 '
 '때</td><td>30</td></tr><tr><td>11) 한 다리가 3cm 이상 짧아지거나 길어진 '
 '때</td><td>15</td></tr><tr><td>12) 한 다리가 1cm 이상 짧아지거나'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001594',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
