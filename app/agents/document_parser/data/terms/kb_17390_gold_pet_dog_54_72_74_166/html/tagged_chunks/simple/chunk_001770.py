from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="2">하부호흡기</td><td>고체 및 액체에 의한 '
 '폐렴</td><td>J69</td></tr><tr><td>폐기종</td><td>J43</td></tr><tr><td '
 'rowspan="3"></td><td>특정질환 기관지확장증</td><td>J47</td></tr><tr><td>달리 분류되지 않은 '
 '흉막삼출액</td><td>J90</td></tr><tr><td>흉막판</td><td>J92</td></tr><tr><td>흉막특정질환</td><td>기타 '
 '흉막의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001770',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
