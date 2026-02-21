from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이 계약에서 보장하는 위험과 동일한 위험을 보장하는 계약을 다른 보험자와 체결하</p><footer id='90' "
 "style='font-size:14px'>- 27 -</footer><p id='91' data-category='paragraph' "
 "style='font-size:14px'>고자 할 때 또는 이와 같은 계약이 있음을 알았을 때</p><br><p id='92' "
 "data-category='paragraph' style='font-size:14px'>3"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000260',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
