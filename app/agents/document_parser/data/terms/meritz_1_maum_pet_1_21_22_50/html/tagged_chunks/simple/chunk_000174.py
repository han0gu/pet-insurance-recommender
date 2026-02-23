from langchain_core.documents import Document

chunk = Document(
    page_content=("해지되거나 제2항의 규정에 따라 계약이 효력을 잃는 경</p><footer id='89' style='font-size:14px'>- "
 "18 -</footer><p id='90' data-category='paragraph' style='font-size:14px'>우에 "
 "회사는 제33조(보험료의 환급)에 따른 보험료를 계약자에게 지급합니다.</p><h1 id='91' "
 "style='font-size:14px'>제33조(보험료의 환급)</h1><br><p id='92' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000174',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
