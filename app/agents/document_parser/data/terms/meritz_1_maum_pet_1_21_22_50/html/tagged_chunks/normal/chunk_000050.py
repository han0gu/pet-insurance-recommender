from langchain_core.documents import Document

chunk = Document(
    page_content=("id='56' style='font-size:14px'>제8조(보험금의 청구)</h1><br><p id='57' "
 "data-category='paragraph' style='font-size:14px'>① 보험수익자는 다음의 서류를 제출하고 보험금을 "
 "청구하여야 합니다.</p><br><p id='58' data-category='list' style='font-size:14px'>1"),
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
 'indexing': {'chunk_id': 'chunk_000050',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
