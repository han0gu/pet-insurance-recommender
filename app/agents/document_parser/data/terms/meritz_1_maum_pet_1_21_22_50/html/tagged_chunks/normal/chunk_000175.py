from langchain_core.documents import Document

chunk = Document(
    page_content=("환급)</h1><br><p id='92' data-category='paragraph' style='font-size:14px'>① 이 "
 "계약이 무효, 효력상실, 해지 또는 소멸된 때에는 다음과 같이 보험료를 돌려드립니<br>다.</p><br><p id='93' "
 "data-category='list' style='font-size:14px'>1"),
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
 'indexing': {'chunk_id': 'chunk_000175',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
