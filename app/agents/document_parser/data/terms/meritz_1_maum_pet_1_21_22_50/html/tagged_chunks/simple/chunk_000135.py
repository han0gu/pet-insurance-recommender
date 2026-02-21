from langchain_core.documents import Document

chunk = Document(
    page_content=("14 -</footer><p id='38' data-category='paragraph' "
 "style='font-size:14px'>제24조(계약의 소멸)</p><br><h1 id='39' "
 "style='font-size:14px'>반려동물의 사망 등으로 인하여 이 약관에서 규정하는 보험금 지급사유가 더 이상 발생할<br>수 "
 "없는 경우에는 이 계약은 그 때부터 효력이 없습니다.</h1><p id='40' data-category='paragraph' "
 "style='font-size:14px'>제5관 보험료의 납입</p><h1"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000135',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
