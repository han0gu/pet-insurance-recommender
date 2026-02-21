from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자 또는 피보험자가 알리지 않은 경우 회사가 알고 있는 최종<br>의 주소 또는 연락처로 등기우편 등 우편물에 대한 기록이 '
 '남는 방법으로 회사가 알린<br>사항은 일반적으로 도달에 필요한 기간이 지난 때에는 계약자 또는 피보험자에게 도달<br>한 것으로 '
 "봅니다.</p><br><p id='94' data-category='paragraph' style='font-size:14px'>【계약 "
 "후 알릴 의무】</p><br><p id='95' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000262',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
