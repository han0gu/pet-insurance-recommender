from langchain_core.documents import Document

chunk = Document(
    page_content=('10. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태 11. 대한민국 이외의 지역에서 발생한 사고 및 '
 '손해\n'
 '② 회사는 다음에 정한 사유 중 하나에 의해 피보험자가 부담한 치료비, 비용 또는 손해에 대해서는 보험금을 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 4},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000023',
              'chunk_char_len': 147,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
