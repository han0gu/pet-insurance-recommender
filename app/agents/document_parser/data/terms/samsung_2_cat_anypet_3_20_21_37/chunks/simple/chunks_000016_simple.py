from langchain_core.documents import Document

chunk = Document(
    page_content=('. 수의사의 치료상의 과오로 생긴 상해 또는 질병, 수의사 자격이 없는 자의 치료행위로 인한 비 용 및 그로 인하여 가중된 비용 11. '
 '국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태 12. 대한민국 이외의 지역에서 발생한 사고 및 손해 13. '
 '회사는 아래의 치료비, 비용 또는 손해는 보상하지 아니합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000016',
              'chunk_char_len': 187,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
