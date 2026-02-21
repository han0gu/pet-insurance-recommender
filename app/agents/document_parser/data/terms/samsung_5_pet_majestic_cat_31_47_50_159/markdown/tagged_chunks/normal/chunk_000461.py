from langchain_core.documents import Document

chunk = Document(
    page_content=('제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회\n'
 '사가 전액 부담합니다.# 제5조 (피보험자의 범위)이 특별약관에서 피보험자라 함은 아래에 정한 보험증권에 기재된 피보험자 및 그 가족\n'
 '을 말합니다.- 1. 보험증권에 기재된 피보험자(이하 「피보험자 본인」 이라 합니다)\n'
 '- 2. 피보험자 본인의 가족관계등록상 또는 주민등록상에 기재된 배우자(이하 「배우자」 라\n'
 '- 합니다)\n'
 '- 3. 피보험자 본인 또는 배우자와 생계를 같이 하고, 보험증권에 기재된 주택의 주민등록'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000461',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
