from langchain_core.documents import Document

chunk = Document(
    page_content=('할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있습니다.\n'
 '제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회\n'
 '사가 전액 부담합니다.# 제3조 (자기공명영상(MRI) 및 컴퓨터단층촬영(CT)의 정의)① 이 특별약관에 있어서 「자기공명영상(MRI)」 '
 '이라 함은 제1조(보험금의 지급사유)에서\n'
 '정한 수의사에 의하여 진단 및 치료가 필요하다고 인정된 경우로서 수의사의 관리 하\n'
 '에 자기공명영상(MRI)을 사용하는 촬영 의료행위를 말합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000632',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
