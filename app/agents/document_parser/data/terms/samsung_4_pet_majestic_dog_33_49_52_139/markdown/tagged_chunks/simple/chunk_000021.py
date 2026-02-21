from langchain_core.documents import Document

chunk = Document(
    page_content=('- 및 인공수정 관련 합병증으로 인한 경우에는 보험금을 지급합니다.\n'
 '<용어풀이># [습관성 유산, 불임 및 인공수정 관련 합병증]한국표준질병∙사인분류상의 N96~N98에 해당하는 질병을 말합니다.# 5. '
 '전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동② 회사는 다른 약정이 없으면 피보험자가 직업, 직무 또는 동호회 활동목적으로 아래에\n'
 '열거된 행위로 인하여 제3조(보험금의 지급사유)의 보험금 지급사유가 발생한 때에는'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000021',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
