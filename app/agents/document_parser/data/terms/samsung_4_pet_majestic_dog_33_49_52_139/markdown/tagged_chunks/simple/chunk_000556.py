from langchain_core.documents import Document

chunk = Document(
    page_content=('# ∙ 예시3- 피보험자가 이물질제거(구토유도약물) 시행 당일 부담한 의료비 : 10만원\n'
 '- 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함) 특별약관 지급금액 : 4.9만원\n'
 '- 보험금 미지급- \n'
 '⑥ 제1항의 「연간」이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의\n'
 '기간을 의미합니다.제 2조 (보험금 지급에 관한 세부규정)보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 '
 '못\n'
 '할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000556',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
