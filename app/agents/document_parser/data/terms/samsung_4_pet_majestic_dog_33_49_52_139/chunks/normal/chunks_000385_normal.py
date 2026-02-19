from langchain_core.documents import Document

chunk = Document(
    page_content=('수술을 받거나 같은 종류의 상해 통원 수술을 2회 이상 받은 경우에는 하나의 상해 통원 수술비만 지급합니다.\n'
 '제3조 (보험금 지급에 관한 세부규정)\n'
 '보험수익자와 회사가 제2조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못 할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 '
 '제3자의 의견에 따를 수 있습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하며, 보험금 지 급사유 '
 '판정에 드는 의료비용은 회사가 전액 부담합니다.\n'
 '<관련법규>'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 75},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000385',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
