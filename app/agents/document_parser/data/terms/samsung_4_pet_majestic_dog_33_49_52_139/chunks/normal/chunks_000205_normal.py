from langchain_core.documents import Document

chunk = Document(
    page_content=('제11조 (보험금 받는 방법의 변경)\n'
 '① 계약자(보험금 지급사유 발생 후에는 보험수익자)는 회사의 사업방법서에서 정한 바에 따라 보험금의 전부 또는 일부에 대하여 나누어 '
 '지급받거나 일시에 지급받는 방법으 로 변경할 수 있습니다. ② 회사는 제1항에 따라 일시에 지급할 금액을 나누어 지급하는 경우에는 나중에 '
 '지급할 금액에 대하여 평균공시이율을 연단위 복리로 계산한 금액을 더하며, 나누어 지급할 금액을 일시에 지급하는 경우에는 평균공시이율을 '
 '연단위 복리로 할인한 금액을 지급 합니다.\n'
 '<예시안내>'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 54},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000205',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
