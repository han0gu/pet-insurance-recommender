from langchain_core.documents import Document

chunk = Document(
    page_content=('제13조 (보험수익자의 지정)\n'
 '① 사망보험금의 경우는 보험수익자를 지정하지 않은 때에는 보험수익자를 피보험자의 법정상속인, 기타 보험금의 경우는 피보험자로 합니다. ② '
 '제1항에 따라 지정된 보험수익자가 보험기간 중에 사망한 때에는 계약자는 다시 보험 수익자를 지정할 수 있으며, 이 경우에 계약자가 '
 '보험수익자를 지정하지 않고 사망한 때에는 보험수익자의 법정상속인을 보험수익자로 합니다.\n'
 '<용어풀이>\n'
 '[법정상속인]\n'
 '피상속인의 사망에 의하여 민법의 규정에 의한 상속순위에 따라 상속받는 자를 말합니다.\n'
 '※ 상속순위'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 55},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000209',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
