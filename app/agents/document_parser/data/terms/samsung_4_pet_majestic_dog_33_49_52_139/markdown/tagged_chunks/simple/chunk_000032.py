from langchain_core.documents import Document

chunk = Document(
    page_content=('- 로 계산한 금액을 보험금에 더하여 지급합니다. 그러나 계약자, 피보험자 또는 보험수\n'
 '- 익자의 책임있는 사유로 지급이 지연된 때에는 그 해당기간에 대한 이자는 더하여 지\n'
 '- 급하지 않습니다.\n'
 '- ⑥ 계약자, 피보험자 또는 보험수익자는 제18조(알릴 의무 위반의 효과) 및 제2항의 보험\n'
 '- 금 지급사유조사와 관련하여 의료기관, 국민건강보험공단, 경찰서 등 관공서에 대한\n'
 '- 회사의 서면에 의한 조사요청에 동의하여야 합니다. 다만, 정당한 사유없이 이에 동의'),
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
 'indexing': {'chunk_id': 'chunk_000032',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
