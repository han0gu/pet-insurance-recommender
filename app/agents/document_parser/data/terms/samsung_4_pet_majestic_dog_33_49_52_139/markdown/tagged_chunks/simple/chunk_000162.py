from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 보험수익자가 고의로 피보험자를 해친 경우. 다만, 그 보험수익자가 보험금의 일부\n'
 '- 보험수익자인 경우에는 다른 보험수익자에 대한 보험금은 지급합니다.\n'
 '- 3. 계약자가 고의로 피보험자를 해친 경우\n'
 '- 4. 피보험자의 임신, 출산(제왕절개를 포함합니다), 산후기. 그러나 회사가 보장하는\n'
 '- 보험금 지급사유와 보험계약일로부터 2년이 지난 후에 발생한 습관성 유산, 불임\n'
 '- 및 인공수정 관련 합병증으로 인한 경우에는 보험금을 지급합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000162',
              'chunk_char_len': 248,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
