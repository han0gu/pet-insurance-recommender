from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이 및 품종에 해당하는 보험금 및 보험료로 변경합니다. 다만, 반려동물의 나이 및 품\n'
 '- 종이 정정되기 이전에는 「나이 및 품종이 정정되기 전에 적용된 보험료율」의 「나\n'
 '- 이 및 품종이 정정된 후에 적용해야할 보험료율」에 대한 비율에 따라 보험금을 삭감\n'
 '- 하여 지급합니다.\n'
 '<예시안내>- [계약해당일 계산]\n'
 '최초계약일과 동일한 월, 일을 말합니다.\n'
 '계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000514',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
