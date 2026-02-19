from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 제1항에 따른 계약자 명의변경 신청 및 특별약관의 특별부활(효력회복) 청약을 승낙하며, 특별약관은 청약한 때부터 '
 '특별부활(효력회복) 됩니다. ③ 회사는 제1항의 통지를 지정된 보험수익자에게 하여야 합니다. 다만, 회사는 법정상속 인이 보험수익자로 '
 '지정된 경우에는 제1항의 통지를 계약자에게 할 수 있습니다.\n'
 '<용어풀이>\n'
 '[법정상속인]\n'
 '피상속인의 사망에 의하여 민법의 규정에 의한 상속순위에 따라 상속받는 자를 말합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 107},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000626',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
