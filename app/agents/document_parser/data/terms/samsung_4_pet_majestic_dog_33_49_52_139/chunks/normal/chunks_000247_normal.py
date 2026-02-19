from langchain_core.documents import Document

chunk = Document(
    page_content=('⑧ 제6항의 ‘청약일로부터 5년이 지나는 동안’이라 함은 제29조(보험료의 납입이 연체되 는 경우 납입최고(독촉)와 특별약관의 해지)에서 '
 '정한 특별약관의 해지가 발생하지 않 은 경우를 말합니다. ⑨ 제30조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))에서 '
 '정한 특별 약관의 부활이 이루어진 경우 부활을 청약한 날을 제6항의 청약일로 하여 적용합니 다.\n'
 '제20조 (청약의 철회)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 58},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000247',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
