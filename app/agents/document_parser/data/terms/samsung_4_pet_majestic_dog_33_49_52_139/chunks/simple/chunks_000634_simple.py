from langchain_core.documents import Document

chunk = Document(
    page_content=('② 이 계약의 보험기간 종료 후 계약자가 재가입을 원하는 경우 계약자는 재가입 시점에 서 회사가 판매하는 동일하거나 객관적이고 합리적인 '
 '범위내에서 기존 계약내용에 상 응한 반려동물보험 상품(보험업감독규정 제1-2조(정의)에서 정한 장기손해보험에 한 하며 이하「반려동물보험 '
 '상품」이라 합니다)으로 가입을 할 수 있으며, 회사는 이를 거절할 수 없습니다. 다만, 재가입 계약이 직전계약보다 보장내용 및 범위 등이 '
 '확대 된 경우 확대된 내용에 대해 회사는 재가입 시점의 인수기준에 따라 승낙하거나 일부 보장을 제한할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 108},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000634',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
