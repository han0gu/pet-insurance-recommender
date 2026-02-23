from langchain_core.documents import Document

chunk = Document(
    page_content=('고(독촉)하며, 이 납입최고(독촉)기간 안에 보험료가 납입되지 않은 경우 납입최고(독\n'
 '촉)기간이 끝나는 날의 다음날 갱신 계약을 해제합니다.\n'
 '② 회사는 납입최고(독촉)기간 안에 발생한 사고에 대하여 약정한 보험금을 지급합니다.\n'
 '이 경우 계약자는 즉시 갱신계약 보험료를 납입하여야 합니다. 만약, 이 보험료를 납\n'
 '입하지 않으면 회사는 지급할 보험금에서 이를 차감할 수 있습니다.-'),
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
 'indexing': {'chunk_id': 'chunk_000697',
              'chunk_char_len': 211,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
