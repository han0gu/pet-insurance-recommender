from langchain_core.documents import Document

chunk = Document(
    page_content=('② 납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그 다음 날 까지로 합니다. ③ 타인을 위한 계약의 경우 '
 '특정된 타인에게도 제1항 및 제2항에 따른 내용을 알려 드 립니다. ④ 보험료 납입이 연체중이라도 특별약관의 해지 전에 발생한 보험금 '
 '지급사유에 대하여 회사는 보상합니다. ⑤ 회사가 제1항에 의한 납입최고(독촉) 등을 전자문서로 안내하고자 할 경우에는 계약자'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 106},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000613',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
