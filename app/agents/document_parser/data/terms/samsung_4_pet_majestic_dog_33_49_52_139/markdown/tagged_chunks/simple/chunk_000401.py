from langchain_core.documents import Document

chunk = Document(
    page_content=('② 피보험자가 보험기간 중에 이 특별약관에서 보장하지 않는 사유로 사망하였을 경우에\n'
 '는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 회사가 적립한 사망당시\n'
 '이 특별약관의 계약자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은\n'
 '더 이상 효력이 없습니다.- \n'
 '- 88 -2. 질병 관련 특별약관2-1. 특정법정감염병 진단비 특별약관# 제1관 일반사항① 제2관 개별사항에서 정하지 않은 사항은 '
 '특별약관의 일반사항을 적용합니다. 단, 특별\n'
 '약관 일반사항 제7조(보험금을 지급하지 않는 사유)는 제외합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000401',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
