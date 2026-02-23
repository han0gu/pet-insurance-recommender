from langchain_core.documents import Document

chunk = Document(
    page_content=('페이지 등을 통해 공시합니다.# 제10조 (환급금의 중도인출)① 계약자는 계약일부터 2년 이후에 계약이 유효한 경우 매보험년도마다 4회에 '
 '한하여\n'
 '해당시점의 적립부분 해약환급금(“보험료 및 해약환급금 산출방법서”에 따라서 중\n'
 '도인출금은 인출시점에 차감되며, 기본계약 해약환급금이 적립부분 해약환급금보다\n'
 '적은 경우에는 기본계약 해약환급금을 한도로 함)의 80%한도 내에서 인출할 수 있습\n'
 '니다. 다만, 이 약관에서 정한 보험계약대출이 있는 경우에는 그 원금과 이자합계액을\n'
 '한도에서 공제한 후의 잔액을 기준으로 합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000036',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
