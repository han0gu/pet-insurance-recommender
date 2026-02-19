from langchain_core.documents import Document

chunk = Document(
    page_content=('② 해약환급금의 지급사유가 발생한 경우 계약자는 회사에 해약환급금을 청구하여야 하 며, 회사는 청구를 접수한 날부터 3영업일 이내에 '
 '해약환급금을 지급합니다. 해약환급 금 지급일까지의 기간에 대한 이자의 계산은 기본계약 약관의 [별표1] 보험금을 지급 할 때의 적립이율 '
 '계산을 따릅니다. ③ 제23조(특별약관 내용의 변경 등) 제1항에서 정한 보험가입금액 등을 감액할 경우 제 1항에 정한 해약환급금은 '
 '없거나 최초가입시 안내한 금액보다 적어질 수 있습니다. ④ 회사는 경과기간별 해약환급금에 관한 표를 계약자에게 제공하여 드립니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 65},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000322',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
