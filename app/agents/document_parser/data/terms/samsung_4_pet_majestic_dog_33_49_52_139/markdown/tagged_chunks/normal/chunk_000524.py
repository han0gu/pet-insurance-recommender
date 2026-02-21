from langchain_core.documents import Document

chunk = Document(
    page_content=('중단할 것을 요구하는 경우, 회사는 전화 (음성녹음) 방법으로 전환하여 제1항에 따른\n'
 '납입최고(독촉) 등을 실시할 것4.전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 있는 기능을 갖출 것\n'
 '5.제3호 및 제4호의 내용에 관한 사항을 계약자에게 안내할 것⑦ 제1항에 따라 특별약관이 해지된 경우에는 이 특별약관의 해약환급금을 '
 '계약자에게\n'
 '지급합니다.# 제22조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))- ① 제21조(보험료의 납입이 연체되는 경우 '
 '납입최고(독촉)와 특별약관의 해지)에 따라 특'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000524',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
