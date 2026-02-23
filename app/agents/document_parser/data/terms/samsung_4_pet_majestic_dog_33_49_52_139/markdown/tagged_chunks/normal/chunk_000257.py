from langchain_core.documents import Document

chunk = Document(
    page_content=('체납처분 절차에 따라 회사는 채권자에게 해약환급금을 지급하게 됩니다.- ② 회사는 제1항에 따른 계약자 명의변경 신청 및 특별약관의 '
 '특별부활(효력회복) 청약을\n'
 '- 승낙합니다.\n'
 '- ③ 회사는 제1항의 통지를 지정된 보험수익자에게 하여야 합니다. 다만, 회사는 법정상속\n'
 '- 인이 보험수익자로 지정된 경우에는 제1항의 통지를 계약자에게 할 수 있습니다.\n'
 '- ④ 회사는 제1항의 통지를 계약이 해지된 날부터 7일 이내에 하여야 합니다.\n'
 '- ⑤ 보험수익자는 통지를 받은 날(제3항에 따라 계약자에게 통지된 경우에는 계약자가 통'),
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
 'indexing': {'chunk_id': 'chunk_000257',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
