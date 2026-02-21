from langchain_core.documents import Document

chunk = Document(
    page_content=('력회복)을 청약할 수 있음을 보험수익자에게 통지하여야 합\n'
 '니다.\n'
 '\uf000 회사는 제1항에 따른 계약자 명의변경 신청 및 계약의\n'
 '특별부활(효력회복) 청약을 승낙합니다.\n'
 '\uf000 회사는 제1항의 통지를 지정된 보험수익자에게 하여야\n'
 '합니다. 다만, 회사는 법정상속인이 보험수익자로 지정된\n'
 '경우에는 제1항의 통지를 계약자에게 할 수 있습니다.\n'
 '\uf000 회사는 제1항의 통지를 계약이 해지된 날부터 7일 이내\n'
 '에 하여야 합니다.\n'
 '\uf000 보험수익자는 통지를 받은 날(제3항에 따라 계약자에게\n'
 '통지된 경우에는 계약자가 통지를 받은 날을 말합니다)부터'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000216',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
