from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 제1항에 따른 계약자 명의변경 신청 및 계약의 특별부활(효력회복) 청약을 승 낙합니다. ③ 회사는 제1항의 통지를 지정된 '
 '보험수익자에게 하여야 합니다. 다만, 회사는 법정상속 인이 보험수익자로 지정된 경우에는 제1항의 통지를 계약자에게 할 수 있습니다. ④ '
 '회사는 제1항의 통지를 계약이 해지된 날부터 7일 이내에 하여야 합니다. ⑤ 보험수익자는 통지를 받은 날(제3항에 따라 계약자에게 통지된 '
 '경우에는 계약자가 통 지를 받은 날을 말합니다)부터 15일 이내에 제1항의 절차를 이행할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 39},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000122',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
