from langchain_core.documents import Document

chunk = Document(
    page_content=('【예금자보호제도】 예금보험공사에서 금융기관 등으로부터 미리 보험료를 받아 적립해 두었다가 금융기관이 예금보험공사에서 금융기관 등으로부터 '
 '미리 보험료를 받아 적립해 두었다가 금융기관이 경영악화나 파산 등 으로 예금을 지급할 수 없는 경우 해당 금융기관을 대신하여 예금자에게 '
 '보험금 또는 환급금을 지급함으로써 예금자를 보호하는 제도를 말합니다. 이 보험계약은 예금자보호법에 따라 해약환급금(또는 만기 시 '
 '보험금)에 기타지급금을 합한 금액이 1인당 "1억원까지"(본 보험회사의 여타 보호상품과 합산) 보호됩니다'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 20},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000111',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
