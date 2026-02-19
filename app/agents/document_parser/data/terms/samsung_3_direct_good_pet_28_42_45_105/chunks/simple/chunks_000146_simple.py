from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[예금자보호제도]\n'
 '예금자보호제도란 예금보험공사에서 금융기관 등으로부터 미리 보험료를 받아 적립해 두었다가 금 융기관이 경영악화나 파산 등으로 예금을 지급할 '
 '수 없는 경우 해당 금융기관을 대신하여 예금자 에게 보험금 또는 환급금을 지급함으로써 예금자를 보호하는 제도를 말합니다. 본 회사에 있는 '
 '모 든 예금보호 대상 금융상품의 해약환급금(또는 만기 시 보험금)에 기타 지급금을 합한 금액이 1인 당 "1억원까지" 예금자 보호가 '
 '됩니다. 이와 별도로 본 회사 보호상품의 사고보험금을 합산한 금액'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 42},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000146',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
