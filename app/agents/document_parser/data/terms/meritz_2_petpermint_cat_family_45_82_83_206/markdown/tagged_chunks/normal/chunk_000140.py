from langchain_core.documents import Document

chunk = Document(
    page_content=('우에는 예금자보호법에서 정하는 바에 따라 그 지급을 보장\n'
 '합니다.# 【예금자보호제도】예금자보호제도란 예금보험공사가 평소에 금융기관으로\n'
 '부터 보험료를 받아 기금을 적립한 후, 금융기관이 영업\n'
 '정지나 파산 등으로 예금을 지급할 수 없게되면 금융기\n'
 '관을 대신하여 예금을 지급하는 제도를 말합니다.이 보험계약은 예금자보호법에 따라 해약환급금(또는 만\n'
 '기 시 보험금)에 기타지급금을 합한 금액이 1인당 “1억\n'
 '원까지”(본 보험회사의 여타 보호상품과 합산) 보호됩\n'
 '니다. 이와 별도로 본 보험회사 보호상품의 사고보험금'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000140',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
