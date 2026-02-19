from langchain_core.documents import Document

chunk = Document(
    page_content=('예금자보호제도란 예금보험공사가 평소에 금융기관으로 부터 보험료를 받아 기금을 적립한 후, 금융기관이 영업 정지나 파산 등으로 예금을 '
 '지급할 수 없게되면 금융기 관을 대신하여 예금을 지급하는 제도를 말합니다. 이 보험계약은 예금자보호법에 따라 해약환급금(또는 만 기 시 '
 '보험금)에 기타지급금을 합한 금액이 1인당 “1억 원까지”(본 보험회사의 여타 보호상품과 합산) 보호됩 니다. 이와 별도로 본 보험회사 '
 '보호상품의 사고보험금 을 합산한 금액이 1인당 “1억원까지” 보호됩니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 86},
 'term_type': 'basic',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000177',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
