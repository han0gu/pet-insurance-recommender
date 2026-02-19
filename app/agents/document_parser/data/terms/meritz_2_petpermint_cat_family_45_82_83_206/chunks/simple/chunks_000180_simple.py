from langchain_core.documents import Document

chunk = Document(
    page_content=('이 보험계약은 예금자보호법에 따라 해약환급금(또는 만 기 시 보험금)에 기타지급금을 합한 금액이 1인당 “1억 원까지”(본 보험회사의 '
 '여타 보호상품과 합산) 보호됩 니다. 이와 별도로 본 보험회사 보호상품의 사고보험금 을 합산한 금액이 1인당 “1억원까지” 보호됩니다. '
 '다 만, 계약자 및 보험료납부자가 법인인 보험계약의 경우 에는 보호되지 않습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 82},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000180',
              'chunk_char_len': 196,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
