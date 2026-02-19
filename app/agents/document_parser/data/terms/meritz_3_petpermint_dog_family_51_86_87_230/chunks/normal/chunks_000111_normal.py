from langchain_core.documents import Document

chunk = Document(
    page_content=('제24조(보험나이 등)\n'
 '\uf000 이 약관에서의 피보험자의 나이는 보험나이를 기준으로 합니다. 다만, 제22조(계약의 무효) 제1항 제2호의 경우에 는 실제 '
 '만 나이를 적용합니다. \uf000 제1항의 보험나이는 계약일 현재 피보험자의 실제 만 나 이를 기준으로 6개월 미만의 끝수는 버리고 '
 '6개월 이상의 끝수는 1년으로 하여 계산하며, 이후 매년 계약해당일에 나 이가 증가하는 것으로 합니다. \uf000 피보험자의 나이 또는 '
 '성별에 관한 청약서상 기재사항이'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 73},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000111',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
