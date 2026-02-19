from langchain_core.documents import Document

chunk = Document(
    page_content=('2027년 4월 1일 | -\n'
 '제12조(주소변경통지)\n'
 '\uf000 계약자(보험수익자가 계약자와 다른 경우 보험수익자를 포함합니다)는 주소 또는 연락처가 변경된 경우에는 지체없 이 그 '
 '변경내용을 회사에 알려야 합니다. \uf000 제1항에서 정한대로 계약자 또는 보험수익자가 변경내용 을 알리지 않은 경우에는 계약자 또는 '
 '보험수익자가 회사에 알린 최종의 주소 또는 연락처로 등기우편 등 우편물에 대 한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로 '
 '도달에 필요한 기간이 지난 때에 계약자 또는 보험수익자에 게 도달된 것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 60},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000046',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
