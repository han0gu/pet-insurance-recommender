from langchain_core.documents import Document

chunk = Document(
    page_content=('【자동갱신 적용대상 특별약관】\n'
 '･ 갱신형 펫퍼민트 반려견 배상책임보장 특별약관\n'
 '제2조(자동갱신 적용대상 계약의 자동갱신)\n'
 '\uf000 보장계약이 다음 각 호의 조건을 충족하고, 보장계약이 끝나는 날의 전일까지 계약자로부터 별도의 의사표시가 없 을 때에는 '
 '종전의 자동갱신 적용대상 계약(이하「갱신전 보 장계약」이라 합니다)이 끝나는 날의 다음날(이하「갱신 일」이라 합니다)에 동일한 '
 '보장내용으로 갱신되는 것으로 합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 189},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000645',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
