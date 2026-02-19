from langchain_core.documents import Document

chunk = Document(
    page_content=('【별표1】\n'
 '보험금을 지급할 때의 적립이율 계산 (제8조 제5항, 제10조 제3항 및 제35조 제2항 관련)\n'
 '구 분 | 기 간 | 지 급 이 자\n'
 '보장관련 보험금 (보통약관 제3조) (특별약관이 부가된 경우 특별약관의 보험금 포함) | 지급기일의 다음 날부터 30일 이내 기간 | '
 '보험계약대출이율\n'
 '지급기일의 31일 이후부터 60일 이내 기간 | 보험계약대출이율 + 가산이율(4.0%)\n'
 '지급기일의 61일 이후부터 90일 이내 기간 | 보험계약대출이율 + 가산이율(6.0%)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 199},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000688',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
