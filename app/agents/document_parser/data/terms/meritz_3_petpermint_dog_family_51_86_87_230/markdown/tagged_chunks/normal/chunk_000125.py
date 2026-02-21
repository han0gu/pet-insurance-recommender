from langchain_core.documents import Document

chunk = Document(
    page_content=('의 원금과 이자를 차감합니다.\n'
 '\uf000 회사는 보험수익자에게 보험계약대출 사실을 통지할 수\n'
 '있습니다.# 제37조(배당금의 지급)회사는 이 계약에 대하여 계약자에게 배당금을 지급하지 않\n'
 '습니다.# 제38조(중도인출)\uf000 계약자는 보장개시일부터 2년 이상 지난 유효한 계약으\n'
 '로서 계약자의 요청이 있는 경우에 한하여 보험연도 기준\n'
 '연4회에 한하여 중도인출 할 수 있습니다.\n'
 '\uf000 제1항의 중도인출금은 계약자가 요청한 시점에서 계산된\n'
 '기본계약 해약환급금과 기본계약 적립부분 해약환급금 중\n'
 '적은 금액의 80% 범위 내에서 신청할 수 있습니다. 중도인'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000125',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
