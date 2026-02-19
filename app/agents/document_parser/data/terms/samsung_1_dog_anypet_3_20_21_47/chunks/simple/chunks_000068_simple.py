from langchain_core.documents import Document

chunk = Document(
    page_content=('의 책임있는 사유로 자동이체 또는 매출승인이 불가능한 경우에는 제1회 보험료 등이 납입되지 않 은 것으로 봅니다.\n'
 '⑤ 계약이 갱신되는 경우에는 제1항 내지 제3항에 의한 보장은 기존 계약에 의한 보장이 종료하는 때 부터 적용합니다.\n'
 '제22조(제2회 이후 보험료의 납입)\n'
 '계약자는 제2회 이후의 보험료를 납입기일까지 납입하여야 하며, 회사는 계약자가 보험료를 납입한 경 우에는 영수증을 발행하여 드립니다. '
 '다만, 금융회사(우체국을 포함합니다)를 통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서류를 영수증으로 대신합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 14},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000068',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
