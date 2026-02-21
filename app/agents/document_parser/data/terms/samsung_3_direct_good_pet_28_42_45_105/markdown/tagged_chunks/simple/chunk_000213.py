from langchain_core.documents import Document

chunk = Document(
    page_content=('- 단계약에서 진단을 받지 않은 경우라도 상해로 보험금 지급사유가 발생하는 경우\n'
 '에는 보장을 해드립니다.# 제27조 (제2회 이후 보험료의 납입)계약자는 제2회 이후의 보험료를 납입기일까지 납입하여야 하며, 회사는 '
 '계약자가 보험\n'
 '료를 납입한 경우에는 영수증을 발행하여 드립니다. 다만, 금융회사(우체국을 포함합니\n'
 '다)를 통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서류를 영수증으로 대신합'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000213',
              'chunk_char_len': 222,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
