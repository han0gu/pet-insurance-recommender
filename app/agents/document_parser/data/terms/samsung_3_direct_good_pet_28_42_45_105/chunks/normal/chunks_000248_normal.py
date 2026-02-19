from langchain_core.documents import Document

chunk = Document(
    page_content=('에는 보장을 해드립니다.\n'
 '제27조 (제2회 이후 보험료의 납입)\n'
 '계약자는 제2회 이후의 보험료를 납입기일까지 납입하여야 하며, 회사는 계약자가 보험 료를 납입한 경우에는 영수증을 발행하여 드립니다. '
 '다만, 금융회사(우체국을 포함합니 다)를 통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서류를 영수증으로 대신합 니다.\n'
 '<용어풀이>\n'
 '[납입기일]\n'
 '계약자가 제2회 이후의 보험료를 납입하기로 한 날을 말합니다.\n'
 '제 28조 (보험료의 자동대출납입)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 54},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000248',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
