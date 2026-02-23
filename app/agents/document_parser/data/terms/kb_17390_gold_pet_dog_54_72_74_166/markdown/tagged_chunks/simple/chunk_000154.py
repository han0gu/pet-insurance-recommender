from langchain_core.documents import Document

chunk = Document(
    page_content=('합니다)를 통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서류를 영수증으\n'
 '로 대신합니다.66 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)|  |\n'
 '| --- |\n'
 '| 용 어 풀 이 납입기일 |\n'
 '# 계약자가 제2회 이후의 보험료를 납입하기로 한 날을 말합니다.- 제27조(보험료의 자동대출납입)\n'
 '- \uf000 계약자는 제28조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)\n'
 '- 에 따른 보험료의 납입최고(독촉)기간이 지나기 전까지 회사가 정한 방법에 따라'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000154',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
