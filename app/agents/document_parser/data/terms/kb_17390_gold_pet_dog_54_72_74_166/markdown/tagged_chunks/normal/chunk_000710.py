from langchain_core.documents import Document

chunk = Document(
    page_content=('- 돌려드립니다.\n'
 '# \uf000# 제21조(조사)회사는 보험의 목적에 대한 위험상태를 조사하기 위하여 보험기간 중 언제든지- 피보험자의 시설과 업무내용을 '
 '조사할 수 있고 필요한 경우에는 그의 개선을 피\n'
 '- 보험자에게 요청할 수 있습니다.\n'
 '- \uf000 회사는 제1항에 따른 개선이 완료될 때까지 계약의 효력을 정지할 수 있습니다.\n'
 '- \uf000 회사는 이 계약의 중요사항과 관련된 범위 내에서는 보험기간 중 또는 회사에서\n'
 '- 정한 보험금 청구서류를 접수한 날부터 1년 이내에는 언제든지 피보험자의 회계\n'
 '- 장부를 열람할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000710',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
