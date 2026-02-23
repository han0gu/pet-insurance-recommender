from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 더하여 지급합니다.\n'
 '제8조(특별약관의 보험료)이 특별약관의 보험료는없습니다.# 제9조(준용규정)제이 특별약관에서 정하지 않은 사항은 보통약관 및 '
 '사망보장특별약관을 따릅니다.도성특약KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 133- 133 -3. 보험료자동납입제1조(보험료의 '
 '납입)\n'
 '계약자는 제2회 이후의 보험료부터 이 특별약관에 따라 계약자의 거래은행 지정계좌# 를 이용하여 보험료를 자동 납입합니다.제2조(보험료의 '
 '영수)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000769',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
