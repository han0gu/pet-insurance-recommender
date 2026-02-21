from langchain_core.documents import Document

chunk = Document(
    page_content=('차이로 계약자가 추가로 납입하여야 할 (또는 반환받을) 금액이 발생할 수 있습\n'
 '니다.124 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)\uf000 회사는 제1항에 따라 위험이 감소된 경우에는 그 차액보험료를 '
 '돌려드리며, 위험# 이 증가된 경우에는 통지를 받은 날부터 1개월 이내에 보험료의 증액을 청구하거# 나 특별약관을 해지할 수 '
 '있습니다.제18조(알릴 의무 위반의 효과)\n'
 '\uf000 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생여부에 관계없이 그 사실- 을 안 날부터 1개월 이내에 이 특별약관을 '
 '해지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000700',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
