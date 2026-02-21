from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계약자에게 지급합니다. 물\n'
 '- \uf000 보험의 목적이 다수인 경우 제1항은 보험의 목적별로 각각 적용합니다.\n'
 '# 제26조(특별약관의 자동갱신)\uf000 이 특별약관의 【갱신계약】은"제도성 특별약관 - 보장특약 자동갱신(추가납입- 형) '
 '특별약관"에 의해 계약자의 선택에 따라 자동갱신으로 운영합니다. 성\n'
 '- \uf000 제1항에 의해 자동갱신을 적용할 경우 보험증권에 그 내용을 기재하여 드립니다. 특\n'
 '- 약\n'
 '제병KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 125- 125 -병도- 제27조(준용규정)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000715',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
