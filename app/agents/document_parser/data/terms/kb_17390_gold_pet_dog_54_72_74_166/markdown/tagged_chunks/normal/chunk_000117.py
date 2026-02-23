from langchain_core.documents import Document

chunk = Document(
    page_content=('- 고지의무 및 통지의무 위반의 효과\n'
 '- 계약의 취소 및 무효에 관한 사항\n'
 '- 해약환급금에 관한 사항(납부한 보험료보다 적거나 없을 수 있다는 사실 포함)\n'
 '- 민원처리 및 분쟁조정절차에 관한 사항 법\n'
 '- 만기시 자동갱신되는 보험계약의 경우 자동갱신의 조건 ㆍ\n'
 '- 저축성 보험계약의 공시이율 규정\n'
 '- 유배당 보험계약의 경우 계약자 배당에 관한 사항\n'
 '- 그 밖에 약관에 기재된 보험계약의 중요사항\n'
 '∙- 63 -KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 63사항관|  |\n'
 '| --- |'),
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
 'indexing': {'chunk_id': 'chunk_000117',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
