from langchain_core.documents import Document

chunk = Document(
    page_content=('여 각각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 손\n'
 '해를 보상합니다. 이 계약과 다른 계약이 모두 의무보험인 경우에도 같습니다.이 계약의 보상책임액\n'
 '손해액 ×\n'
 '다른계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액![image](/image/placeholder)\n'
 '\uf000 이 계약이 의무보험이 아니고 다른 의무보험이 있는 경우에는 다른 의무보험에서\n'
 '보상되는 금액(피보험자가 가입을 하지 않은 경우에는 보상될 것으로 추정되는\n'
 '금액)을 차감한 금액을 손해액으로 간주하여 제1항에 의한 보상할 금액을 결정합\n'
 '니다.'),
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
 'indexing': {'chunk_id': 'chunk_000681',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
