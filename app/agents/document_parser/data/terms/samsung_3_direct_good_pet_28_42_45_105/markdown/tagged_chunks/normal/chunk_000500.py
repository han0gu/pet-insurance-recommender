from langchain_core.documents import Document

chunk = Document(
    page_content=('니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 보상\n'
 '책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 손해를 보상합니다. 이 계약\n'
 '과 다른 계약이 모두 의무보험인 경우에도 같습니다.이 계약의 보상책임액손해액 × 다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의\n'
 '합계액- ② 이 계약이 의무보험이 아니고 다른 의무보험이 있는 경우에는 다른 의무보험에서 보\n'
 '- 상되는 금액(피보험자가 가입을 하지 않은 경우에는 보상될 것으로 추정되는 금액)을'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000500',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
