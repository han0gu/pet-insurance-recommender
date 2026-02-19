from langchain_core.documents import Document

chunk = Document(
    page_content=('이 계약의 보상책임액 손해액 × 다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액\n'
 '② 이 계약이 의무보험이 아니고 다른 의무보험이 있는 경우에는 다른 의무보험에서 보상되는 금액(피 보험자가 가입을 하지 않은 경우에는 '
 '보상될 것으로 추정되는 금액)을 차감한 금액을 손해액으로 간주하여 제1항에 의한 보상할 금액을 결정합니다. ③ 피보험자가 다른 계약에 '
 '대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한 지급보험금 결정에는 영향을 미치지 않습니다.\n'
 '제7조(손해방지의무)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 27},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000145',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
