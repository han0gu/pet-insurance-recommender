from langchain_core.documents import Document

chunk = Document(
    page_content=('따라 손해를 보상합니다. 이 특별약관과 다른 계약이 모두\n'
 '의무보험인 경우에도 같습니다.# 이 특별약관의 보상책임액손해액 × 다른 계약이 없는 것으로 하여 각각 계산한\n'
 '보상책임액의 합계액\uf000 이 특별약관이 의무보험이 아니고 다른 의무보험이 있는\n'
 '경우에는 다른 의무보험에서 보상되는 금액(피보험자가 가\n'
 '입을 하지 않은 경우에는 보상될 것으로 추정되는 금액)을\n'
 '차감한 금액을 손해액으로 간주하여 제1항에 의한 보상할\n'
 '금액을 결정합니다.\n'
 '\uf000 피보험자가 다른 계약에 대하여 보험금 청구를 포기한'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000494',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
