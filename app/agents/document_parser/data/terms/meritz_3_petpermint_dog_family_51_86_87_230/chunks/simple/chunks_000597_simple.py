from langchain_core.documents import Document

chunk = Document(
    page_content=('손해액 × 다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액\n'
 '\uf000 이 특별약관이 의무보험이 아니고 다른 의무보험이 있는 경우에는 다른 의무보험에서 보상되는 금액(피보험자가 가 입을 하지 않은 '
 '경우에는 보상될 것으로 추정되는 금액)을 차감한 금액을 손해액으로 간주하여 제1항에 의한 보상할 금액을 결정합니다. \uf000 '
 '피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 따른 지급보험금 결정에는 영향을 미치지 않습니다.\n'
 '제8조(손해방지의무)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 178},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000597',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
