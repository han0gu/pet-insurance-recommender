from langchain_core.documents import Document

chunk = Document(
    page_content=('여 각각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 손\n'
 '해를 보상합니다.다른 계약이 없을 때\n'
 '피보험자가 이 계약의 지급보험금\n'
 '×부담한 장례비용# 다른 계약이 없는 것으로 하여별\n'
 '각각 계산한 지급보험금의 합계액 약\n'
 '\uf000 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 관특| 의한 지급보험금 결정에는 | 영향을 '
 '미치지 않습니다. |\n'
 '| --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000651',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
