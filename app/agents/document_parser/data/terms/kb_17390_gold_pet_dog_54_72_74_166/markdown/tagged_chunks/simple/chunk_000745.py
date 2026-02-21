from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 피보험자가 부담한 위탁비용 피보험자가 다른 계약에 | × 대하여 | 각각 계산한 지급보험금의 합계액 보험금 청구를 포기한 경우에도 '
 '회사의 제1항에 |\n'
 '![image](/image/placeholder)\n'
 '의한 지급보험금 결정에는 영향을 미치지 않습니다.\n'
 '용 어 풀 이 공제계약| 유사보험으로서 공제 | 사업을 실시하는 경영주체와 공제 계약자 사이에 체결되 |\n'
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000745',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
