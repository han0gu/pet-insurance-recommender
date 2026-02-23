from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는 외부지표금리와 운용자산이익률을 가중평균하여 산출된 공시기준이율에<br>향후 예상수익 등을 고려한 조정률을 적용하여 '
 '"보장성-1701 공시이율"을 결정<br>합니다.<br>3. "보장성-1701 공시이율"의 최저보증이율은 연단위 복리 0.2%를 '
 '적용합니다. 별<br>표<br>4'),
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
 'indexing': {'chunk_id': 'chunk_000064',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
