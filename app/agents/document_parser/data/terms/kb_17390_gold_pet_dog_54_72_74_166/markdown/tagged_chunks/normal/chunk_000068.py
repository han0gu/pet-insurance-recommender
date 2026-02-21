from langchain_core.documents import Document

chunk = Document(
    page_content=('급사유에 관해서는 원래대로 지급합니다.- \n'
 '비례 보상 예시예 시| 보험기간 중 | 직업의 변경으로 위험이 증가(상해급수 2급)되었으나, |  | 1급 → 이를 |\n'
 '| --- | --- | --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000068',
              'chunk_char_len': 116,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
