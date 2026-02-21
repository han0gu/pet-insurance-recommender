from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>제2조(보험금 지급에 관한 세부규정)<br>\uf000 제1조(보험금의 지급사유)의 "
 '환경성질환입원일당은 같은 질병의 치료를 목적으로<br>특<br>2회 이상 입원한 경우 이를 1회 입원으로 보아 각 입원일수를 더합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000687',
              'chunk_char_len': 139,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
