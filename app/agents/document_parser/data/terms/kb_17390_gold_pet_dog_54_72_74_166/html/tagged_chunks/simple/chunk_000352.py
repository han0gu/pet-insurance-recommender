from langchain_core.documents import Document

chunk = Document(
    page_content=("지급합니다.</p><p id='4' data-category='paragraph' style='font-size:14px'>제2조(보험금 "
 '지급에 관한 세부규정)<br>\uf000 제1조(보험금의 지급사유)의 "사망"에는 보험기간에 다음 어느 하나의 사유가 '
 "발</p><br><p id='5' data-category='list' style='font-size:14px'>생한 경우를 "
 '포함합니다.<br>1'),
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
 'indexing': {'chunk_id': 'chunk_000352',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
