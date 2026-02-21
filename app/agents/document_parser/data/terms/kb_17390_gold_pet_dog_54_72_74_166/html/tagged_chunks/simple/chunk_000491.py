from langchain_core.documents import Document

chunk = Document(
    page_content=("지급합니다.</p><p id='212' data-category='paragraph' "
 "style='font-size:16px'>제2조(보험금 지급에 관한 세부규정)</p><br><p id='213' "
 "data-category='list' style='font-size:14px'>\uf000 제1조(보험금의 지급사유)의 골절진단비는 같은 "
 '상해를 직접적인 원인으로 2가지<br>이상의 골절 발생시에는 1회에 한하여 골절진단비를 지급합니다.<br>\uf000 보험수익자와 회사가 '
 '제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의<br>상<br>하지'),
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
 'indexing': {'chunk_id': 'chunk_000491',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
