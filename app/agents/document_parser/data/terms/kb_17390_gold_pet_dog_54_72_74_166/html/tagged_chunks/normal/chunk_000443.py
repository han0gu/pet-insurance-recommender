from langchain_core.documents import Document

chunk = Document(
    page_content=("후에 성형수술이 가능하다는 진단을 받은 경우에는 그</p><br><h1 id='132' style='font-size:16px'>진단으로 "
 "대체할 수 있습니다.</h1><br><p id='133' data-category='list' "
 "style='font-size:16px'>제2조(보험금 지급에 관한 세부규정)<br>\uf000 제1조(보험금의 지급사유)의 "
 '상해흉터복원수술비Ⅱ(안면부)는 하나의 사고로 동<br>일부위에 대한 성형수술을 2회 이상 받은 경우에는 최초로 받은 수술에 '
 '대해서만<br>지급합니다.<br>\uf000 보험수익자와 회사가'),
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
 'indexing': {'chunk_id': 'chunk_000443',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
