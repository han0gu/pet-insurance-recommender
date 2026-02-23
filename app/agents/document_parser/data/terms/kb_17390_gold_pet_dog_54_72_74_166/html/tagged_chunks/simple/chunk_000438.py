from langchain_core.documents import Document

chunk = Document(
    page_content=("id='123' style='font-size:14px'>제1조(보험금의 지급사유)</h1><br><p id='124' "
 "data-category='paragraph' style='font-size:14px'>\uf000 회사는 피보험자가 이 특별약관의 "
 '보험기간 중에 급격하고도 우연한 외래의 사고<br>로 병원 또는 의원(한방병원 또는 한의원을 포함합니다)등에서 치료를 받고 '
 '그<br>직접적인 결과로 인하여 안면부에 외형상의 반흔(흉터)이나 추상장해, 신체의 기<br>형이나 기능장해가 발생하여 그 원상회복을 '
 '목적으로 사고일로부터 2년'),
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
 'indexing': {'chunk_id': 'chunk_000438',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
